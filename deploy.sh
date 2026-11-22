#!/usr/bin/env bash
# Build and deploy the multi-agent RAG service to Google Cloud Run.
#
# Reads credentials from .env (gitignored, never committed, excluded from the build
# context by .gcloudignore/.dockerignore). Keys reach Cloud Run through an env-vars YAML
# file written to a private temp file and deleted on exit, so no key appears in this
# script, in the image, or on the gcloud command line (and so not in shell history or
# `ps` output).
#
# Prerequisites (generated artifacts, baked into the image):
#   python -m src.rag.ingest       -> chroma_index/
#   python -m src.data.build_db    -> data/papers.db
set -euo pipefail

# Never let gcloud stop for an interactive prompt (e.g. "API not enabled... enable and
# retry? (y/N)"). Some calls hide stderr, so a prompt would hang the script silently.
export CLOUDSDK_CORE_DISABLE_PROMPTS=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

fail() {
    echo "error: $*" >&2
    exit 1
}

warn() {
    echo "warning: $*" >&2
}

# ------------------------------------------------------------------ 1. .env
[[ -f .env ]] || fail ".env not found. Copy .env.example to .env and fill it in."

set -a
# shellcheck disable=SC1091
source .env
set +a

# A value still holding the .env.example placeholder counts as unset.
is_placeholder() {
    local value="$1"
    [[ -z "${value}" || "${value}" == *replace-me* || "${value}" == *"<"*">"* || "${value}" == your-* ]]
}

# The router skips any provider without a key and DuckDuckGo stands in for Tavily, so a
# deploy needs the project id and at least one LLM key. Missing keys only reduce routing
# choice (and lose the MCP path for search, without Tavily), so they warn instead of failing.
REQUIRED_VARS=(GCP_PROJECT_ID)
LLM_KEY_VARS=(GOOGLE_API_KEY NVIDIA_API_KEY OPENROUTER_API_KEY)
OPTIONAL_VARS=(
    TAVILY_API_KEY
    NVIDIA_MODEL_FAST NVIDIA_MODEL_STRONG
    GEMINI_MODEL_FAST GEMINI_MODEL_STRONG
    OPENROUTER_FREE_MODELS
)

missing_vars=()
for var_name in "${REQUIRED_VARS[@]}"; do
    if is_placeholder "${!var_name:-}"; then
        missing_vars+=("${var_name}")
    fi
done
if (( ${#missing_vars[@]} > 0 )); then
    fail "required variables missing or still placeholders in .env: ${missing_vars[*]}"
fi

present_llm_keys=()
for var_name in "${LLM_KEY_VARS[@]}"; do
    if is_placeholder "${!var_name:-}"; then
        warn "${var_name} is not set; the router will skip that provider."
    else
        present_llm_keys+=("${var_name}")
    fi
done
if (( ${#present_llm_keys[@]} == 0 )); then
    fail "no LLM key set in .env: set at least one of ${LLM_KEY_VARS[*]}"
fi

for var_name in "${OPTIONAL_VARS[@]}"; do
    if is_placeholder "${!var_name:-}"; then
        warn "${var_name} is not set; the router falls back to its built-in default for it."
    fi
done

# ------------------------------------------------------- 2. baked artifacts
[[ -d chroma_index ]] || fail "chroma_index/ is missing. Build it first: python -m src.rag.ingest"
[[ -n "$(ls -A chroma_index 2>/dev/null)" ]] || fail "chroma_index/ is empty. Rebuild it: python -m src.rag.ingest"
[[ -f chroma_index/chroma.sqlite3 ]] || fail "chroma_index/chroma.sqlite3 not found; the index looks incomplete. Rebuild: python -m src.rag.ingest"
[[ -s data/papers.db ]] || fail "data/papers.db is missing or empty. Build it first: python -m src.data.build_db"
[[ -f static/index.html ]] || fail "static/index.html is missing."
command -v gcloud >/dev/null 2>&1 || fail "gcloud CLI not found on PATH."
command -v python3 >/dev/null 2>&1 || fail "python3 not found on PATH (used to write the env-vars file)."

# --------------------------------------------------------------- settings
PROJECT_ID="${GCP_PROJECT_ID}"
REGION="${GCP_REGION:-us-west1}"
SERVICE_NAME="${CLOUD_RUN_SERVICE:-multi-agent-rag}"
AR_REPOSITORY="${AR_REPOSITORY:-cloud-run-images}"
IMAGE_TAG="$(date +%Y%m%d-%H%M%S)"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${AR_REPOSITORY}/${SERVICE_NAME}:${IMAGE_TAG}"

echo "==> Project ${PROJECT_ID}, region ${REGION}, service ${SERVICE_NAME}"

# -------------------------------------------------------------- 3. APIs
# Enabled before anything else touches the project: on a fresh project the guard's
# `gcloud run services describe` would otherwise fail (or prompt) on the disabled Run API.
echo "==> Enabling required APIs"
gcloud services enable \
    run.googleapis.com \
    cloudbuild.googleapis.com \
    artifactregistry.googleapis.com \
    compute.googleapis.com \
    --project "${PROJECT_ID}"

# ------------------------------------------- shared-project safety guard
# The GCP project is shared with Assignment 1 (Cloud Run service csci599-a1). Every
# resource this script creates has its own name (service, Artifact Registry repo, image),
# and IAM bindings are additive, so nothing of A1's is replaced. This guard makes that
# explicit: deploying refuses to touch an existing service that this script did not
# create (it lacks our label), so a typo in CLOUD_RUN_SERVICE cannot overwrite A1.
SERVICE_LABEL_KEY="assignment"
SERVICE_LABEL_VALUE="csci599-a2"
existing_label="$(gcloud run services describe "${SERVICE_NAME}" \
    --project "${PROJECT_ID}" --region "${REGION}" \
    --format="value(metadata.labels.${SERVICE_LABEL_KEY})" 2>/dev/null || true)"
if gcloud run services describe "${SERVICE_NAME}" --project "${PROJECT_ID}" --region "${REGION}" >/dev/null 2>&1; then
    [[ "${existing_label}" == "${SERVICE_LABEL_VALUE}" ]] \
        || fail "Cloud Run service '${SERVICE_NAME}' already exists and is not labelled ${SERVICE_LABEL_KEY}=${SERVICE_LABEL_VALUE}. Refusing to overwrite it (it may be Assignment 1). Set CLOUD_RUN_SERVICE to a new name."
    echo "    updating existing A2 service (label ${SERVICE_LABEL_KEY}=${SERVICE_LABEL_VALUE})"
else
    echo "    service does not exist yet; it will be created"
fi
echo "    index: $(du -sh chroma_index | cut -f1), papers.db: $(du -h data/papers.db | cut -f1)"

# ------------------------------------------------- 4. Artifact Registry repo
if ! gcloud artifacts repositories describe "${AR_REPOSITORY}" \
        --location "${REGION}" --project "${PROJECT_ID}" >/dev/null 2>&1; then
    echo "==> Creating Artifact Registry repository ${AR_REPOSITORY}"
    gcloud artifacts repositories create "${AR_REPOSITORY}" \
        --repository-format docker \
        --location "${REGION}" \
        --project "${PROJECT_ID}" \
        --description "Images for ${SERVICE_NAME}"
fi

# ---------------------------------------------------------- 5. build IAM
# Assignment 1 lesson: Cloud Build could not read its own source bucket until its
# service account was granted roles. New projects run builds as the Compute Engine
# default service account, which exists only once compute.googleapis.com is enabled
# (step 3 enables it). Bindings are idempotent; SKIP_IAM=1 skips this step, and a
# failure (e.g. caller lacks setIamPolicy) only warns so an already-configured project
# still deploys.
if [[ "${SKIP_IAM:-0}" != "1" ]]; then
    PROJECT_NUMBER="$(gcloud projects describe "${PROJECT_ID}" --format='value(projectNumber)')"
    BUILD_SERVICE_ACCOUNT="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"
    echo "==> Ensuring build roles for ${BUILD_SERVICE_ACCOUNT}"
    for role in roles/cloudbuild.builds.builder roles/storage.objectViewer \
                roles/artifactregistry.writer roles/logging.logWriter; do
        if ! gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
                --member "serviceAccount:${BUILD_SERVICE_ACCOUNT}" \
                --role "${role}" --condition None --quiet >/dev/null; then
            warn "could not grant ${role}; if the build fails with a permission error, grant it manually."
        fi
    done
fi

# --------------------------------------------------------------- 6. build
# The upload honours .gcloudignore, which keeps chroma_index/ and data/papers.db (both
# gitignored) and drops .env, .venv and the corpus PDFs.
echo "==> Building ${IMAGE} with Cloud Build"
gcloud builds submit --tag "${IMAGE}" --project "${PROJECT_ID}" .

# ------------------------------------------------------- 7. env-vars file
ENV_VARS_FILE="$(mktemp "${TMPDIR:-/tmp}/cloudrun-env.XXXXXX")"
chmod 600 "${ENV_VARS_FILE}"
cleanup_env_file() {
    rm -f "${ENV_VARS_FILE}"
}
trap cleanup_env_file EXIT

# JSON is valid YAML and json.dumps quotes every value safely. PORT is reserved by
# Cloud Run and is never set here. Paths point at the baked artifacts inside /app.
python3 - "${ENV_VARS_FILE}" <<'PY'
import json
import os
import sys

passthrough_names = [
    "NVIDIA_API_KEY", "NVIDIA_BASE_URL", "NVIDIA_MODEL_FAST", "NVIDIA_MODEL_STRONG",
    "GOOGLE_API_KEY", "GEMINI_MODEL_FAST", "GEMINI_MODEL_STRONG",
    "OPENROUTER_API_KEY", "OPENROUTER_BASE_URL", "OPENROUTER_FREE_MODELS",
    "TAVILY_API_KEY", "CHROMA_COLLECTION", "RETRIEVAL_K", "RERANK_MODEL", "RERANK_THREADS",
]
env_vars = {}
for name in passthrough_names:
    value = os.environ.get(name, "").strip()
    is_placeholder = (not value) or ("replace-me" in value) or (value.startswith("<") and value.endswith(">"))
    if not is_placeholder:
        env_vars[name] = value
env_vars["CHROMA_PERSIST_DIR"] = "/app/chroma_index"
env_vars["PAPERS_DB_PATH"] = "/app/data/papers.db"
env_vars["ANONYMIZED_TELEMETRY"] = "False"

with open(sys.argv[1], "w", encoding="utf-8") as env_file:
    json.dump(env_vars, env_file, indent=2)
print(f"    env-vars file: {len(env_vars)} variables ({', '.join(sorted(env_vars))})")
PY

# -------------------------------------------------------------- 8. deploy
# concurrency 5 per instance (each request may re-rank, ~0.3 GB and most of a vCPU) with max 3
# instances caps spend; min 0 scales to zero. 4 GiB leaves headroom over Chroma + both ONNX models.
# The service is stateless (each request is independent), so any instance can serve any
# request. --cpu-boost shortens cold starts (Chroma + ONNX load).
echo "==> Deploying ${SERVICE_NAME} to Cloud Run (${REGION})"
gcloud run deploy "${SERVICE_NAME}" \
    --image "${IMAGE}" \
    --project "${PROJECT_ID}" \
    --region "${REGION}" \
    --platform managed \
    --allow-unauthenticated \
    --memory 4Gi \
    --cpu 2 \
    --concurrency 5 \
    --timeout 300 \
    --min-instances 0 \
    --max-instances 3 \
    --cpu-boost \
    --labels "${SERVICE_LABEL_KEY}=${SERVICE_LABEL_VALUE}" \
    --env-vars-file "${ENV_VARS_FILE}"

SERVICE_URL="$(gcloud run services describe "${SERVICE_NAME}" \
    --project "${PROJECT_ID}" --region "${REGION}" --format='value(status.url)')"

echo
echo "==> Live URL: ${SERVICE_URL}"
echo "    Health (Cloud Run's frontend reserves /healthz, so use /health):"
echo "    curl -s ${SERVICE_URL}/health"
echo "    Stream smoke test:"
echo "    curl -N -X POST ${SERVICE_URL}/api/chat -H 'Content-Type: application/json' \\"
echo "         -d '{\"query\":\"How does ReAct interleave reasoning and acting?\",\"session_id\":\"smoke\"}'"
