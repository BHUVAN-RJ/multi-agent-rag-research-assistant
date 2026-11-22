"""Server, frontend and deploy-layer tests. No network, no keys.

TestClient is used without a `with` block so the lifespan warm-up (which may load
Chroma) doesn't run; warm_up_resources is tested directly instead.
"""

import asyncio
import json
import os
import re
import subprocess
import sys
import time
import types
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import main
from src import graph as graph_module

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = PROJECT_ROOT / "static"


async def stub_shaped_graph(query: str, session_id: str):
    """Copy of the original src/graph.py stub: the exact event shapes the real graph emits.

    Tests patch this in so they stay offline and deterministic after the real graph
    (which calls LLM providers) replaces the stub.
    """
    started_at = time.perf_counter()
    yield {
        "type": "trace",
        "data": {
            "kind": "agent_route", "node": "supervisor", "summary": "stub -> research",
            "detail": {"next_agent": "research", "reason": "stub graph"}, "ts": time.time(),
        },
    }
    yield {
        "type": "trace",
        "data": {
            "kind": "llm_route", "node": "research", "summary": "gemini / stub-model",
            "detail": {"provider": "gemini", "model": "stub-model", "dimension": "hybrid", "reason": "stub"},
            "ts": time.time(),
        },
    }
    answer_words = f"Stub answer for: {query} [1]".split(" ")
    for word in answer_words:
        await asyncio.sleep(0.01)
        yield {"type": "token", "data": {"text": word + " ", "node": "research"}}
    yield {
        "type": "citation",
        "data": {
            "id": "[1]", "source_type": "corpus", "title": "Stub Paper", "source_id": "0000.00000",
            "url": "https://arxiv.org/abs/0000.00000", "section": "Introduction", "page": 1,
            "snippet": "stub snippet",
        },
    }
    elapsed_ms = (time.perf_counter() - started_at) * 1000
    yield {"type": "done", "data": {"answer": " ".join(answer_words), "latency_ms": elapsed_ms, "ttft_ms": None}}


@pytest.fixture
def client(monkeypatch) -> TestClient:
    monkeypatch.setattr(graph_module, "stream_query", stub_shaped_graph)
    return TestClient(main.app)


def parse_sse_body(body: str) -> list[dict]:
    """Split an SSE body into frames: {"event", "data", "comments"} per blank-line block."""
    frames = []
    for block in body.split("\n\n"):
        if not block.strip():
            continue
        event_name = None
        data_lines = []
        comments = []
        for line in block.split("\n"):
            if line.startswith(":"):
                comments.append(line[1:].strip())
            elif line.startswith("event:"):
                event_name = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data_lines.append(line[len("data:"):].strip())
        data = json.loads("\n".join(data_lines)) if data_lines else None
        frames.append({"event": event_name, "data": data, "comments": comments})
    return frames


def post_chat(client: TestClient, payload: dict) -> tuple[int, dict, str]:
    with client.stream("POST", "/api/chat", json=payload) as response:
        body = "".join(response.iter_text())
        return response.status_code, dict(response.headers), body


def event_names(frames: list[dict]) -> list[str]:
    return [frame["event"] for frame in frames if frame["event"] is not None]


# ------------------------------------------------------------------ basics


def test_healthz_returns_ok(client):
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_health_alias_returns_ok(client):
    # /healthz is intercepted by Cloud Run's frontend; /health is the deployed probe.
    response = client.get("/health")
    assert response.json() == {"status": "ok"}


def test_root_serves_html_with_chat_form(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert '<form id="chat-form"' in response.text
    assert 'id="query-input"' in response.text


def test_static_assets_are_served(client):
    script_response = client.get("/static/app.js")
    style_response = client.get("/static/styles.css")
    assert script_response.status_code == 200
    assert "createSseParser" in script_response.text
    assert style_response.status_code == 200


# --------------------------------------------------------------- streaming


def test_chat_streams_stub_frames_in_order(client):
    status, headers, body = post_chat(client, {"query": "What is ReAct?", "session_id": "test-1"})
    assert status == 200
    assert headers["content-type"].startswith("text/event-stream")
    assert headers["cache-control"] == "no-cache"
    assert headers["x-accel-buffering"] == "no"

    frames = parse_sse_body(body)
    names = event_names(frames)
    # Stub order: two traces, one token per word, one citation, done.
    token_count = len("Stub answer for: What is ReAct? [1]".split(" "))
    assert names == ["trace", "trace"] + ["token"] * token_count + ["citation", "done"]

    first_trace = frames[0]["data"]
    assert first_trace["kind"] == "agent_route"
    assert frames[1]["data"]["kind"] == "llm_route"
    streamed_text = "".join(frame["data"]["text"] for frame in frames if frame["event"] == "token")
    assert "What is ReAct?" in streamed_text
    citation = next(frame["data"] for frame in frames if frame["event"] == "citation")
    assert citation["id"] == "[1]" and citation["source_type"] == "corpus"
    done = frames[-1]["data"]
    assert isinstance(done["latency_ms"], (int, float)) and done["latency_ms"] > 0
    # The stub reports ttft_ms=None; the server fills it from the first token it saw.
    assert isinstance(done["ttft_ms"], (int, float))
    assert done["ttft_ms"] <= done["latency_ms"]


def test_every_frame_is_event_then_single_data_line(client):
    _, _, body = post_chat(client, {"query": "multi\nline\nquery", "session_id": "s"})
    blocks = [block for block in body.split("\n\n") if block.strip()]
    for block in blocks:
        lines = block.split("\n")
        assert len(lines) == 2, block
        assert lines[0].startswith("event: ")
        assert lines[1].startswith("data: ")


@pytest.mark.parametrize(
    "payload",
    [
        {"query": "", "session_id": "s"},
        {"query": "   \n\t ", "session_id": "s"},
        {"query": "x" * (main.MAX_QUERY_CHARS + 1), "session_id": "s"},
        {"session_id": "s"},
        {"query": "ok", "session_id": "   "},
    ],
)
def test_invalid_input_is_rejected(client, payload):
    response = client.post("/api/chat", json=payload)
    assert response.status_code in (400, 422)


def test_query_at_max_length_is_accepted(client):
    status, _, body = post_chat(client, {"query": "x" * main.MAX_QUERY_CHARS, "session_id": "s"})
    assert status == 200
    assert event_names(parse_sse_body(body))[-1] == "done"


def test_session_id_is_passed_to_graph(client, monkeypatch):
    seen_arguments = []

    async def recording_graph(query, session_id):
        seen_arguments.append((query, session_id))
        yield {"type": "done", "data": {"answer": "", "latency_ms": 1.0, "ttft_ms": None}}

    monkeypatch.setattr(graph_module, "stream_query", recording_graph)
    post_chat(client, {"query": "  hello  ", "session_id": "abc-123"})
    assert seen_arguments == [("hello", "abc-123")]


def test_graph_raising_mid_stream_emits_error_then_done(client, monkeypatch):
    async def exploding_graph(query, session_id):
        yield {"type": "token", "data": {"text": "partial ", "node": "research"}}
        raise RuntimeError("provider exploded")

    monkeypatch.setattr(graph_module, "stream_query", exploding_graph)
    status, _, body = post_chat(client, {"query": "boom", "session_id": "s"})
    assert status == 200
    frames = parse_sse_body(body)
    assert event_names(frames) == ["token", "error", "done"]
    error_data = frames[1]["data"]
    assert "provider exploded" in error_data["message"]
    assert frames[2]["data"]["answer"] == "partial "


def test_graph_raising_before_first_payload_still_ends_with_done(client, monkeypatch):
    async def failing_graph(query, session_id):
        raise ValueError("bad state")
        yield  # pragma: no cover  (makes this an async generator)

    monkeypatch.setattr(graph_module, "stream_query", failing_graph)
    _, _, body = post_chat(client, {"query": "q", "session_id": "s"})
    assert event_names(parse_sse_body(body)) == ["error", "done"]


def test_graph_without_done_gets_synthetic_done(client, monkeypatch):
    async def graph_without_done(query, session_id):
        yield {"type": "token", "data": {"text": "hi", "node": "summary"}}

    monkeypatch.setattr(graph_module, "stream_query", graph_without_done)
    _, _, body = post_chat(client, {"query": "q", "session_id": "s"})
    frames = parse_sse_body(body)
    assert event_names(frames) == ["token", "done"]
    assert frames[-1]["data"]["answer"] == "hi"


def test_graph_done_data_is_passed_through(client, monkeypatch):
    done_data = {"answer": "A", "final_agent": "research", "plan": ["research"], "latency_ms": 5.0, "ttft_ms": 2.0}

    async def graph_with_rich_done(query, session_id):
        yield {"type": "done", "data": done_data}
        yield {"type": "token", "data": {"text": "after done must not be sent", "node": "x"}}

    monkeypatch.setattr(graph_module, "stream_query", graph_with_rich_done)
    _, _, body = post_chat(client, {"query": "q", "session_id": "s"})
    frames = parse_sse_body(body)
    assert event_names(frames) == ["done"]
    assert frames[0]["data"] == done_data


def test_heartbeat_comment_sent_while_graph_is_slow(client, monkeypatch):
    async def slow_graph(query, session_id):
        await asyncio.sleep(0.35)
        yield {"type": "token", "data": {"text": "late", "node": "research"}}
        yield {"type": "done", "data": {"answer": "late", "latency_ms": 350.0, "ttft_ms": 350.0}}

    monkeypatch.setattr(main, "SSE_HEARTBEAT_SECONDS", 0.05)
    monkeypatch.setattr(graph_module, "stream_query", slow_graph)
    _, _, body = post_chat(client, {"query": "q", "session_id": "s"})
    assert body.startswith(": ping\n\n")
    frames = parse_sse_body(body)
    ping_frames = [frame for frame in frames if frame["comments"] == ["ping"]]
    assert len(ping_frames) >= 3
    # Heartbeats never cancel the pending graph step: the late token still arrives.
    assert event_names(frames) == ["token", "done"]


# ----------------------------------------------------------------- warm-up


def test_warm_up_calls_retriever_warm_function(monkeypatch):
    calls = []
    fake_retriever = types.ModuleType("src.rag.retriever")
    fake_retriever.warm_up = lambda: calls.append("warm_up")
    monkeypatch.setitem(sys.modules, "src.rag.retriever", fake_retriever)
    assert main.warm_up_resources() is True
    assert calls == ["warm_up"]


def test_warm_up_survives_missing_or_broken_retriever(monkeypatch):
    # None in sys.modules makes `import` raise ImportError.
    monkeypatch.setitem(sys.modules, "src.rag.retriever", None)
    assert main.warm_up_resources() is False

    broken_retriever = types.ModuleType("src.rag.retriever")

    def exploding_warm():
        raise OSError("index missing")

    broken_retriever.warm = exploding_warm
    monkeypatch.setitem(sys.modules, "src.rag.retriever", broken_retriever)
    assert main.warm_up_resources() is False


# ---------------------------------------------------------------- frontend


FRAMEWORK_MARKERS = [
    "react", "vue", "angular", "svelte", "jquery", "preact", "alpine", "htmx",
    "lit-element", "ember", "backbone", "bootstrap", "tailwind",
]


def test_frontend_uses_no_external_framework_or_cdn():
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    script_sources = re.findall(r"<script[^>]*\bsrc=\"([^\"]+)\"", html)
    stylesheet_links = re.findall(r"<link[^>]*\bhref=\"([^\"]+)\"", html)
    for asset_url in script_sources + stylesheet_links:
        # Inline data: URIs (the favicon) load nothing from the network.
        is_local_asset = asset_url.startswith("/static/") or asset_url.startswith("data:")
        assert is_local_asset, f"external asset: {asset_url}"
    lowered_html = html.lower()
    for marker in FRAMEWORK_MARKERS:
        assert f"{marker}." not in lowered_html and f"/{marker}" not in lowered_html, marker
    for cdn_host in ["cdn.", "unpkg.com", "jsdelivr", "cdnjs", "googleapis.com"]:
        assert cdn_host not in lowered_html, cdn_host


def test_frontend_reads_post_stream_not_eventsource():
    script = (STATIC_DIR / "app.js").read_text(encoding="utf-8")
    assert "new EventSource" not in script
    assert "getReader()" in script
    assert 'fetch("/api/chat"' in script
    for event_name in ["token", "trace", "citation", "error", "done"]:
        assert f'frame.event === "{event_name}"' in script


def test_frontend_handles_stream_restart_and_trace_kinds():
    script = (STATIC_DIR / "app.js").read_text(encoding="utf-8")
    styles = (STATIC_DIR / "styles.css").read_text(encoding="utf-8")
    assert "stream_restarted === true" in script
    assert "handleStreamRestart" in script
    assert "switched provider, restarting answer" in script
    trace_kinds = [
        "agent_route", "llm_route", "llm_fallback", "rag_retrieval",
        "mcp_tool_call", "data_query", "agent_result", "error",
    ]
    for kind in trace_kinds:
        assert f".kind-{kind}" in styles, kind


def test_frontend_has_example_chips_and_reset():
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    assert 'id="reset-button"' in html
    for example_query in [
        "How does ReAct interleave reasoning and acting?",
        "What are the latest developments in AI agents this week?",
        "How many RAG papers were published per year since 2020?",
        "Summarize the key ideas of Self-RAG",
    ]:
        assert f'data-query="{example_query}"' in html


def test_frontend_js_has_no_arrow_callbacks():
    # House style: named functions only, no inline anonymous callbacks.
    script = (STATIC_DIR / "app.js").read_text(encoding="utf-8")
    assert "=>" not in script
    assert re.search(r"\(\s*function\s*\(", script) is None


# ------------------------------------------------------------ deploy layer


def read_dockerfile_instructions() -> list[str]:
    text = (PROJECT_ROOT / "Dockerfile").read_text(encoding="utf-8")
    joined = text.replace("\\\n", " ")
    return [line.strip() for line in joined.splitlines() if line.strip() and not line.strip().startswith("#")]


def test_dockerfile_is_multistage_non_root_and_binds_port():
    instructions = read_dockerfile_instructions()
    from_lines = [line for line in instructions if line.upper().startswith("FROM ")]
    assert len(from_lines) >= 2
    assert all("python:3.13-slim" in line for line in from_lines)

    user_lines = [line for line in instructions if line.upper().startswith("USER ")]
    assert user_lines, "no USER instruction"
    final_user = user_lines[-1].split()[1]
    assert final_user not in ("root", "0")

    cmd_lines = [line for line in instructions if line.upper().startswith("CMD ")]
    assert len(cmd_lines) == 1
    cmd_line = cmd_lines[0]
    assert "${PORT" in cmd_line or "$PORT" in cmd_line
    # Exec (JSON) form would not expand $PORT.
    assert not cmd_line[4:].lstrip().startswith("[") or "sh" in cmd_line
    assert "main:app" in cmd_line


def test_dockerfile_installs_deps_and_bakes_artifacts_and_model():
    instructions = read_dockerfile_instructions()
    text = "\n".join(instructions)
    assert "pip install --no-cache-dir -r requirements.txt" in text
    assert re.search(r"COPY .*chroma_index/", text)
    assert re.search(r"COPY .*data/papers\.db", text)
    assert "DefaultEmbeddingFunction" in text
    # The model download must happen after switching to the runtime user, so it lands
    # in the home directory that user reads at runtime.
    user_index = max(i for i, line in enumerate(instructions) if line.startswith("USER "))
    model_index = next(i for i, line in enumerate(instructions) if "DefaultEmbeddingFunction" in line)
    assert model_index > user_index
    assert "HOME=/home/app" in text


@pytest.mark.parametrize("ignore_file", [".dockerignore", ".gcloudignore"])
def test_ignore_files_drop_secrets_and_corpus_but_keep_artifacts(ignore_file):
    patterns = [
        line.strip()
        for line in (PROJECT_ROOT / ignore_file).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    assert ".env" in patterns
    assert ".venv/" in patterns
    assert "data/corpus/" in patterns
    for pattern in patterns:
        assert not pattern.startswith("chroma_index"), pattern
        assert "papers.db" not in pattern, pattern
        assert "include:.gitignore" not in pattern


def test_deploy_script_is_valid_and_complete():
    script_path = PROJECT_ROOT / "deploy.sh"
    syntax_check = subprocess.run(["bash", "-n", str(script_path)], capture_output=True, text=True)
    assert syntax_check.returncode == 0, syntax_check.stderr
    script = script_path.read_text(encoding="utf-8")
    for required_var in ["GCP_PROJECT_ID", "GOOGLE_API_KEY", "NVIDIA_API_KEY", "OPENROUTER_API_KEY", "TAVILY_API_KEY"]:
        assert required_var in script
    for flag in [
        "--memory 4Gi", "--cpu 2", "--concurrency 5", "--timeout 300",
        "--min-instances 0", "--max-instances 3", "--allow-unauthenticated", "--env-vars-file",
    ]:
        assert flag in script, flag
    assert "--set-env-vars" not in script
    assert "chroma_index" in script and "data/papers.db" in script
    assert "trap cleanup_env_file EXIT" in script
    # No literal key may live in the script.
    assert re.search(r"(nvapi-|sk-or-v1-|tvly-|AIza)[A-Za-z0-9_-]{8,}", script) is None


def test_deploy_script_fails_loudly_without_artifacts(tmp_path):
    # Run a copy of deploy.sh in an empty dir with a valid-looking .env: it must stop at
    # the artifact check, before any gcloud call.
    script_copy = tmp_path / "deploy.sh"
    script_copy.write_text((PROJECT_ROOT / "deploy.sh").read_text(encoding="utf-8"), encoding="utf-8")
    (tmp_path / ".env").write_text(
        "GCP_PROJECT_ID=proj-test\nGOOGLE_API_KEY=k1\nNVIDIA_API_KEY=k2\n"
        "OPENROUTER_API_KEY=k3\nTAVILY_API_KEY=k4\n",
        encoding="utf-8",
    )
    result = subprocess.run(["bash", str(script_copy)], capture_output=True, text=True, cwd=tmp_path)
    assert result.returncode != 0
    assert "chroma_index/ is missing" in result.stderr


def test_deploy_script_fails_when_every_llm_key_is_a_placeholder(tmp_path):
    script_copy = tmp_path / "deploy.sh"
    script_copy.write_text((PROJECT_ROOT / "deploy.sh").read_text(encoding="utf-8"), encoding="utf-8")
    (tmp_path / ".env").write_text(
        "GCP_PROJECT_ID=proj-test\nGOOGLE_API_KEY=replace-me\nNVIDIA_API_KEY=\n"
        "OPENROUTER_API_KEY=replace-me\nTAVILY_API_KEY=k4\n",
        encoding="utf-8",
    )
    result = subprocess.run(["bash", str(script_copy)], capture_output=True, text=True, cwd=tmp_path)
    assert result.returncode != 0
    assert "no LLM key set" in result.stderr


def test_deploy_script_accepts_a_single_llm_key_without_tavily(tmp_path):
    # README promises the app works with one LLM key and DuckDuckGo; deploy must not demand more.
    script_copy = tmp_path / "deploy.sh"
    script_copy.write_text((PROJECT_ROOT / "deploy.sh").read_text(encoding="utf-8"), encoding="utf-8")
    (tmp_path / ".env").write_text("GCP_PROJECT_ID=proj-test\nGOOGLE_API_KEY=k1\n", encoding="utf-8")
    # The test process may hold real keys (src.config loads .env); the script must see only this .env.
    key_names = {"NVIDIA_API_KEY", "OPENROUTER_API_KEY", "TAVILY_API_KEY", "GOOGLE_API_KEY", "GCP_PROJECT_ID"}
    clean_environment = {name: value for name, value in os.environ.items() if name not in key_names}
    result = subprocess.run(["bash", str(script_copy)], capture_output=True, text=True, cwd=tmp_path, env=clean_environment)
    # It gets past the key checks and stops at the next gate (no built index in tmp_path).
    assert "chroma_index/ is missing" in result.stderr
    assert "NVIDIA_API_KEY is not set" in result.stderr
    assert "TAVILY_API_KEY is not set" in result.stderr


def prepare_deploy_sandbox(tmp_path, existing_service_label: str | None):
    """A deploy.sh copy with fake artifacts and a fake gcloud that logs every call it receives."""
    script_copy = tmp_path / "deploy.sh"
    script_copy.write_text((PROJECT_ROOT / "deploy.sh").read_text(encoding="utf-8"), encoding="utf-8")
    (tmp_path / ".env").write_text(
        "GCP_PROJECT_ID=proj-test\nGOOGLE_API_KEY=k1\nNVIDIA_API_KEY=k2\n"
        "OPENROUTER_API_KEY=k3\nTAVILY_API_KEY=k4\nCLOUD_RUN_SERVICE=csci599-a1\n",
        encoding="utf-8",
    )
    (tmp_path / "chroma_index").mkdir()
    (tmp_path / "chroma_index" / "chroma.sqlite3").write_bytes(b"x")
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "papers.db").write_bytes(b"x")
    (tmp_path / "static").mkdir()
    (tmp_path / "static" / "index.html").write_text("<html></html>", encoding="utf-8")

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    call_log = tmp_path / "gcloud_calls.log"
    # "run services describe" succeeds (service exists) and prints the label, if any.
    service_exists = "1" if existing_service_label is not None else "0"
    fake_gcloud = fake_bin / "gcloud"
    fake_gcloud.write_text(
        "#!/usr/bin/env bash\n"
        f'echo "$*" >> "{call_log}"\n'
        # Record the prompt setting each call ran with, so a test can prove none could prompt.
        f'echo "${{CLOUDSDK_CORE_DISABLE_PROMPTS:-unset}}" >> "{call_log}.prompts"\n'
        'if [[ "$1 $2 $3" == "run services describe" ]]; then\n'
        f'  [[ "{service_exists}" == "1" ]] || exit 1\n'
        f'  [[ "$*" == *metadata.labels* ]] && echo "{existing_service_label or ""}"\n'
        "  exit 0\n"
        "fi\n"
        "exit 0\n",
        encoding="utf-8",
    )
    fake_gcloud.chmod(0o755)
    environment = dict(os.environ, PATH=f"{fake_bin}:{os.environ['PATH']}", SKIP_IAM="1")
    return script_copy, call_log, environment


def test_deploy_refuses_to_overwrite_unlabelled_service(tmp_path):
    # Simulates pointing CLOUD_RUN_SERVICE at Assignment 1's service in the shared project.
    script_copy, call_log, environment = prepare_deploy_sandbox(tmp_path, existing_service_label="")
    result = subprocess.run(["bash", str(script_copy)], capture_output=True, text=True, cwd=tmp_path, env=environment)
    assert result.returncode != 0
    assert "Refusing to overwrite" in result.stderr
    gcloud_calls = call_log.read_text(encoding="utf-8")
    assert "run deploy" not in gcloud_calls
    assert "builds submit" not in gcloud_calls


def test_deploy_updates_its_own_labelled_service(tmp_path):
    script_copy, call_log, environment = prepare_deploy_sandbox(tmp_path, existing_service_label="csci599-a2")
    result = subprocess.run(["bash", str(script_copy)], capture_output=True, text=True, cwd=tmp_path, env=environment)
    assert result.returncode == 0, result.stderr
    gcloud_calls = call_log.read_text(encoding="utf-8")
    assert "run deploy csci599-a1" in gcloud_calls
    assert "--labels assignment=csci599-a2" in gcloud_calls


def test_deploy_enables_apis_before_the_label_guard(tmp_path):
    # On a fresh project the guard's `run services describe` hits a disabled Run API; with a
    # TTY gcloud would then prompt on the hidden stderr and hang. APIs must be enabled first.
    # A labelled service exists so the fake's final URL lookup succeeds and the run completes.
    script_copy, call_log, environment = prepare_deploy_sandbox(tmp_path, existing_service_label="csci599-a2")
    result = subprocess.run(["bash", str(script_copy)], capture_output=True, text=True, cwd=tmp_path, env=environment)
    assert result.returncode == 0, result.stderr
    call_lines = call_log.read_text(encoding="utf-8").splitlines()
    enable_indexes = [index for index, line in enumerate(call_lines) if line.startswith("services enable")]
    describe_indexes = [index for index, line in enumerate(call_lines) if line.startswith("run services describe")]
    assert enable_indexes, call_lines
    assert describe_indexes, call_lines
    assert enable_indexes[0] < describe_indexes[0], call_lines
    enable_call = call_lines[enable_indexes[0]]
    for api_name in ["run.googleapis.com", "cloudbuild.googleapis.com", "artifactregistry.googleapis.com", "compute.googleapis.com"]:
        assert api_name in enable_call, api_name


def test_deploy_disables_gcloud_prompts_for_every_call(tmp_path):
    script_copy, call_log, environment = prepare_deploy_sandbox(tmp_path, existing_service_label="csci599-a2")
    environment.pop("CLOUDSDK_CORE_DISABLE_PROMPTS", None)
    result = subprocess.run(["bash", str(script_copy)], capture_output=True, text=True, cwd=tmp_path, env=environment)
    assert result.returncode == 0, result.stderr
    prompt_settings = (tmp_path / "gcloud_calls.log.prompts").read_text(encoding="utf-8").splitlines()
    assert prompt_settings
    assert set(prompt_settings) == {"1"}, prompt_settings


def extract_dockerfile_index_check() -> str:
    dockerfile_text = (PROJECT_ROOT / "Dockerfile").read_text(encoding="utf-8")
    match = re.search(r'RUN python -c "(import chromadb, sqlite3;.*?)"\n', dockerfile_text, re.DOTALL)
    assert match, "Dockerfile lost its baked-index integrity check"
    # Undo the Dockerfile line continuations so the snippet runs as plain Python.
    return match.group(1).replace("\\\n", "")


@pytest.mark.skipif(not (PROJECT_ROOT / "chroma_index" / "chroma.sqlite3").exists(), reason="local index not built")
def test_dockerfile_index_check_passes_on_local_artifacts():
    check_code = extract_dockerfile_index_check()
    local_check = check_code.replace("/app/chroma_index", str(PROJECT_ROOT / "chroma_index"))
    local_check = local_check.replace("/app/data/papers.db", str(PROJECT_ROOT / "data" / "papers.db"))
    result = subprocess.run([sys.executable, "-c", local_check], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    assert "baked index ok" in result.stdout


def test_dockerfile_index_check_rejects_an_empty_db(tmp_path):
    import sqlite3

    empty_db = tmp_path / "papers.db"
    connection = sqlite3.connect(empty_db)
    connection.execute("CREATE TABLE papers (id TEXT)")
    connection.commit()
    connection.close()
    check_code = extract_dockerfile_index_check()
    # Keep only the papers.db half so this runs without an index.
    db_check = check_code[check_code.index("paper_count"):].replace("/app/data/papers.db", str(empty_db))
    result = subprocess.run([sys.executable, "-c", f"import sqlite3; {db_check}"], capture_output=True, text=True)
    assert result.returncode != 0
    assert "papers.db too small" in result.stderr


def test_validation_error_does_not_echo_the_query(client):
    oversized_query = "secret-looking-text " * 300
    response = client.post("/api/chat", json={"query": oversized_query, "session_id": "s"})
    assert response.status_code == 422
    assert "secret-looking-text" not in response.text
    assert response.json()["detail"][0]["field"] == "body.query"
