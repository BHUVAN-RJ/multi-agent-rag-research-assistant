"""Hybrid cost/capability LLM router over three free providers.

decide_route() is a pure, inspectable policy: (role, complexity, context size) -> one primary
(provider, model) plus a fallback chain that always crosses providers. complete() walks that
chain: providers without a key are skipped, a 429 moves to the next entry immediately, a
transient error (transport error, 5xx other than 503) gets one short retry, the whole walk
stops after TOTAL_CALL_BUDGET_SECONDS, and nothing is ever raised. Every call emits llm_route
first and llm_result last, all sharing one call_id so the UI can pair them.

Policy (see docs/routing_notes.md for the rationale):

| role       | condition                              | primary                 | dimension  |
|------------|----------------------------------------|-------------------------|------------|
| supervisor | any                                    | NVIDIA fast             | cost       |
| data       | simple                                 | NVIDIA fast             | cost       |
| data       | complex                                | NVIDIA strong           | capability |
| research   | simple and context <= 20k chars        | Gemini fast             | hybrid     |
| research   | complex or context > 20k chars         | Gemini strong           | capability |
| summary    | context <= 20k chars                   | OpenRouter free chain   | cost       |
| summary    | context > 20k chars                    | Gemini fast             | capability |
| judge      | any                                    | Gemini strong           | capability |
"""

import asyncio
import logging
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Literal

from src.llm import providers
from src.trace_events import TraceEvent, emit_token, emit_trace

Role = Literal["supervisor", "research", "summary", "data", "judge"]
Provider = Literal["nvidia", "gemini", "openrouter"]
Dimension = Literal["cost", "capability", "hybrid"]

logger = logging.getLogger(__name__)

# Above this many prompt characters (~5k tokens) we route for long-context capability.
LONG_CONTEXT_CHARS = 20_000

# A (provider, model) that returned 429 is tried last for this long, instead of first.
RATE_LIMIT_COOLDOWN_SECONDS = 30.0
MAX_COOLDOWN_SECONDS = 120.0
TRANSIENT_RETRY_DELAY_SECONDS = 0.75

# Wall-clock budget for one complete() call across the whole fallback walk. Cloud Run kills a
# request at 300 s and one turn makes several LLM calls, so one call must never walk the full
# chain of 60 s timeouts. Each attempt is also capped at whatever budget is left.
TOTAL_CALL_BUDGET_SECONDS = 90.0

# Generous because several free models spend part of the output budget on hidden reasoning.
DEFAULT_MAX_TOKENS: dict[str, int] = {
    "supervisor": 1024,
    "data": 1024,
    "research": 4096,
    "summary": 2048,
    "judge": 2048,
}

# Test/eval hook: pin the primary to one provider to compare providers on the same workload.
FORCE_PROVIDER_ENV = "ROUTER_FORCE_PROVIDER"
VALID_PROVIDERS: tuple[str, ...] = ("nvidia", "gemini", "openrouter")

# (provider, model) -> monotonic time until which it is considered rate limited.
rate_limited_until: dict[tuple[str, str], float] = {}


@dataclass
class TaskProfile:
    complexity: Literal["simple", "complex"]  # classification/short lookup vs multi-step synthesis
    context_chars: int = 0  # size of prompt context, drives capability routing


@dataclass
class RouteDecision:
    role: Role
    provider: Provider
    model: str
    dimension: Dimension
    reason: str  # human-readable, shown in UI
    fallback_chain: list[tuple[str, str]] = field(default_factory=list)  # tried in order after primary


@dataclass
class LLMResult:
    text: str
    provider: str
    model: str
    decision: RouteDecision
    input_tokens: int | None
    output_tokens: int | None
    latency_ms: float
    ttft_ms: float | None
    attempts: list[dict]  # one per try: provider, model, ok, error
    error: str | None = None  # set only if every provider failed


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


def nvidia_fast() -> tuple[str, str]:
    return ("nvidia", providers.resolve_model("nvidia", "fast"))


def nvidia_strong() -> tuple[str, str]:
    return ("nvidia", providers.resolve_model("nvidia", "strong"))


def gemini_fast() -> tuple[str, str]:
    return ("gemini", providers.resolve_model("gemini", "fast"))


def gemini_strong() -> tuple[str, str]:
    return ("gemini", providers.resolve_model("gemini", "strong"))


def openrouter_chain() -> list[tuple[str, str]]:
    free_models = providers.openrouter_free_chain()
    return [("openrouter", model_id) for model_id in free_models]


def is_long_context(profile: TaskProfile) -> bool:
    return profile.context_chars > LONG_CONTEXT_CHARS


def context_label(profile: TaskProfile) -> str:
    kilo_chars = profile.context_chars / 1000
    return f"context {kilo_chars:.1f}k chars"


def select_policy(role: str, profile: TaskProfile) -> tuple[list[tuple[str, str]], str, str]:
    """Return (ordered candidates, dimension, reason). The first candidate is the primary."""
    long_context = is_long_context(profile)
    context_text = context_label(profile)
    threshold_text = f"{LONG_CONTEXT_CHARS // 1000}k threshold"

    if role == "supervisor":
        candidates = [nvidia_fast(), gemini_fast(), *openrouter_chain(), nvidia_strong()]
        reason = "cost: supervisor routing is short classification -> cheapest low-latency NVIDIA fast model"
        return candidates, "cost", reason

    if role == "data":
        if profile.complexity == "complex":
            candidates = [nvidia_strong(), gemini_strong(), nvidia_fast(), *openrouter_chain()]
            reason = "capability: large SQL result (>20 rows) to explain -> stronger NVIDIA Nemotron model"
            return candidates, "capability", reason
        candidates = [nvidia_fast(), gemini_fast(), *openrouter_chain(), nvidia_strong()]
        reason = "cost: simple structured SQL generation -> cheapest NVIDIA fast model"
        return candidates, "cost", reason

    if role == "research":
        if long_context:
            candidates = [gemini_strong(), gemini_fast(), nvidia_strong(), *openrouter_chain()]
            reason = f"capability: {context_text} > {threshold_text} -> long-context Gemini strong"
            return candidates, "capability", reason
        if profile.complexity == "complex":
            candidates = [gemini_strong(), nvidia_strong(), gemini_fast(), *openrouter_chain()]
            reason = "capability: complex multi-source synthesis -> strongest generator, Gemini strong"
            return candidates, "capability", reason
        candidates = [gemini_fast(), nvidia_fast(), *openrouter_chain(), gemini_strong()]
        reason = f"hybrid: simple lookup, {context_text} <= {threshold_text} -> fast, cheap Gemini Flash-Lite"
        return candidates, "hybrid", reason

    if role == "summary":
        if long_context:
            candidates = [gemini_fast(), gemini_strong(), nvidia_strong(), *openrouter_chain()]
            reason = f"capability: {context_text} > {threshold_text} -> long-context Gemini (1M-token window)"
            return candidates, "capability", reason
        candidates = [*openrouter_chain(), gemini_fast(), nvidia_fast()]
        reason = f"cost: condensation of {context_text} <= {threshold_text} -> zero-cost OpenRouter free chain (openrouter/free router first)"
        return candidates, "cost", reason

    if role == "judge":
        candidates = [gemini_strong(), nvidia_strong(), gemini_fast(), *openrouter_chain()]
        reason = "capability: RAGAS judging needs the strongest instruction-follower -> Gemini strong"
        return candidates, "capability", reason

    # Unknown role: behave like a cheap generic call rather than failing.
    candidates = [nvidia_fast(), gemini_fast(), *openrouter_chain()]
    reason = f"cost: unknown role '{role}' -> default cheapest NVIDIA fast model"
    return candidates, "cost", reason


def dedupe_candidates(candidates: list[tuple[str, str]]) -> list[tuple[str, str]]:
    seen: set[tuple[str, str]] = set()
    unique: list[tuple[str, str]] = []
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        unique.append(candidate)
    return unique


def forced_provider() -> str | None:
    raw_value = os.environ.get(FORCE_PROVIDER_ENV, "")
    normalized = raw_value.strip().lower()
    if normalized in VALID_PROVIDERS:
        return normalized
    return None


def apply_force_override(candidates: list[tuple[str, str]], forced: str) -> list[tuple[str, str]]:
    """Move the forced provider's candidates to the front, keeping everything else as fallback."""
    forced_entries = [candidate for candidate in candidates if candidate[0] == forced]
    if not forced_entries:
        if forced == "openrouter":
            forced_entries = openrouter_chain()
        else:
            forced_entries = [(forced, providers.resolve_model(forced, "fast"))]
    other_entries = [candidate for candidate in candidates if candidate[0] != forced]
    return [*forced_entries, *other_entries]


def decide_route(role: Role, profile: TaskProfile) -> RouteDecision:
    candidates, dimension, reason = select_policy(role, profile)
    forced = forced_provider()
    if forced is not None:
        candidates = apply_force_override(candidates, forced)
        reason = f"override: {FORCE_PROVIDER_ENV}={forced} (policy would say: {reason})"

    ordered = dedupe_candidates(candidates)
    primary_provider, primary_model = ordered[0]
    return RouteDecision(
        role=role,
        provider=primary_provider,
        model=primary_model,
        dimension=dimension,
        reason=reason,
        fallback_chain=ordered[1:],
    )


def routing_table() -> list[RouteDecision]:
    """One decision per (role, sample profile), for the UI / architecture doc. Honors the force override."""
    sample_profiles = [
        TaskProfile("simple", 2_000),
        TaskProfile("complex", 8_000),
        TaskProfile("complex", LONG_CONTEXT_CHARS * 2),
    ]
    roles: list[Role] = ["supervisor", "data", "research", "summary", "judge"]
    table: list[RouteDecision] = []
    for role in roles:
        for profile in sample_profiles:
            table.append(decide_route(role, profile))
    return table


# ---------------------------------------------------------------------------
# Execution with fallback
# ---------------------------------------------------------------------------


def display_model(requested_model: str, served_model: str | None) -> str:
    """Show which model really answered when a router id (openrouter/free) picked one."""
    if requested_model == providers.OPENROUTER_FREE_ROUTER_ID and served_model and served_model != requested_model:
        return f"{requested_model} -> {served_model}"
    return requested_model


def budget_clock() -> float:
    """Clock for the per-call time budget (a function so tests can fake elapsed time)."""
    return time.monotonic()


def new_call_id() -> str:
    return uuid.uuid4().hex[:12]


def is_cooling_down(candidate: tuple[str, str]) -> bool:
    until = rate_limited_until.get(candidate, 0.0)
    return time.monotonic() < until


def cooldown_remaining_seconds(candidate: tuple[str, str]) -> float:
    until = rate_limited_until.get(candidate, 0.0)
    return max(0.0, until - time.monotonic())


def mark_rate_limited(candidate: tuple[str, str], retry_after_seconds: float | None) -> None:
    cooldown = retry_after_seconds if retry_after_seconds else RATE_LIMIT_COOLDOWN_SECONDS
    bounded_cooldown = min(cooldown, MAX_COOLDOWN_SECONDS)
    rate_limited_until[candidate] = time.monotonic() + bounded_cooldown


def order_for_attempts(decision: RouteDecision) -> list[tuple[str, str]]:
    """Primary then fallbacks, but anything that recently hit 429 goes to the back, not dropped."""
    full_chain = [(decision.provider, decision.model), *decision.fallback_chain]
    ready = [candidate for candidate in full_chain if not is_cooling_down(candidate)]
    cooling = [candidate for candidate in full_chain if is_cooling_down(candidate)]
    return [*ready, *cooling]


def deferred_for_cooldown(decision: RouteDecision) -> list[tuple[tuple[str, str], tuple[str, str]]]:
    """(cooling candidate, ready candidate tried in its place) for each entry moved to the back.

    A cooling entry counts as moved only if some ready entry came after it in the policy order;
    one that was already behind every ready entry keeps its place and needs no trace.
    """
    full_chain = [(decision.provider, decision.model), *decision.fallback_chain]
    moved: list[tuple[tuple[str, str], tuple[str, str]]] = []
    for index, candidate in enumerate(full_chain):
        if not is_cooling_down(candidate):
            continue
        later_ready = [later for later in full_chain[index + 1:] if not is_cooling_down(later)]
        if later_ready:
            moved.append((candidate, later_ready[0]))
    return moved


def emit_route_trace(decision: RouteDecision, node: str, call_id: str) -> None:
    primary_label = providers.candidate_label(decision.provider, decision.model)
    summary = f"{decision.role} -> {primary_label} [{decision.dimension}]"
    detail = {**asdict(decision), "call_id": call_id}
    emit_trace(TraceEvent("llm_route", node, summary, detail))


def emit_fallback_trace(node: str, attempt: dict, next_candidate: tuple[str, str] | None, extra: dict | None = None) -> None:
    if next_candidate is None:
        next_text = "no providers left"
    else:
        next_text = f"next: {providers.candidate_label(next_candidate[0], next_candidate[1])}"
    detail = {**attempt, "next": list(next_candidate) if next_candidate else None}
    if extra:
        detail.update(extra)
    verb = "skipped" if detail.get("skipped") else "failed"
    attempt_label = providers.candidate_label(attempt["provider"], attempt["model"])
    summary = f"{attempt_label} {verb} ({attempt['error']}); {next_text}"
    emit_trace(TraceEvent("llm_fallback", node, summary, detail))


def emit_cooldown_skip_traces(decision: RouteDecision, node: str, call_id: str) -> None:
    """Say why a cooling-down candidate (often the announced primary) is not tried first."""
    for candidate, replacement in deferred_for_cooldown(decision):
        provider, model = candidate
        seconds_left = round(cooldown_remaining_seconds(candidate), 1)
        skip_note = {
            "provider": provider,
            "model": model,
            "ok": False,
            "error": f"skipped: cooling down ({seconds_left:.0f}s left), tried last",
        }
        extra = {
            "skipped": True,
            "reason": "cooldown",
            "cooldown_remaining_s": seconds_left,
            "call_id": call_id,
            "stream_restarted": False,
        }
        emit_fallback_trace(node, skip_note, replacement, extra)


def log_reason(provider_error: providers.ProviderError) -> str:
    """Short failure reason for logs: status code or error class, never the response body."""
    if provider_error.status_code is not None:
        return f"HTTP {provider_error.status_code}"
    return provider_error.message[:80]


def emit_result_trace(
    node: str,
    call_id: str,
    role: str,
    result_fields: dict,
) -> None:
    """One llm_result per complete() call: which candidate really answered, or that none did."""
    detail = {"call_id": call_id, "role": role, **result_fields}
    if detail["ok"]:
        answered_label = providers.candidate_label(detail["provider"], detail["model"])
        summary = f"{role} answered by {answered_label} ({detail['latency_ms']:.0f} ms, {detail['attempts']} attempt(s))"
    else:
        summary = f"{role}: no provider answered ({detail['attempts']} attempt(s))"
    emit_trace(TraceEvent("llm_result", node, summary, detail))


async def call_with_one_retry(
    provider: str,
    model: str,
    messages: list[dict],
    stream_tokens: bool,
    node: str,
    max_tokens: int,
    temperature: float,
    token_counter: list[int],
) -> providers.ProviderResponse:
    def forward_token(token_text: str) -> None:
        token_counter[0] += 1
        emit_token(token_text, node)

    on_token = forward_token if stream_tokens else None
    try:
        return await providers.call_provider(
            provider, model, messages, stream=stream_tokens, on_token=on_token,
            max_tokens=max_tokens, temperature=temperature,
        )
    except providers.ProviderError as error:
        # Retry once only for transient errors that happened before any token reached the UI.
        # 429s are never retried here: on free tiers the next provider is faster than waiting.
        should_retry = error.retryable and not error.rate_limited and token_counter[0] == 0
        if not should_retry:
            raise
    await asyncio.sleep(TRANSIENT_RETRY_DELAY_SECONDS)
    return await providers.call_provider(
        provider, model, messages, stream=stream_tokens, on_token=on_token,
        max_tokens=max_tokens, temperature=temperature,
    )


async def complete(
    role: Role,
    messages: list[dict],
    profile: TaskProfile,
    stream_tokens: bool = False,
    node: str = "",
    *,
    max_tokens: int | None = None,
    temperature: float = 0.2,
) -> LLMResult:
    """Route, then walk the fallback chain. Never raises; total failure sets LLMResult.error."""
    start_time = time.perf_counter()
    budget_start = budget_clock()
    trace_node = node or role
    call_id = new_call_id()
    attempts: list[dict] = []
    untried_count = 0
    token_budget = max_tokens if max_tokens is not None else DEFAULT_MAX_TOKENS.get(role, 1024)

    try:
        decision = decide_route(role, profile)
    except Exception as error:
        # Policy bugs must not take down a graph run; fall back to a minimal decision.
        fallback_provider, fallback_model = nvidia_fast()
        decision = RouteDecision(role, fallback_provider, fallback_model, "cost", f"policy error: {error}", [])

    emit_route_trace(decision, trace_node, call_id)
    emit_cooldown_skip_traces(decision, trace_node, call_id)
    ordered_candidates = order_for_attempts(decision)

    for index, candidate in enumerate(ordered_candidates):
        provider, model = candidate
        remaining = ordered_candidates[index + 1:]
        next_candidate = remaining[0] if remaining else None

        if not providers.has_api_key(provider):
            attempt = {"provider": provider, "model": model, "ok": False, "error": "skipped: no API key"}
            attempts.append(attempt)
            logger.info("LLM skip: %s has no API key", provider)
            emit_fallback_trace(trace_node, attempt, next_candidate, {"skipped": True, "call_id": call_id})
            continue

        # Stop walking the chain once the call's time budget is spent; the rest stay untried.
        budget_left = TOTAL_CALL_BUDGET_SECONDS - (budget_clock() - budget_start)
        if budget_left <= 0:
            untried_count = len(ordered_candidates) - index
            logger.warning(
                "LLM time budget of %.0fs spent for %s; %d candidate(s) not tried",
                TOTAL_CALL_BUDGET_SECONDS, role, untried_count,
            )
            break

        token_counter = [0]
        attempt_start = time.perf_counter()
        try:
            pending_call = call_with_one_retry(
                provider, model, messages, stream_tokens, trace_node, token_budget, temperature, token_counter
            )
            # The retry and the per-call 60 s timeout both live inside this cap.
            response = await asyncio.wait_for(pending_call, timeout=budget_left)
        except Exception as error:
            provider_error = providers.error_from_exception(provider, model, error)
            if provider_error.rate_limited or provider_error.overloaded:
                mark_rate_limited(candidate, provider_error.retry_after_seconds)
            attempt = {
                "provider": provider,
                "model": model,
                "ok": False,
                "error": str(provider_error),
                "status_code": provider_error.status_code,
                "rate_limited": provider_error.rate_limited,
                "latency_ms": round((time.perf_counter() - attempt_start) * 1000.0, 1),
            }
            attempts.append(attempt)
            next_text = providers.candidate_label(*next_candidate) if next_candidate else "nothing (chain exhausted)"
            logger.warning(
                "LLM fallback: %s/%s failed (%s), trying %s",
                provider, model, log_reason(provider_error), next_text,
            )
            # If tokens were already streamed, the UI must discard them before the next attempt.
            stream_note = {"stream_restarted": token_counter[0] > 0, "call_id": call_id}
            emit_fallback_trace(trace_node, attempt, next_candidate, stream_note)
            continue

        answered_model = display_model(model, response.served_model)
        attempts.append({
            "provider": provider,
            "model": model,
            "served_model": response.served_model,
            "ok": True,
            "error": None,
            "latency_ms": round(response.latency_ms, 1),
        })
        total_latency = (time.perf_counter() - start_time) * 1000.0
        emit_result_trace(trace_node, call_id, role, {
            "ok": True,
            "provider": provider,
            "model": answered_model,
            "requested_model": model,
            "served_model": response.served_model,
            "latency_ms": round(total_latency, 1),
            "ttft_ms": round(response.ttft_ms, 1) if response.ttft_ms is not None else None,
            "attempts": len(attempts),
            "fell_back": len(attempts) > 1,
            "budget_exhausted": False,
            "error": None,
        })
        return LLMResult(
            text=response.text,
            provider=provider,
            model=answered_model,
            decision=decision,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            latency_ms=total_latency,
            ttft_ms=response.ttft_ms,
            attempts=attempts,
        )

    total_latency = (time.perf_counter() - start_time) * 1000.0
    failure_summary = "; ".join(str(attempt["error"]) for attempt in attempts)
    error_text = f"all {len(attempts)} provider attempts failed: {failure_summary}"
    budget_exhausted = untried_count > 0
    if budget_exhausted:
        error_text += f"; stopped after the {TOTAL_CALL_BUDGET_SECONDS:.0f}s time budget with {untried_count} candidate(s) untried"
    logger.warning("LLM call failed: role=%s, %d attempt(s), budget_exhausted=%s", role, len(attempts), budget_exhausted)
    error_detail = {"attempts": attempts, "call_id": call_id, "budget_exhausted": budget_exhausted}
    emit_trace(TraceEvent("error", trace_node, "LLM call failed on every provider", error_detail))
    emit_result_trace(trace_node, call_id, role, {
        "ok": False,
        "provider": "",
        "model": "",
        "requested_model": "",
        "served_model": None,
        "latency_ms": round(total_latency, 1),
        "ttft_ms": None,
        "attempts": len(attempts),
        "fell_back": len(attempts) > 1,
        "budget_exhausted": budget_exhausted,
        "error": error_text,
    })
    return LLMResult(
        text="",
        provider="",
        model="",
        decision=decision,
        input_tokens=None,
        output_tokens=None,
        latency_ms=total_latency,
        ttft_ms=None,
        attempts=attempts,
        error=error_text,
    )
