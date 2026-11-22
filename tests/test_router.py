"""Router policy, fallback and provider-client tests. Offline by default; live pings are opt-in."""

import asyncio
import json
from dataclasses import replace
from pathlib import Path

import httpx
import pytest

from src.llm import providers, router
from src.llm.providers import ProviderError, ProviderResponse
from src.llm.router import LONG_CONTEXT_CHARS, TaskProfile, complete, decide_route
from src.trace_events import fallback_sink

ALL_ROLES = ["supervisor", "data", "research", "summary", "judge"]
MESSAGES = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Say ok."}]


@pytest.fixture(autouse=True)
def clean_router_state(monkeypatch):
    monkeypatch.delenv(router.FORCE_PROVIDER_ENV, raising=False)
    router.rate_limited_until.clear()
    monkeypatch.setattr(router, "TRANSIENT_RETRY_DELAY_SECONDS", 0.0)
    yield
    router.rate_limited_until.clear()


@pytest.fixture
def captured_events():
    events: list[dict] = []
    token = fallback_sink.set(events.append)
    yield events
    fallback_sink.reset(token)


@pytest.fixture
def all_keys_present(monkeypatch):
    def pretend_key_present(provider: str) -> bool:
        return True

    monkeypatch.setattr(providers, "has_api_key", pretend_key_present)


def traces_of_kind(events: list[dict], kind: str) -> list[dict]:
    trace_payloads = [event["data"] for event in events if event["type"] == "trace"]
    return [payload for payload in trace_payloads if payload["kind"] == kind]


def make_scripted_provider(script: dict[str, object], calls: list[tuple[str, str]]):
    """Fake call_provider: script maps provider name -> Exception to raise, or text to return."""

    async def scripted_call_provider(provider, model, messages, stream=False, on_token=None, max_tokens=1024, temperature=0.2, transport=None):
        calls.append((provider, model))
        outcome = script.get(provider, "default ok")
        if isinstance(outcome, Exception):
            raise outcome
        if stream and on_token is not None:
            for word in str(outcome).split(" "):
                on_token(word)
        return ProviderResponse(str(outcome), 11, 7, 5.0, 1.0 if stream else None)

    return scripted_call_provider


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


def test_different_profiles_yield_different_providers():
    supervisor_route = decide_route("supervisor", TaskProfile("simple"))
    research_route = decide_route("research", TaskProfile("complex", 5_000))
    summary_route = decide_route("summary", TaskProfile("simple", 3_000))

    assert supervisor_route.provider == "nvidia"
    assert research_route.provider == "gemini"
    assert summary_route.provider == "openrouter"


def test_at_least_three_providers_across_roles():
    providers_seen: set[str] = set()
    models_seen: set[str] = set()
    for role in ALL_ROLES:
        for complexity in ("simple", "complex"):
            for context_chars in (1_000, LONG_CONTEXT_CHARS * 3):
                decision = decide_route(role, TaskProfile(complexity, context_chars))
                providers_seen.add(decision.provider)
                models_seen.add(decision.model)
    assert providers_seen == {"nvidia", "gemini", "openrouter"}
    # Not just three providers: the fast/strong tiers are both used too.
    assert len(models_seen) >= 5


def test_complexity_changes_model_within_role():
    simple_data = decide_route("data", TaskProfile("simple"))
    complex_data = decide_route("data", TaskProfile("complex"))
    simple_research = decide_route("research", TaskProfile("simple", 1_000))
    complex_research = decide_route("research", TaskProfile("complex", 1_000))

    assert simple_data.model != complex_data.model
    assert simple_research.model != complex_research.model


def test_long_context_moves_summary_off_openrouter():
    short_summary = decide_route("summary", TaskProfile("simple", LONG_CONTEXT_CHARS - 1))
    long_summary = decide_route("summary", TaskProfile("simple", LONG_CONTEXT_CHARS + 1))

    assert short_summary.provider == "openrouter"
    assert short_summary.dimension == "cost"
    assert long_summary.provider == "gemini"
    assert long_summary.dimension == "capability"
    assert "threshold" in long_summary.reason


def test_reasons_name_the_dimension():
    for role in ALL_ROLES:
        for complexity in ("simple", "complex"):
            for context_chars in (500, LONG_CONTEXT_CHARS * 2):
                decision = decide_route(role, TaskProfile(complexity, context_chars))
                assert decision.dimension in ("cost", "capability", "hybrid")
                assert decision.reason.strip()
                assert decision.reason.startswith(f"{decision.dimension}:"), decision.reason


def test_fallback_chain_crosses_providers_and_excludes_primary():
    for role in ALL_ROLES:
        for complexity in ("simple", "complex"):
            decision = decide_route(role, TaskProfile(complexity, 1_000))
            chain_providers = {provider for provider, _model in decision.fallback_chain}
            other_providers = chain_providers - {decision.provider}
            assert other_providers, f"{role}/{complexity} chain never leaves {decision.provider}"
            assert (decision.provider, decision.model) not in decision.fallback_chain


def is_explicitly_free(model_id: str) -> bool:
    # Spelled out here rather than reusing providers.is_free_openrouter_model, so a loosened guard fails.
    return model_id.endswith(":free") or model_id == "openrouter/free"


def test_default_openrouter_chain_is_only_the_free_router():
    assert providers.DEFAULT_OPENROUTER_FREE_MODELS == ["openrouter/free"]


def test_env_example_openrouter_chain_is_only_the_free_router():
    env_example = (Path(__file__).parent.parent / ".env.example").read_text()
    assert "OPENROUTER_FREE_MODELS=openrouter/free\n" in env_example


@pytest.mark.parametrize("paid_router_id", ["openrouter/auto", "openrouter/auto-beta", "openrouter/fusion"])
async def test_paid_openrouter_routers_are_refused_without_network(paid_router_id):
    # openrouter/auto picks paid models; only openrouter/free may pass the guard.
    with pytest.raises(ProviderError, match="not a ':free' model"):
        await providers.call_provider("openrouter", paid_router_id, MESSAGES)


async def test_free_router_reports_the_model_that_answered(monkeypatch, all_keys_present):
    def handle_request(request: httpx.Request) -> httpx.Response:
        request_body = json.loads(request.content)
        assert request_body["model"] == "openrouter/free"
        return httpx.Response(200, json={
            "model": "google/gemma-4-31b-it:free",
            "choices": [{"message": {"content": "routed answer"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2},
        })

    response = await providers.call_provider(
        "openrouter", "openrouter/free", MESSAGES, transport=httpx.MockTransport(handle_request)
    )
    assert response.served_model == "google/gemma-4-31b-it:free"
    assert router.display_model("openrouter/free", response.served_model) == "openrouter/free -> google/gemma-4-31b-it:free"
    # Pinned ids are shown as-is even if the provider echoes a different alias.
    assert router.display_model("google/gemma-4-31b-it:free", "google/gemma-4-31b-it") == "google/gemma-4-31b-it:free"


def test_openrouter_chain_only_free_models():
    default_chain = providers.openrouter_free_chain()
    assert default_chain == ["openrouter/free"]
    assert all(is_explicitly_free(model_id) for model_id in default_chain)

    for role in ALL_ROLES:
        for complexity in ("simple", "complex"):
            decision = decide_route(role, TaskProfile(complexity, 1_000))
            full_chain = [(decision.provider, decision.model), *decision.fallback_chain]
            openrouter_models = [model for provider, model in full_chain if provider == "openrouter"]
            assert all(is_explicitly_free(model) for model in openrouter_models)


def test_paid_openrouter_ids_in_config_are_dropped(monkeypatch):
    mixed_settings = replace(
        providers.settings,
        openrouter_free_models=["openai/gpt-paid-model", "google/gemma-4-31b-it:free"],
    )
    monkeypatch.setattr(providers, "settings", mixed_settings)
    assert providers.openrouter_free_chain() == ["google/gemma-4-31b-it:free"]


async def test_paid_openrouter_model_is_refused_without_network():
    with pytest.raises(ProviderError, match="not a ':free' model"):
        await providers.call_provider("openrouter", "openai/gpt-paid-model", MESSAGES)


def test_placeholder_model_ids_fall_back_to_defaults(monkeypatch):
    placeholder_settings = replace(providers.settings, nvidia_model_fast="<verified-nim-model-id>", gemini_model_strong="")
    monkeypatch.setattr(providers, "settings", placeholder_settings)
    assert providers.resolve_model("nvidia", "fast") == providers.DEFAULT_MODELS["nvidia"]["fast"]
    assert providers.resolve_model("gemini", "strong") == providers.DEFAULT_MODELS["gemini"]["strong"]


def test_force_provider_override(monkeypatch):
    monkeypatch.setenv(router.FORCE_PROVIDER_ENV, "openrouter")
    decision = decide_route("judge", TaskProfile("complex"))
    assert decision.provider == "openrouter"
    assert decision.reason.startswith("override:")
    # The rest of the policy chain is kept as fallback.
    assert any(provider == "gemini" for provider, _model in decision.fallback_chain)


# ---------------------------------------------------------------------------
# Execution and fallback
# ---------------------------------------------------------------------------


async def test_rate_limit_falls_back_to_next_provider(monkeypatch, captured_events, all_keys_present):
    calls: list[tuple[str, str]] = []
    rate_limit_error = ProviderError("gemini", "m", "quota exceeded", status_code=429)
    scripted = make_scripted_provider({"gemini": rate_limit_error, "nvidia": "fallback answer"}, calls)
    monkeypatch.setattr(providers, "call_provider", scripted)

    result = await complete("research", MESSAGES, TaskProfile("complex", 2_000), node="research")

    assert result.error is None
    assert result.text == "fallback answer"
    assert result.provider == "nvidia"
    assert result.decision.provider == "gemini"
    # 429 means move on immediately: the rate-limited model was called exactly once.
    gemini_calls = [call for call in calls if call[0] == "gemini"]
    assert len(gemini_calls) == 1
    assert [attempt["ok"] for attempt in result.attempts] == [False, True]
    assert result.attempts[0]["rate_limited"] is True

    route_traces = traces_of_kind(captured_events, "llm_route")
    fallback_traces = traces_of_kind(captured_events, "llm_fallback")
    assert len(route_traces) == 1
    assert route_traces[0]["detail"]["dimension"] == "capability"
    assert len(fallback_traces) == 1
    assert fallback_traces[0]["detail"]["provider"] == "gemini"
    assert fallback_traces[0]["detail"]["next"][0] == "nvidia"


async def test_rate_limited_model_is_tried_last_next_time(monkeypatch, all_keys_present):
    calls: list[tuple[str, str]] = []
    rate_limit_error = ProviderError("gemini", "m", "quota exceeded", status_code=429)
    scripted = make_scripted_provider({"gemini": rate_limit_error, "nvidia": "ok"}, calls)
    monkeypatch.setattr(providers, "call_provider", scripted)

    await complete("research", MESSAGES, TaskProfile("complex", 2_000))
    calls.clear()
    second_result = await complete("research", MESSAGES, TaskProfile("complex", 2_000))

    assert calls[0][0] == "nvidia"
    assert second_result.provider == "nvidia"


async def test_transient_error_gets_exactly_one_retry(monkeypatch, all_keys_present):
    calls: list[tuple[str, str]] = []
    server_error = ProviderError("nvidia", "m", "bad gateway", status_code=502, retryable=True)
    scripted = make_scripted_provider({"nvidia": server_error, "gemini": "from gemini"}, calls)
    monkeypatch.setattr(providers, "call_provider", scripted)

    result = await complete("supervisor", MESSAGES, TaskProfile("simple"))

    nvidia_calls = [call for call in calls if call[0] == "nvidia"]
    assert len(nvidia_calls) == 2
    assert result.provider == "gemini"


async def test_missing_key_is_skipped(monkeypatch, captured_events):
    calls: list[tuple[str, str]] = []

    def only_gemini_has_key(provider: str) -> bool:
        return provider == "gemini"

    monkeypatch.setattr(providers, "has_api_key", only_gemini_has_key)
    monkeypatch.setattr(providers, "call_provider", make_scripted_provider({"gemini": "gemini text"}, calls))

    result = await complete("supervisor", MESSAGES, TaskProfile("simple"))

    assert result.provider == "gemini"
    assert all(provider == "gemini" for provider, _model in calls)
    assert result.attempts[0]["provider"] == "nvidia"
    assert result.attempts[0]["error"] == "skipped: no API key"
    skip_traces = [trace for trace in traces_of_kind(captured_events, "llm_fallback") if trace["detail"].get("skipped")]
    assert skip_traces
    assert "no API key" in skip_traces[0]["summary"]


async def test_total_failure_returns_error_without_raising(monkeypatch, captured_events, all_keys_present):
    calls: list[tuple[str, str]] = []
    script = {
        "nvidia": ProviderError("nvidia", "m", "rate", status_code=429),
        "gemini": ProviderError("gemini", "m", "server", status_code=500, retryable=True),
        "openrouter": RuntimeError("unexpected crash"),
    }
    monkeypatch.setattr(providers, "call_provider", make_scripted_provider(script, calls))

    result = await complete("summary", MESSAGES, TaskProfile("simple", 1_000))

    assert result.error is not None
    assert result.error.startswith("all ")
    assert result.text == ""
    decision_chain_length = 1 + len(result.decision.fallback_chain)
    assert len(result.attempts) == decision_chain_length
    assert not any(attempt["ok"] for attempt in result.attempts)
    assert traces_of_kind(captured_events, "error")


async def test_no_keys_at_all_returns_error(monkeypatch):
    def no_key(provider: str) -> bool:
        return False

    monkeypatch.setattr(providers, "has_api_key", no_key)
    result = await complete("judge", MESSAGES, TaskProfile("complex"))
    assert result.error is not None
    assert all(attempt["error"] == "skipped: no API key" for attempt in result.attempts)


async def test_streaming_emits_tokens_in_order(monkeypatch, captured_events, all_keys_present):
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(providers, "call_provider", make_scripted_provider({"nvidia": "alpha beta gamma delta"}, calls))

    result = await complete("supervisor", MESSAGES, TaskProfile("simple"), stream_tokens=True, node="supervisor")

    token_payloads = [event["data"] for event in captured_events if event["type"] == "token"]
    token_texts = [payload["text"] for payload in token_payloads]
    assert token_texts == ["alpha", "beta", "gamma", "delta"]
    assert all(payload["node"] == "supervisor" for payload in token_payloads)
    assert result.ttft_ms is not None


async def test_no_tokens_emitted_when_not_streaming(monkeypatch, captured_events, all_keys_present):
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(providers, "call_provider", make_scripted_provider({"nvidia": "alpha beta"}, calls))
    await complete("supervisor", MESSAGES, TaskProfile("simple"), stream_tokens=False)
    assert not [event for event in captured_events if event["type"] == "token"]


# ---------------------------------------------------------------------------
# Provider clients against a mocked HTTP transport
# ---------------------------------------------------------------------------


def sse_body(payloads: list[str]) -> bytes:
    lines = [f"data: {payload}\n\n" for payload in payloads]
    return "".join(lines).encode()


async def test_openai_compatible_stream_parses_tokens_and_usage(monkeypatch):
    monkeypatch.setattr(providers, "settings", replace(providers.settings, nvidia_api_key="nvapi-test"))
    chunks = [
        json.dumps({"choices": [{"delta": {"content": "Hel"}}]}),
        json.dumps({"choices": [{"delta": {"content": "lo"}}]}),
        json.dumps({"choices": [], "usage": {"prompt_tokens": 9, "completion_tokens": 2}}),
        "[DONE]",
    ]
    seen_requests: list[dict] = []

    def handle_request(request: httpx.Request) -> httpx.Response:
        seen_requests.append(json.loads(request.content))
        return httpx.Response(200, content=sse_body(chunks), headers={"content-type": "text/event-stream"})

    streamed_tokens: list[str] = []
    response = await providers.call_provider(
        "nvidia", "nvidia/nemotron-3.5-lightning-30b-a3b", MESSAGES, stream=True,
        on_token=streamed_tokens.append, transport=httpx.MockTransport(handle_request),
    )

    assert response.text == "Hello"
    assert streamed_tokens == ["Hel", "lo"]
    assert (response.input_tokens, response.output_tokens) == (9, 2)
    assert response.ttft_ms is not None
    assert seen_requests[0]["stream_options"] == {"include_usage": True}
    assert seen_requests[0]["chat_template_kwargs"] == {"enable_thinking": False}


async def test_http_429_maps_to_rate_limited_error():
    def handle_request(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": "slow down"}, headers={"retry-after": "7"})

    with pytest.raises(ProviderError) as error_info:
        await providers.call_provider(
            "openrouter", "google/gemma-4-31b-it:free", MESSAGES, transport=httpx.MockTransport(handle_request)
        )
    assert error_info.value.rate_limited
    assert error_info.value.retry_after_seconds == 7.0


async def test_gemini_non_streaming_parses_text_and_usage(monkeypatch):
    monkeypatch.setattr(providers, "settings", replace(providers.settings, google_api_key="test-key"))
    seen_requests: list[httpx.Request] = []
    payload = {
        "candidates": [{"content": {"parts": [{"text": "thinking...", "thought": True}, {"text": "answer"}]}}],
        "usageMetadata": {"promptTokenCount": 12, "candidatesTokenCount": 3},
    }

    def handle_request(request: httpx.Request) -> httpx.Response:
        seen_requests.append(request)
        return httpx.Response(200, json=payload)

    response = await providers.call_provider("gemini", "gemini-3.5-flash-lite", MESSAGES, transport=httpx.MockTransport(handle_request))

    assert response.text == "answer"
    assert (response.input_tokens, response.output_tokens) == (12, 3)
    sent_body = json.loads(seen_requests[0].content)
    assert sent_body["systemInstruction"]["parts"][0]["text"] == "Be brief."
    assert sent_body["contents"][0]["role"] == "user"
    assert seen_requests[0].headers["x-goog-api-key"] == "test-key"
    assert ":generateContent" in str(seen_requests[0].url)


async def test_gemini_stream_parses_chunks():
    chunks = [
        json.dumps({"candidates": [{"content": {"parts": [{"text": "one "}]}}]}),
        json.dumps({"candidates": [{"content": {"parts": [{"text": "two"}]}}], "usageMetadata": {"promptTokenCount": 4, "candidatesTokenCount": 2}}),
    ]

    def handle_request(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=sse_body(chunks), headers={"content-type": "text/event-stream"})

    streamed_tokens: list[str] = []
    response = await providers.call_provider(
        "gemini", "gemini-3.8-flash", MESSAGES, stream=True,
        on_token=streamed_tokens.append, transport=httpx.MockTransport(handle_request),
    )
    assert streamed_tokens == ["one ", "two"]
    assert response.output_tokens == 2


async def test_empty_completion_is_an_error():
    def handle_request(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": ""}}]})

    with pytest.raises(ProviderError, match="empty completion"):
        await providers.call_provider("nvidia", "nvidia/nemotron-3-super-120b-a12b", MESSAGES, transport=httpx.MockTransport(handle_request))


# ---------------------------------------------------------------------------
# Live pings (need real keys; skipped otherwise)
# ---------------------------------------------------------------------------

PING_MESSAGES = [{"role": "user", "content": "Reply with the single word: pong"}]


@pytest.mark.live
@pytest.mark.parametrize("provider_name,tier", [("nvidia", "fast"), ("nvidia", "strong"), ("gemini", "fast"), ("gemini", "strong")])
async def test_live_ping_tiered_provider(provider_name, tier):
    if not providers.has_api_key(provider_name):
        pytest.skip(f"no {provider_name} key")
    model = providers.resolve_model(provider_name, tier)
    response = await providers.call_provider(provider_name, model, PING_MESSAGES, stream=True, max_tokens=512)
    assert "pong" in response.text.lower()
    assert response.ttft_ms is not None


@pytest.mark.live
async def test_live_ping_openrouter_free_chain():
    if not providers.has_api_key("openrouter"):
        pytest.skip("no openrouter key")
    failures: list[str] = []
    for model_id in providers.openrouter_free_chain():
        try:
            response = await providers.call_provider("openrouter", model_id, PING_MESSAGES, max_tokens=512)
        except ProviderError as error:
            failures.append(str(error))
            continue
        assert response.text.strip()
        return
    pytest.fail(f"every OpenRouter free model failed: {failures}")


@pytest.mark.live
async def test_live_complete_through_router():
    if not any(providers.has_api_key(name) for name in ("nvidia", "gemini", "openrouter")):
        pytest.skip("no provider keys")
    result = await complete("supervisor", PING_MESSAGES, TaskProfile("simple"))
    assert result.error is None, result.attempts
    assert result.text.strip()


async def test_timeout_moves_on_without_retry_and_cools_the_model_down(monkeypatch, all_keys_present):
    calls: list[tuple[str, str]] = []

    async def scripted_call(provider, model, messages, **kwargs):
        calls.append((provider, model))
        if provider == "gemini":
            raise httpx.ReadTimeout("no bytes")
        return ProviderResponse("nvidia answer", 1, 1, 5.0, None)

    monkeypatch.setattr(providers, "call_provider", scripted_call)
    first = await complete("research", MESSAGES, TaskProfile("complex", 2_000), node="research")
    assert first.provider == "nvidia"
    # One gemini call only: a timeout is not retried.
    assert [call[0] for call in calls].count("gemini") == 1

    calls.clear()
    second = await complete("research", MESSAGES, TaskProfile("complex", 2_000), node="research")
    # The timed-out model is cooling down, so the next request goes to nvidia first.
    assert calls[0][0] == "nvidia"
    assert second.provider == "nvidia"


def test_http_503_marks_error_overloaded():
    response = httpx.Response(503, text="high demand")
    error = providers.error_from_response("gemini", "m", 503, "high demand", response)
    assert error.overloaded
    assert not error.rate_limited
    not_overloaded = providers.error_from_response("gemini", "m", 500, "boom", httpx.Response(500))
    assert not not_overloaded.overloaded


async def test_unavailable_model_404_is_skipped_on_the_next_request(monkeypatch, all_keys_present):
    # Live finding: pinned OpenRouter ids returned 404 "no endpoints" (account data policy) on
    # every summary call, costing ~4 s each time. A 404 now cools the model down.
    calls: list[tuple[str, str]] = []

    async def scripted_call(provider, model, messages, **kwargs):
        calls.append((provider, model))
        if provider == "openrouter":
            raise providers.error_from_response(provider, model, 404, "0 endpoints", httpx.Response(404))
        return ProviderResponse("gemini answer", 1, 1, 5.0, None)

    monkeypatch.setattr(providers, "call_provider", scripted_call)
    first = await complete("summary", MESSAGES, TaskProfile("simple", 2_000), node="summary")
    assert first.provider == "gemini"
    calls.clear()
    second = await complete("summary", MESSAGES, TaskProfile("simple", 2_000), node="summary")
    assert second.provider == "gemini"
    # Every OpenRouter model is cooling down, so the second request goes straight to Gemini.
    assert calls[0][0] == "gemini"
    cooldown_left = max(router.rate_limited_until.values()) - router.time.monotonic()
    assert cooldown_left > 100


# ---------------------------------------------------------------------------
# Grader fixes: cooldown visibility, llm_result, 402, 503, logging, labels, budget
# ---------------------------------------------------------------------------


async def test_cooldown_reorder_emits_a_skipped_fallback_trace(monkeypatch, captured_events, all_keys_present):
    # Live finding: llm_route announced openrouter/free, Gemini answered, and no trace said why.
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(providers, "call_provider", make_scripted_provider({"gemini": "gemini answer"}, calls))
    primary = decide_route("summary", TaskProfile("simple", 2_000))
    router.mark_rate_limited((primary.provider, primary.model), 45.0)

    result = await complete("summary", MESSAGES, TaskProfile("simple", 2_000), node="summary")

    route_trace = traces_of_kind(captured_events, "llm_route")[0]
    skip_traces = [trace for trace in traces_of_kind(captured_events, "llm_fallback") if trace["detail"].get("reason") == "cooldown"]
    assert len(skip_traces) == 1
    skip_detail = skip_traces[0]["detail"]
    assert skip_detail["skipped"] is True
    assert (skip_detail["provider"], skip_detail["model"]) == (primary.provider, primary.model)
    assert 40 < skip_detail["cooldown_remaining_s"] <= 45
    # The next candidate is the one actually tried in the primary's place.
    assert tuple(skip_detail["next"]) == calls[0]
    assert skip_detail["call_id"] == route_trace["detail"]["call_id"]
    assert skip_detail["stream_restarted"] is False
    assert "skipped" in skip_traces[0]["summary"]
    # The cooling model is moved to the back, not dropped: it is not tried while others answer.
    # The answer is the next candidate's (scripted: Gemini says "gemini answer", others "default ok").
    expected_text = "gemini answer" if calls[0][0] == "gemini" else "default ok"
    assert result.text == expected_text
    assert (primary.provider, primary.model) not in calls


async def test_cooldown_entry_already_last_is_not_reported_as_moved(monkeypatch, captured_events, all_keys_present):
    monkeypatch.setattr(providers, "call_provider", make_scripted_provider({}, []))
    decision = decide_route("supervisor", TaskProfile("simple"))
    last_candidate = decision.fallback_chain[-1]
    router.mark_rate_limited(last_candidate, 30.0)

    await complete("supervisor", MESSAGES, TaskProfile("simple"))

    assert not traces_of_kind(captured_events, "llm_fallback")


async def test_llm_result_trace_reports_the_answering_model(monkeypatch, captured_events, all_keys_present):
    async def openrouter_free_router(provider, model, messages, **kwargs):
        return ProviderResponse("summary text", 3, 2, 12.0, None, served_model="google/gemma-4-31b-it:free")

    monkeypatch.setattr(providers, "call_provider", openrouter_free_router)
    await complete("summary", MESSAGES, TaskProfile("simple", 2_000), node="summary")

    route_trace = traces_of_kind(captured_events, "llm_route")[0]
    result_traces = traces_of_kind(captured_events, "llm_result")
    assert len(result_traces) == 1
    detail = result_traces[0]["detail"]
    assert detail["call_id"] == route_trace["detail"]["call_id"]
    assert detail["role"] == "summary"
    assert detail["ok"] is True
    assert detail["provider"] == "openrouter"
    assert detail["model"] == "openrouter/free -> google/gemma-4-31b-it:free"
    assert detail["requested_model"] == "openrouter/free"
    assert detail["served_model"] == "google/gemma-4-31b-it:free"
    assert detail["attempts"] == 1
    assert detail["error"] is None
    assert isinstance(detail["latency_ms"], float)
    # llm_result is the last trace of the call.
    assert captured_events[-1]["data"]["kind"] == "llm_result"


async def test_llm_result_on_total_failure_carries_the_error(monkeypatch, captured_events):
    def no_key(provider: str) -> bool:
        return False

    monkeypatch.setattr(providers, "has_api_key", no_key)
    result = await complete("judge", MESSAGES, TaskProfile("complex"), node="eval")

    result_detail = traces_of_kind(captured_events, "llm_result")[0]["detail"]
    assert result_detail["ok"] is False
    assert result_detail["provider"] == ""
    assert result_detail["error"] == result.error
    assert result_detail["attempts"] == len(result.attempts)
    call_ids = {trace["detail"]["call_id"] for trace in traces_of_kind(captured_events, "llm_fallback")}
    assert call_ids == {result_detail["call_id"]}


async def test_each_complete_call_gets_its_own_call_id(monkeypatch, captured_events, all_keys_present):
    monkeypatch.setattr(providers, "call_provider", make_scripted_provider({}, []))
    await complete("data", MESSAGES, TaskProfile("simple"), node="data")
    await complete("data", MESSAGES, TaskProfile("complex"), node="data")
    route_ids = [trace["detail"]["call_id"] for trace in traces_of_kind(captured_events, "llm_route")]
    result_ids = [trace["detail"]["call_id"] for trace in traces_of_kind(captured_events, "llm_result")]
    assert len(set(route_ids)) == 2
    assert route_ids == result_ids


def test_http_402_cools_the_model_down_like_404():
    error = providers.error_from_response("openrouter", "openrouter/free", 402, "Insufficient credits", httpx.Response(402))
    assert error.overloaded
    assert not error.retryable
    assert error.retry_after_seconds == providers.UNAVAILABLE_MODEL_COOLDOWN_SECONDS


async def test_primary_out_of_credits_402_still_answers_from_fallback(monkeypatch, captured_events):
    # Course rule: force the primary to 402 and check the turn still gets a real answer.
    keyed_settings = replace(providers.settings, nvidia_api_key="nvapi-test", google_api_key="g-test", openrouter_api_key="sk-or-test")
    monkeypatch.setattr(providers, "settings", keyed_settings)
    real_call_provider = providers.call_provider
    calls: list[tuple[str, str]] = []

    def handle_request(request: httpx.Request) -> httpx.Response:
        if "generativelanguage" in str(request.url):
            return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": "fallback plan"}]}}]})
        return httpx.Response(402, json={"error": {"code": 402, "message": "Insufficient credits"}})

    async def call_through_mock_http(provider, model, messages, **kwargs):
        calls.append((provider, model))
        kwargs["transport"] = httpx.MockTransport(handle_request)
        return await real_call_provider(provider, model, messages, **kwargs)

    monkeypatch.setattr(providers, "call_provider", call_through_mock_http)
    result = await complete("supervisor", MESSAGES, TaskProfile("simple"), node="supervisor")

    assert result.error is None
    assert result.text == "fallback plan"
    assert result.provider == "gemini"
    assert result.decision.provider == "nvidia"
    assert result.attempts[0]["status_code"] == 402
    # 402 is not retried: the out-of-credits primary was called exactly once.
    assert [call[0] for call in calls].count("nvidia") == 1
    nvidia_primary = (result.decision.provider, result.decision.model)
    assert router.cooldown_remaining_seconds(nvidia_primary) > 100
    result_detail = traces_of_kind(captured_events, "llm_result")[0]["detail"]
    assert (result_detail["ok"], result_detail["provider"]) == (True, "gemini")


def test_http_503_is_not_retried_and_cools_for_30_seconds():
    error = providers.error_from_response("gemini", "m", 503, "high demand", httpx.Response(503))
    assert error.overloaded
    assert not error.retryable
    assert error.retry_after_seconds == 30.0
    # Other 5xx keep their single transient retry.
    bad_gateway = providers.error_from_response("gemini", "m", 502, "bad gateway", httpx.Response(502))
    assert bad_gateway.retryable


async def test_503_moves_on_after_one_call(monkeypatch, all_keys_present):
    calls: list[tuple[str, str]] = []

    async def scripted_call(provider, model, messages, **kwargs):
        calls.append((provider, model))
        if provider == "nvidia":
            raise providers.error_from_response(provider, model, 503, "high demand", httpx.Response(503))
        return ProviderResponse("gemini answer", 1, 1, 5.0, None)

    monkeypatch.setattr(providers, "call_provider", scripted_call)
    result = await complete("supervisor", MESSAGES, TaskProfile("simple"))

    assert result.provider == "gemini"
    assert [call[0] for call in calls].count("nvidia") == 1
    cooldown_left = router.cooldown_remaining_seconds((result.decision.provider, result.decision.model))
    assert 25 < cooldown_left <= 30


async def test_fallback_and_total_failure_are_logged_without_secrets(monkeypatch, caplog, all_keys_present):
    secret_prompt = [{"role": "user", "content": "TOP-SECRET-PROMPT-TEXT"}]
    script = {
        "nvidia": ProviderError("nvidia", "m", "body echoing TOP-SECRET-PROMPT-TEXT", status_code=429),
        "gemini": ProviderError("gemini", "m", "boom", status_code=500),
        "openrouter": ProviderError("openrouter", "m", "nope", status_code=400),
    }
    monkeypatch.setattr(providers, "call_provider", make_scripted_provider(script, []))
    caplog.set_level("WARNING", logger="src.llm.router")

    await complete("supervisor", secret_prompt, TaskProfile("simple"))

    warning_lines = [record.getMessage() for record in caplog.records if record.name == "src.llm.router"]
    fallback_lines = [line for line in warning_lines if line.startswith("LLM fallback: ")]
    assert fallback_lines[0].startswith("LLM fallback: nvidia/")
    assert "(HTTP 429), trying gemini/" in fallback_lines[0]
    assert any(line.startswith("LLM call failed: role=supervisor") for line in warning_lines)
    assert not any("TOP-SECRET" in line for line in warning_lines)


def test_route_summary_does_not_repeat_the_provider(captured_events):
    decision = decide_route("supervisor", TaskProfile("simple"))
    router.emit_route_trace(decision, "supervisor", "abc")
    summary = traces_of_kind(captured_events, "llm_route")[0]["summary"]
    assert decision.model.startswith("nvidia/")
    assert summary == f"supervisor -> {decision.model} [cost]"
    assert "nvidia/nvidia/" not in summary
    assert providers.candidate_label("openrouter", "openrouter/free") == "openrouter/free"
    assert providers.candidate_label("openrouter", "nvidia/nemotron-3-super-120b-a12b:free") == "openrouter/nvidia/nemotron-3-super-120b-a12b:free"
    assert providers.candidate_label("gemini", "gemini-3.8-flash") == "gemini/gemini-3.8-flash"


async def test_fallback_summary_does_not_repeat_the_provider(monkeypatch, captured_events, all_keys_present):
    rate_limit_error = ProviderError("gemini", "gemini-3.8-flash", "quota", status_code=429)
    monkeypatch.setattr(providers, "call_provider", make_scripted_provider({"gemini": rate_limit_error}, []))
    await complete("research", MESSAGES, TaskProfile("complex", 2_000), node="research")
    fallback_summary = traces_of_kind(captured_events, "llm_fallback")[0]["summary"]
    assert "next: nvidia/nemotron-3-ultra" in fallback_summary
    assert "nvidia/nvidia/" not in fallback_summary


async def test_time_budget_stops_the_fallback_walk(monkeypatch, captured_events, all_keys_present):
    fake_now = [1000.0]
    calls: list[tuple[str, str]] = []

    def fake_budget_clock() -> float:
        return fake_now[0]

    async def slow_failing_call(provider, model, messages, **kwargs):
        # Each failed attempt "takes" 50 s of fake time; nothing really sleeps.
        calls.append((provider, model))
        fake_now[0] += 50.0
        raise ProviderError(provider, model, "server", status_code=500)

    monkeypatch.setattr(router, "budget_clock", fake_budget_clock)
    monkeypatch.setattr(providers, "call_provider", slow_failing_call)
    result = await complete("supervisor", MESSAGES, TaskProfile("simple"), node="supervisor")

    chain_length = 1 + len(result.decision.fallback_chain)
    # 90 s budget: attempt 1 ends at 50 s, attempt 2 at 100 s, then the walk stops (these 500s
    # are not marked retryable, so each candidate is called once).
    tried_candidates = {call for call in calls}
    assert len(tried_candidates) == 2
    assert len(result.attempts) == 2 < chain_length
    assert result.error.startswith("all 2 provider attempts failed")
    assert "time budget" in result.error
    result_detail = traces_of_kind(captured_events, "llm_result")[0]["detail"]
    assert result_detail["budget_exhausted"] is True
    assert result_detail["ok"] is False


async def test_slow_attempt_is_cut_at_the_remaining_budget(monkeypatch, all_keys_present):
    monkeypatch.setattr(router, "TOTAL_CALL_BUDGET_SECONDS", 0.2)
    calls: list[str] = []

    async def hanging_then_fast(provider, model, messages, **kwargs):
        calls.append(provider)
        if provider == "nvidia":
            await asyncio.sleep(5)
        return ProviderResponse("late", 1, 1, 5.0, None)

    monkeypatch.setattr(providers, "call_provider", hanging_then_fast)
    started = router.time.monotonic()
    result = await complete("supervisor", MESSAGES, TaskProfile("simple"))
    elapsed = router.time.monotonic() - started

    assert elapsed < 1.0
    assert result.error is not None
    assert calls == ["nvidia"]
