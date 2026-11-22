"""Thin async provider clients for the LLM router.

Three free providers, all called with raw httpx so time-to-first-token and token usage are
measured the same way everywhere:

- NVIDIA NIM: OpenAI-compatible chat completions at settings.nvidia_base_url
- OpenRouter: OpenAI-compatible, `:free` models only (paid ids are dropped, never called)
- Google Gemini: REST generateContent / streamGenerateContent (SSE)

Every failure is raised as ProviderError so the router can decide whether to move down the
fallback chain. This module never decides routing; it only calls one (provider, model).
"""

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator, Callable, Literal

import httpx

from src.config import settings

ProviderName = Literal["nvidia", "gemini", "openrouter"]
ModelTier = Literal["fast", "strong"]
TokenCallback = Callable[[str], None]

# Verified 2026-10-25 (sources in docs/routing_notes.md). Used when .env leaves the id empty.
DEFAULT_MODELS: dict[str, dict[str, str]] = {
    "nvidia": {
        "fast": "nvidia/nemotron-3.5-lightning-30b-a3b",
        "strong": "nvidia/nemotron-3-ultra-550b-a55b",
    },
    "gemini": {
        "fast": "gemini-3.5-flash-lite",
        "strong": "gemini-3.8-flash",
    },
}

# OpenRouter's own free router: it picks whichever `:free` model is available, so the chain
# survives free models being retired. It is priced at $0 and can only reach free models.
OPENROUTER_FREE_ROUTER_ID = "openrouter/free"

# The free router alone: it already picks among the free models, and if it is rate limited the
# router falls over to NVIDIA or Gemini instead of trying pinned free ids one by one.
DEFAULT_OPENROUTER_FREE_MODELS: list[str] = [OPENROUTER_FREE_ROUTER_ID]

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
OPENROUTER_APP_URL = "https://github.com/csci599/multi-agent-rag"
OPENROUTER_APP_TITLE = "CSCI 599 Multi-Agent RAG"

TOTAL_TIMEOUT_SECONDS = 60.0
CONNECT_TIMEOUT_SECONDS = 10.0
# Max silence between bytes. A free-tier model that says nothing for this long is usually
# overloaded, and the next provider answers sooner than waiting longer would.
READ_TIMEOUT_SECONDS = 20.0
UNAVAILABLE_MODEL_COOLDOWN_SECONDS = 120.0
# 503 "high demand" and timeouts: skip the model for a short while, but do not retry it now.
OVERLOADED_MODEL_COOLDOWN_SECONDS = 30.0


@dataclass
class ProviderResponse:
    text: str
    input_tokens: int | None
    output_tokens: int | None
    latency_ms: float
    ttft_ms: float | None
    # The model that actually answered, as the provider reports it (differs from the requested
    # id when OpenRouter's free router picked one).
    served_model: str | None = None


# Status codes that put a (provider, model) in cooldown instead of being retried right away:
# 402 out of credits, 404 model unavailable to this account, 503 overloaded.
COOLDOWN_STATUS_CODES: frozenset[int] = frozenset({402, 404, 503})


class ProviderError(Exception):
    """One failed call. `rate_limited` means 429/quota: the router moves on without retrying."""

    def __init__(
        self,
        provider: str,
        model: str,
        message: str,
        status_code: int | None = None,
        retryable: bool = False,
        retry_after_seconds: float | None = None,
        overloaded: bool = False,
    ):
        super().__init__(message)
        self.provider = provider
        self.model = model
        self.message = message
        self.status_code = status_code
        self.retryable = retryable
        self.retry_after_seconds = retry_after_seconds
        # Timeout, 503 "high demand", 402 out of credits or 404 unavailable: skip this model for
        # a while across requests (the router puts it in cooldown).
        self.overloaded = overloaded or status_code in COOLDOWN_STATUS_CODES

    @property
    def rate_limited(self) -> bool:
        return self.status_code == 429

    def __str__(self) -> str:
        status_part = f"HTTP {self.status_code}: " if self.status_code is not None else ""
        return f"{candidate_label(self.provider, self.model)}: {status_part}{self.message}"


def candidate_label(provider: str, model: str) -> str:
    """'provider/model', without repeating the provider when the model id already starts with it.

    NVIDIA ids are 'nvidia/nemotron-...' and the free router is 'openrouter/free', so a plain
    f"{provider}/{model}" read 'nvidia/nvidia/nemotron-...'.
    """
    if model.startswith(f"{provider}/"):
        return model
    return f"{provider}/{model}"


# ---------------------------------------------------------------------------
# Keys and model resolution
# ---------------------------------------------------------------------------


def api_key_for(provider: str) -> str:
    key_by_provider = {
        "nvidia": settings.nvidia_api_key,
        "gemini": settings.google_api_key,
        "openrouter": settings.openrouter_api_key,
    }
    return key_by_provider.get(provider, "")


def has_api_key(provider: str) -> bool:
    key = api_key_for(provider)
    # .env.example ships "replace-me" placeholders; treat those as missing, not as a real key.
    return bool(key) and "replace-me" not in key


def resolve_model(provider: str, tier: ModelTier) -> str:
    configured_by_provider = {
        "nvidia": {"fast": settings.nvidia_model_fast, "strong": settings.nvidia_model_strong},
        "gemini": {"fast": settings.gemini_model_fast, "strong": settings.gemini_model_strong},
    }
    configured = configured_by_provider[provider][tier]
    # An unfilled "<verified-...>" placeholder copied from .env.example counts as unset.
    if configured and not configured.startswith("<"):
        return configured
    return DEFAULT_MODELS[provider][tier]


def is_free_openrouter_model(model_id: str) -> bool:
    return model_id.endswith(":free") or model_id == OPENROUTER_FREE_ROUTER_ID


def openrouter_free_chain() -> list[str]:
    """Configured OpenRouter models, keeping only free ids. Falls back to verified defaults."""
    configured = settings.openrouter_free_models
    free_only = [model_id for model_id in configured if is_free_openrouter_model(model_id)]
    if free_only:
        return free_only
    return list(DEFAULT_OPENROUTER_FREE_MODELS)


# ---------------------------------------------------------------------------
# Shared HTTP helpers
# ---------------------------------------------------------------------------


def build_timeout() -> httpx.Timeout:
    return httpx.Timeout(READ_TIMEOUT_SECONDS, connect=CONNECT_TIMEOUT_SECONDS)


def parse_retry_after(response: httpx.Response) -> float | None:
    raw_value = response.headers.get("retry-after", "")
    try:
        return float(raw_value)
    except ValueError:
        return None


def classify_status(status_code: int | None, retry_after_seconds: float | None) -> tuple[bool, float | None, bool]:
    """Return (retryable, cooldown seconds, overloaded) for one HTTP error status."""
    if status_code is None:
        return False, retry_after_seconds, False
    # 404: the model is not available to this account (OpenRouter "no endpoints" under the
    # account's data policy, an NVIDIA function not enabled). 402: the account is out of
    # credits. Neither fixes itself within one request, so the router skips the model for the
    # longest cooldown instead of paying for the failed call on every request.
    if status_code in (402, 404):
        return False, UNAVAILABLE_MODEL_COOLDOWN_SECONDS, True
    # 503 "high demand": not retried (the same model would most likely fail again), cooled
    # down for 30 s unless the server says how long.
    if status_code == 503:
        return False, retry_after_seconds or OVERLOADED_MODEL_COOLDOWN_SECONDS, True
    # Other 5xx and 408 are transient and get one short retry; 429 is handled by the router
    # as "move on now".
    is_transient = status_code >= 500 or status_code == 408
    return is_transient, retry_after_seconds, False


def error_from_response(provider: str, model: str, status_code: int, body_text: str, response: httpx.Response) -> ProviderError:
    short_body = body_text[:300]
    retryable, retry_after_seconds, overloaded = classify_status(status_code, parse_retry_after(response))
    return ProviderError(
        provider,
        model,
        short_body or "empty error body",
        status_code=status_code,
        retryable=retryable,
        retry_after_seconds=retry_after_seconds,
        overloaded=overloaded,
    )


def error_from_exception(provider: str, model: str, error: Exception) -> ProviderError:
    if isinstance(error, ProviderError):
        return error
    if isinstance(error, (httpx.TimeoutException, asyncio.TimeoutError, TimeoutError)):
        # Not retried: the same overloaded model would most likely time out again.
        return ProviderError(provider, model, f"timeout: {type(error).__name__}", overloaded=True)
    if isinstance(error, httpx.TransportError):
        return ProviderError(provider, model, f"transport error: {error}", retryable=True)
    return ProviderError(provider, model, f"{type(error).__name__}: {error}")


async def iterate_sse_data(response: httpx.Response) -> AsyncIterator[str]:
    """Yield the payload of each `data:` line of a server-sent-event stream."""
    async for line in response.aiter_lines():
        stripped_line = line.strip()
        if not stripped_line.startswith("data:"):
            continue
        payload = stripped_line[len("data:"):].strip()
        if payload:
            yield payload


def elapsed_ms(start_time: float) -> float:
    return (time.perf_counter() - start_time) * 1000.0


# ---------------------------------------------------------------------------
# OpenAI-compatible (NVIDIA NIM, OpenRouter)
# ---------------------------------------------------------------------------


def openai_compatible_endpoint(provider: str) -> tuple[str, dict[str, str]]:
    headers = {
        "Authorization": f"Bearer {api_key_for(provider)}",
        "Content-Type": "application/json",
    }
    if provider == "openrouter":
        # Optional attribution headers recommended by OpenRouter.
        headers["HTTP-Referer"] = OPENROUTER_APP_URL
        headers["X-Title"] = OPENROUTER_APP_TITLE
        base_url = settings.openrouter_base_url
    else:
        base_url = settings.nvidia_base_url
    trimmed_base = base_url.rstrip("/")
    return f"{trimmed_base}/chat/completions", headers


def openai_compatible_body(provider: str, model: str, messages: list[dict], stream: bool, max_tokens: int, temperature: float) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": stream,
    }
    if stream:
        # Ask for a final usage chunk so streamed calls still report token counts.
        body["stream_options"] = {"include_usage": True}
    if provider == "nvidia" and model.startswith("nvidia/nemotron"):
        # Nemotron 3.x reasons by default; turning thinking off keeps latency predictable.
        body["chat_template_kwargs"] = {"enable_thinking": False}
    if provider == "openrouter":
        # Hide reasoning text from the answer for models that think by default.
        body["reasoning"] = {"exclude": True}
    return body


def usage_from_openai_payload(payload: dict[str, Any]) -> tuple[int | None, int | None]:
    usage = payload.get("usage") or {}
    return usage.get("prompt_tokens"), usage.get("completion_tokens")


def text_from_openai_choice(choice: dict[str, Any], streaming: bool) -> str:
    container_key = "delta" if streaming else "message"
    container = choice.get(container_key) or {}
    content = container.get("content")
    return content if isinstance(content, str) else ""


async def call_openai_compatible(
    provider: str,
    model: str,
    messages: list[dict],
    stream: bool,
    on_token: TokenCallback | None,
    max_tokens: int,
    temperature: float,
    transport: httpx.AsyncBaseTransport | None = None,
) -> ProviderResponse:
    url, headers = openai_compatible_endpoint(provider)
    body = openai_compatible_body(provider, model, messages, stream, max_tokens, temperature)
    start_time = time.perf_counter()
    ttft_ms: float | None = None
    text_parts: list[str] = []
    input_tokens: int | None = None
    output_tokens: int | None = None
    served_model: str | None = None

    async with httpx.AsyncClient(timeout=build_timeout(), transport=transport) as client:
        if not stream:
            response = await client.post(url, headers=headers, json=body)
            if response.status_code != 200:
                raise error_from_response(provider, model, response.status_code, response.text, response)
            payload = response.json()
            served_model = payload.get("model") or None
            choices = payload.get("choices") or []
            for choice in choices:
                text_parts.append(text_from_openai_choice(choice, streaming=False))
            input_tokens, output_tokens = usage_from_openai_payload(payload)
        else:
            async with client.stream("POST", url, headers=headers, json=body) as response:
                if response.status_code != 200:
                    error_body = await response.aread()
                    error_text = error_body.decode("utf-8", errors="replace")
                    raise error_from_response(provider, model, response.status_code, error_text, response)
                async for data_line in iterate_sse_data(response):
                    if data_line == "[DONE]":
                        break
                    chunk = json.loads(data_line)
                    # OpenRouter reports mid-stream failures as an "error" object in a chunk.
                    if "error" in chunk:
                        error_info = chunk["error"] or {}
                        error_code = error_info.get("code")
                        status_code = error_code if isinstance(error_code, int) else None
                        # Same cooldown rules as an HTTP error status (a 402 can arrive here too).
                        retryable, retry_after_seconds, overloaded = classify_status(status_code, None)
                        raise ProviderError(
                            provider,
                            model,
                            str(error_info.get("message", error_info)),
                            status_code=status_code,
                            retryable=retryable,
                            retry_after_seconds=retry_after_seconds,
                            overloaded=overloaded,
                        )
                    if served_model is None and chunk.get("model"):
                        served_model = chunk["model"]
                    chunk_input, chunk_output = usage_from_openai_payload(chunk)
                    if chunk_input is not None:
                        input_tokens, output_tokens = chunk_input, chunk_output
                    for choice in chunk.get("choices") or []:
                        token_text = text_from_openai_choice(choice, streaming=True)
                        if not token_text:
                            continue
                        if ttft_ms is None:
                            ttft_ms = elapsed_ms(start_time)
                        text_parts.append(token_text)
                        if on_token is not None:
                            on_token(token_text)

    full_text = "".join(text_parts)
    return ProviderResponse(full_text, input_tokens, output_tokens, elapsed_ms(start_time), ttft_ms, served_model)


# ---------------------------------------------------------------------------
# Google Gemini (REST)
# ---------------------------------------------------------------------------


def gemini_body(messages: list[dict], max_tokens: int, temperature: float) -> dict[str, Any]:
    system_texts: list[str] = []
    contents: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role", "user")
        content = str(message.get("content", ""))
        if role == "system":
            system_texts.append(content)
            continue
        gemini_role = "model" if role == "assistant" else "user"
        contents.append({"role": gemini_role, "parts": [{"text": content}]})

    body: dict[str, Any] = {
        "generationConfig": {"maxOutputTokens": max_tokens, "temperature": temperature},
    }
    joined_system = "\n\n".join(system_texts)
    if contents:
        body["contents"] = contents
        if joined_system:
            body["systemInstruction"] = {"parts": [{"text": joined_system}]}
    else:
        # Gemini requires at least one content turn; a system-only prompt becomes the user turn.
        body["contents"] = [{"role": "user", "parts": [{"text": joined_system}]}]
    return body


def text_from_gemini_payload(payload: dict[str, Any]) -> str:
    candidates = payload.get("candidates") or []
    text_parts: list[str] = []
    for candidate in candidates:
        candidate_content = candidate.get("content") or {}
        for part in candidate_content.get("parts") or []:
            # Thought summaries (part.thought == True) are not part of the answer.
            if part.get("thought"):
                continue
            part_text = part.get("text")
            if isinstance(part_text, str):
                text_parts.append(part_text)
    return "".join(text_parts)


def usage_from_gemini_payload(payload: dict[str, Any]) -> tuple[int | None, int | None]:
    usage = payload.get("usageMetadata") or {}
    return usage.get("promptTokenCount"), usage.get("candidatesTokenCount")


def raise_if_gemini_blocked(payload: dict[str, Any], model: str) -> None:
    prompt_feedback = payload.get("promptFeedback") or {}
    block_reason = prompt_feedback.get("blockReason")
    if block_reason:
        raise ProviderError("gemini", model, f"prompt blocked: {block_reason}")


async def call_gemini(
    model: str,
    messages: list[dict],
    stream: bool,
    on_token: TokenCallback | None,
    max_tokens: int,
    temperature: float,
    transport: httpx.AsyncBaseTransport | None = None,
) -> ProviderResponse:
    headers = {"x-goog-api-key": api_key_for("gemini"), "Content-Type": "application/json"}
    body = gemini_body(messages, max_tokens, temperature)
    start_time = time.perf_counter()
    ttft_ms: float | None = None
    text_parts: list[str] = []
    input_tokens: int | None = None
    output_tokens: int | None = None

    async with httpx.AsyncClient(timeout=build_timeout(), transport=transport) as client:
        if not stream:
            url = f"{GEMINI_BASE_URL}/models/{model}:generateContent"
            response = await client.post(url, headers=headers, json=body)
            if response.status_code != 200:
                raise error_from_response("gemini", model, response.status_code, response.text, response)
            payload = response.json()
            raise_if_gemini_blocked(payload, model)
            text_parts.append(text_from_gemini_payload(payload))
            input_tokens, output_tokens = usage_from_gemini_payload(payload)
        else:
            url = f"{GEMINI_BASE_URL}/models/{model}:streamGenerateContent?alt=sse"
            async with client.stream("POST", url, headers=headers, json=body) as response:
                if response.status_code != 200:
                    error_body = await response.aread()
                    error_text = error_body.decode("utf-8", errors="replace")
                    raise error_from_response("gemini", model, response.status_code, error_text, response)
                async for data_line in iterate_sse_data(response):
                    chunk = json.loads(data_line)
                    raise_if_gemini_blocked(chunk, model)
                    chunk_input, chunk_output = usage_from_gemini_payload(chunk)
                    if chunk_input is not None:
                        input_tokens, output_tokens = chunk_input, chunk_output
                    token_text = text_from_gemini_payload(chunk)
                    if not token_text:
                        continue
                    if ttft_ms is None:
                        ttft_ms = elapsed_ms(start_time)
                    text_parts.append(token_text)
                    if on_token is not None:
                        on_token(token_text)

    full_text = "".join(text_parts)
    return ProviderResponse(full_text, input_tokens, output_tokens, elapsed_ms(start_time), ttft_ms)


# ---------------------------------------------------------------------------
# Single entry point used by the router
# ---------------------------------------------------------------------------


async def call_provider(
    provider: str,
    model: str,
    messages: list[dict],
    stream: bool = False,
    on_token: TokenCallback | None = None,
    max_tokens: int = 1024,
    temperature: float = 0.2,
    transport: httpx.AsyncBaseTransport | None = None,
) -> ProviderResponse:
    """Call one (provider, model). Raises ProviderError on any failure, including empty output."""
    if provider == "openrouter" and not is_free_openrouter_model(model):
        # Hard guard: a paid OpenRouter id must never be sent, whatever the config says.
        raise ProviderError(provider, model, "refused: OpenRouter model is not a ':free' model")

    try:
        if provider == "gemini":
            pending_call = call_gemini(model, messages, stream, on_token, max_tokens, temperature, transport)
        elif provider in ("nvidia", "openrouter"):
            pending_call = call_openai_compatible(
                provider, model, messages, stream, on_token, max_tokens, temperature, transport
            )
        else:
            raise ProviderError(provider, model, "unknown provider")
        # httpx timeouts are per network operation; this caps the whole call.
        provider_response = await asyncio.wait_for(pending_call, timeout=TOTAL_TIMEOUT_SECONDS)
    except Exception as error:
        raise error_from_exception(provider, model, error) from error

    if not provider_response.text.strip():
        # Free models sometimes return 200 with no content (e.g. budget spent on reasoning).
        raise ProviderError(provider, model, "empty completion", retryable=False)
    return provider_response
