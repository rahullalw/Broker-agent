"""OpenRouter client, failover and circuit breaker (Phase 3 §1, D1–D3).

The failover wraps **only** `chat.completions.create`, never the loop. That is
the whole design: a turn can be half-written by one model and finished by the
next without the loop noticing, so there is no tool re-execution, no duplicate
`tool_calls` rows and no double escalation.

The chain is three deep, tried in order:

1. `MODEL_PRIMARY` — Gemini 3.1 Flash Lite on BYOK quota (15 req/min, 500/day)
2. `MODEL_FALLBACK` — Gemini 3.5 Flash Lite, same lineage, same quota pool
3. `MODEL_LAST_RESORT` — a pinned `:free` model

Same lineage down the first two rungs means the closest formatting habits, which
is what keeps the Phase 4 byte-exact guard from behaving differently on the
fallback path. The `:free` rung exists for the demo: a quota exhausted between
takes should still answer rather than return a hold message.

`model_used` and `provider` come back with every response and land on the
assistant `messages` row — without them an odd guard rejection on a half-failed
turn is unexplainable.
"""
from __future__ import annotations

import asyncio
import time

import openai
from openai import OpenAI

from app import config

# After a model fails, skip it for this long rather than paying the timeout on
# every subsequent step of the same turn.
CIRCUIT_SECONDS = 60

# A 2-3 sentence WhatsApp reply needs nowhere near a Flash model's default
# ceiling, and that uncapped default is what OpenRouter's credit check bills
# against. Capping keeps requests moderate so the free quota survives the demos.
MAX_OUTPUT_TOKENS = 1024

_client: OpenAI | None = None
_down_until: dict[str, float] = {}


class LLMUnavailable(RuntimeError):
    """Every model in the chain failed. The loop turns this into a graceful hold
    (D20) — infrastructure trouble is not a reason to escalate a lead."""


def client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=config.OPENROUTER_API_KEY,
            timeout=float(config.LLM_TIMEOUT_S),
        )
    return _client


def reset_circuit() -> None:
    _down_until.clear()


def model_chain() -> list[str]:
    """Configured models in preference order, unset slots dropped."""
    ordered = [config.MODEL_PRIMARY, config.MODEL_FALLBACK, config.MODEL_LAST_RESORT]
    return [model for model in ordered if model]


def _should_fail_over(exc: Exception) -> bool:
    """429, 5xx and timeouts are transient and worth the next model. A 400 is a
    payload the next model would reject too."""
    if isinstance(exc, (openai.APITimeoutError, openai.APIConnectionError)):
        return True
    if isinstance(exc, openai.RateLimitError):
        return True
    if isinstance(exc, openai.APIStatusError):
        return exc.response.status_code >= 500
    return False


def _create(model: str, messages: list[dict], tools: list[dict]):
    kwargs = {
        "model": model,
        "messages": messages,
        "temperature": 0,                   # D25 — the demo has to be reproducible
        "max_tokens": MAX_OUTPUT_TOKENS,
    }
    if tools:
        kwargs["tools"] = tools
    return client().chat.completions.create(**kwargs)


def _attempt(model: str, messages: list[dict], tools: list[dict]):
    response = _create(model, messages, tools)
    # OpenRouter reports the serving provider as an extra field on the body.
    return response, model, getattr(response, "provider", None)


def _candidates() -> list[str]:
    """Models worth trying now. If every breaker is open the last rung is still
    attempted — refusing to call anything is worse than one wasted request."""
    chain = model_chain()
    now = time.monotonic()
    open_circuits = [m for m in chain if now >= _down_until.get(m, 0.0)]
    return open_circuits or chain[-1:]


async def llm_call(messages: list[dict], tools: list[dict] | None = None):
    """One completion. Returns `(response, model_used, provider)`."""
    tools = tools or []
    candidates = _candidates()
    if not candidates:
        raise LLMUnavailable("no model is configured")

    last_failure: Exception | None = None
    for model in candidates:
        try:
            return await asyncio.to_thread(_attempt, model, messages, tools)
        except Exception as exc:
            if not _should_fail_over(exc):
                raise
            _down_until[model] = time.monotonic() + CIRCUIT_SECONDS
            last_failure = exc

    raise LLMUnavailable(
        f"every model failed ({', '.join(candidates)}): {last_failure}"
    ) from last_failure
