"""Phase 3 §1 — the OpenRouter client, failover and the circuit breaker.

Nothing here touches the network. The point of these tests is the *shape* of the
failover (D3): it wraps one `chat.completions.create` call and nothing else, so
the loop never learns a switch happened — no tool re-execution, no duplicate
`tool_calls` rows, no double escalation.

The chain is three deep. Two Gemini Flash-Lite models carry the demo on BYOK
quota (15 req/min, 500 req/day), and a pinned `:free` model sits underneath them
so an exhausted quota mid-demo still gets an answer rather than a hold message.
"""
import time
from types import SimpleNamespace

import httpx
import openai
import pytest

from app import config, llm

REQUEST = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")

MESSAGES = [{"role": "user", "content": "3BHK in Bopal"}]
TOOLS = [{"type": "function", "function": {"name": "search_properties"}}]

PRIMARY = "google/gemini-3.1-flash-lite"
FALLBACK = "google/gemini-3.5-flash-lite"
LAST_RESORT = "google/gemma-4-31b-it:free"


def status_error(cls, code):
    return cls("boom", response=httpx.Response(code, request=REQUEST), body=None)


def timeout():
    return openai.APITimeoutError(request=REQUEST)


def reply(text="hello", provider="Google"):
    return SimpleNamespace(
        choices=[SimpleNamespace(
            finish_reason="stop",
            message=SimpleNamespace(content=text, tool_calls=None),
        )],
        provider=provider,
    )


class FakeCompletions:
    """Records every create() and replays a scripted outcome per model."""

    def __init__(self, outcomes):
        self.outcomes = outcomes          # model id -> value or exception
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes[kwargs["model"]]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    @property
    def models(self):
        return [c["model"] for c in self.calls]


class FakeClient:
    def __init__(self, outcomes):
        self.completions = FakeCompletions(outcomes)
        self.chat = SimpleNamespace(completions=self.completions)


@pytest.fixture
def wire(monkeypatch):
    """Installs a fake client and resets every circuit."""
    monkeypatch.setattr(config, "MODEL_PRIMARY", PRIMARY)
    monkeypatch.setattr(config, "MODEL_FALLBACK", FALLBACK)
    monkeypatch.setattr(config, "MODEL_LAST_RESORT", LAST_RESORT)

    def install(outcomes):
        client = FakeClient(outcomes)
        monkeypatch.setattr(llm, "_client", client)
        return client

    llm.reset_circuit()
    yield install
    llm.reset_circuit()


class TestHappyPath:
    async def test_the_primary_serves_the_call(self, wire):
        client = wire({PRIMARY: reply()})
        response, model_used, provider = await llm.llm_call(MESSAGES, TOOLS)
        assert response.choices[0].message.content == "hello"
        assert model_used == PRIMARY
        assert provider == "Google"
        assert client.completions.models == [PRIMARY]

    async def test_temperature_is_zero_on_every_call(self, wire):
        client = wire({PRIMARY: reply()})
        await llm.llm_call(MESSAGES, TOOLS)
        assert client.completions.calls[0]["temperature"] == 0

    async def test_messages_and_tools_pass_through_untouched(self, wire):
        client = wire({PRIMARY: reply()})
        await llm.llm_call(MESSAGES, TOOLS)
        sent = client.completions.calls[0]
        assert sent["messages"] == MESSAGES
        assert sent["tools"] == TOOLS

    async def test_provider_is_none_when_openrouter_does_not_say(self, wire):
        wire({PRIMARY: SimpleNamespace(choices=[])})
        _, _, provider = await llm.llm_call(MESSAGES, TOOLS)
        assert provider is None

    async def test_no_routing_suffixes_are_appended(self, wire):
        client = wire({PRIMARY: reply()})
        await llm.llm_call(MESSAGES, TOOLS)
        model = client.completions.calls[0]["model"]
        assert not model.endswith(":nitro") and not model.endswith(":floor")


class TestOutputCap:
    """A 2-3 sentence WhatsApp reply needs nowhere near the model's default
    ceiling, and the uncapped default is what OpenRouter's credit check bills
    against."""

    async def test_every_call_caps_its_output(self, wire):
        client = wire({PRIMARY: reply()})
        await llm.llm_call(MESSAGES, TOOLS)
        assert client.completions.calls[0]["max_tokens"] == 1024

    async def test_the_cap_is_moderate_not_the_model_maximum(self, wire):
        assert llm.MAX_OUTPUT_TOKENS == 1024

    async def test_the_cap_survives_failover(self, wire):
        client = wire({PRIMARY: timeout(), FALLBACK: reply()})
        await llm.llm_call(MESSAGES, TOOLS)
        assert [c["max_tokens"] for c in client.completions.calls] == [1024, 1024]


class TestFailover:
    @pytest.mark.parametrize("failure", [
        status_error(openai.RateLimitError, 429),
        status_error(openai.InternalServerError, 500),
        status_error(openai.APIStatusError, 503),
        openai.APITimeoutError(request=REQUEST),
        openai.APIConnectionError(request=REQUEST),
    ], ids=["429", "500", "503", "timeout", "connection"])
    async def test_the_next_model_completes_the_call(self, wire, failure):
        wire({PRIMARY: failure, FALLBACK: reply(provider="Google AI Studio")})
        response, model_used, provider = await llm.llm_call(MESSAGES, TOOLS)
        assert response.choices[0].message.content == "hello"
        assert model_used == FALLBACK
        assert provider == "Google AI Studio"

    async def test_an_exhausted_quota_falls_all_the_way_to_the_free_model(self, wire):
        # 429 on both Gemini models is the 15/min limit biting mid-demo.
        client = wire({
            PRIMARY: status_error(openai.RateLimitError, 429),
            FALLBACK: status_error(openai.RateLimitError, 429),
            LAST_RESORT: reply(provider="Together"),
        })
        _, model_used, _ = await llm.llm_call(MESSAGES, TOOLS)
        assert model_used == LAST_RESORT
        assert client.completions.models == [PRIMARY, FALLBACK, LAST_RESORT]

    async def test_the_retry_carries_identical_messages_and_tools(self, wire):
        client = wire({
            PRIMARY: status_error(openai.RateLimitError, 429),
            FALLBACK: status_error(openai.RateLimitError, 429),
            LAST_RESORT: reply(),
        })
        await llm.llm_call(MESSAGES, TOOLS)
        for sent in client.completions.calls:
            assert sent["messages"] == MESSAGES
            assert sent["tools"] == TOOLS

    async def test_no_model_is_tried_twice_in_one_call(self, wire):
        client = wire({PRIMARY: timeout(), FALLBACK: reply()})
        await llm.llm_call(MESSAGES, TOOLS)
        assert client.completions.models == [PRIMARY, FALLBACK]

    async def test_a_bad_request_is_not_failed_over(self, wire):
        # A 400 means the payload is wrong; the next model would reject it too.
        client = wire({
            PRIMARY: status_error(openai.BadRequestError, 400),
            FALLBACK: reply(),
        })
        with pytest.raises(openai.BadRequestError):
            await llm.llm_call(MESSAGES, TOOLS)
        assert client.completions.models == [PRIMARY]

    async def test_every_model_failing_raises_llm_unavailable(self, wire):
        wire({PRIMARY: timeout(), FALLBACK: timeout(), LAST_RESORT: timeout()})
        with pytest.raises(llm.LLMUnavailable):
            await llm.llm_call(MESSAGES, TOOLS)

    async def test_an_unset_slot_is_skipped(self, wire, monkeypatch):
        monkeypatch.setattr(config, "MODEL_FALLBACK", "")
        client = wire({PRIMARY: timeout(), LAST_RESORT: reply()})
        _, model_used, _ = await llm.llm_call(MESSAGES, TOOLS)
        assert model_used == LAST_RESORT
        assert client.completions.models == [PRIMARY, LAST_RESORT]

    async def test_a_lone_primary_raises_after_one_attempt(self, wire, monkeypatch):
        monkeypatch.setattr(config, "MODEL_FALLBACK", "")
        monkeypatch.setattr(config, "MODEL_LAST_RESORT", "")
        client = wire({PRIMARY: timeout()})
        with pytest.raises(llm.LLMUnavailable):
            await llm.llm_call(MESSAGES, TOOLS)
        assert len(client.completions.calls) == 1


class TestCircuitBreaker:
    async def test_a_failed_model_is_skipped_next_time(self, wire):
        client = wire({PRIMARY: status_error(openai.RateLimitError, 429),
                       FALLBACK: reply()})
        await llm.llm_call(MESSAGES, TOOLS)
        await llm.llm_call(MESSAGES, TOOLS)
        # Attempt 1 primary then fallback; attempt 2 straight to fallback.
        assert client.completions.models == [PRIMARY, FALLBACK, FALLBACK]

    async def test_two_burnt_models_leave_the_free_one_serving(self, wire):
        client = wire({
            PRIMARY: status_error(openai.RateLimitError, 429),
            FALLBACK: status_error(openai.RateLimitError, 429),
            LAST_RESORT: reply(),
        })
        await llm.llm_call(MESSAGES, TOOLS)
        await llm.llm_call(MESSAGES, TOOLS)
        assert client.completions.models[-1:] == [LAST_RESORT]
        assert client.completions.models.count(PRIMARY) == 1

    async def test_the_breaker_stays_open_for_about_a_minute(self, wire):
        wire({PRIMARY: status_error(openai.RateLimitError, 429), FALLBACK: reply()})
        await llm.llm_call(MESSAGES, TOOLS)
        assert llm.CIRCUIT_SECONDS == pytest.approx(60, abs=1)
        assert llm._down_until[PRIMARY] > time.monotonic() + 30

    async def test_a_model_is_tried_again_once_its_window_passes(self, wire):
        client = wire({PRIMARY: status_error(openai.RateLimitError, 429),
                       FALLBACK: reply()})
        await llm.llm_call(MESSAGES, TOOLS)
        llm._down_until[PRIMARY] = time.monotonic() - 1
        await llm.llm_call(MESSAGES, TOOLS)
        assert client.completions.models.count(PRIMARY) == 2

    async def test_a_healthy_chain_opens_no_breaker(self, wire):
        wire({PRIMARY: reply()})
        await llm.llm_call(MESSAGES, TOOLS)
        assert llm._down_until == {}

    async def test_the_last_model_is_tried_even_with_every_breaker_open(self, wire):
        """Refusing to call anything is worse than one wasted attempt."""
        client = wire({PRIMARY: timeout(), FALLBACK: timeout(), LAST_RESORT: timeout()})
        with pytest.raises(llm.LLMUnavailable):
            await llm.llm_call(MESSAGES, TOOLS)
        client.completions.outcomes[LAST_RESORT] = reply()
        _, model_used, _ = await llm.llm_call(MESSAGES, TOOLS)
        assert model_used == LAST_RESORT


class TestClientConfiguration:
    def test_it_points_at_openrouter(self, monkeypatch):
        monkeypatch.setattr(llm, "_client", None)
        monkeypatch.setattr(config, "OPENROUTER_API_KEY", "sk-test")
        assert "openrouter.ai" in str(llm.client().base_url)

    def test_the_per_call_timeout_comes_from_config(self, monkeypatch):
        monkeypatch.setattr(llm, "_client", None)
        monkeypatch.setattr(config, "OPENROUTER_API_KEY", "sk-test")
        monkeypatch.setattr(config, "LLM_TIMEOUT_S", 20)
        assert llm.client().timeout == 20
