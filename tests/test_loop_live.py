"""Phase 5 §3 — the loop against a real model: coalescing, trace, failover, latency.

Every assertion here runs through real Gemini over OpenRouter, so the contract is
genuinely verified rather than mocked. That is also why the file is opt-in:
`pytest --live`.

**The quota is the binding constraint.** BYOK gives 15 requests/minute and
500/day across rungs 1 and 2. One turn costs two or three requests; an escalating
turn costs one more for the brief; a guard rejection doubles its turn. So:

- Run this file **serially**. Never `pytest -n`. Parallel live tests are the
  fastest way to burn the day's budget on a suite you then have to re-run.
- `test_model_used_is_recorded` is the canary. Failover is silent by design, so
  if a 429 has quietly moved the whole suite onto rung 3, that assertion is the
  only thing that will say so before the demo does.

**Assert structurally, never on prose.** `"search_properties" in tool_names`,
`status == "escalated"`, `reply is None`, `elapsed < 8`. The moment a test
asserts the model's wording it fails for reasons that are not bugs, and you start
weakening it until it proves nothing.
"""
import asyncio
import time

import httpx
import openai
import pytest
from sqlmodel import select

from app import agent, config, db, llm, main
from app.db import Message, ToolCall

pytestmark = [pytest.mark.live, pytest.mark.usefixtures("seeded")]


@pytest.fixture(autouse=True)
def clean_state():
    agent.reset_state()
    llm.reset_circuit()
    yield
    agent.reset_state()
    llm.reset_circuit()


@pytest.fixture
def fast(monkeypatch):
    monkeypatch.setattr(config, "DEBOUNCE_MS", 0)
    monkeypatch.setattr(config, "DEBOUNCE_MAX_MS", 10_000)


@pytest.fixture
async def client():
    transport = httpx.ASGITransport(app=main.app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://api", timeout=120.0
    ) as c:
        yield c


async def send(client, conversation_id, text, wa_message_id=None):
    return await client.post(
        f"/conversations/{conversation_id}/messages",
        json={"text": text, "wa_message_id": wa_message_id},
    )


def tool_names(body):
    return [call["tool_name"] for call in body["tool_calls"]]


# --------------------------------------------------------------------------
# Coalescing (D8) — §13
# --------------------------------------------------------------------------

class TestCoalescing:
    """§13 — three rapid inbound messages coalesce into one turn, one reply.

    This is the line that most needs a real model behind it. Offline the fake
    proves the plumbing; live it also proves the model is handed one coherent
    message rather than three fragments it has to reconcile.
    """

    async def test_one_turn_one_reply_two_coalesced(self, client, monkeypatch):
        monkeypatch.setattr(config, "DEBOUNCE_MS", 400)
        monkeypatch.setattr(config, "DEBOUNCE_MAX_MS", 3_000)

        bodies = [
            response.json() for response in await asyncio.gather(
                send(client, 1, "Hi", "live-a"),
                send(client, 1, "3bhk chahiye", "live-b"),
                send(client, 1, "bopal me", "live-c"),
            )
        ]

        replied = [b for b in bodies if b["status"] != "coalesced"]
        coalesced = [b for b in bodies if b["status"] == "coalesced"]
        assert len(replied) == 1, bodies
        assert len(coalesced) == 2
        assert all(b["reply"] is None for b in coalesced)
        assert replied[0]["reply"]

    async def test_only_one_assistant_row_is_written(self, client, monkeypatch, session):
        monkeypatch.setattr(config, "DEBOUNCE_MS", 400)
        monkeypatch.setattr(config, "DEBOUNCE_MAX_MS", 3_000)

        await asyncio.gather(
            send(client, 1, "Hi", "live-d"),
            send(client, 1, "3bhk chahiye", "live-e"),
            send(client, 1, "bopal me", "live-f"),
        )

        assistant = session.exec(
            select(Message).where(Message.conversation_id == 1)
            .where(Message.role == "assistant")
        ).all()
        assert len(assistant) == 1


# --------------------------------------------------------------------------
# The trace (D3) — §13
# --------------------------------------------------------------------------

class TestTranscriptAndTrace:
    async def test_the_trace_endpoint_shows_the_turn_end_to_end(self, client, fast):
        await send(client, 1, "3BHK in Bopal, budget 80 to 90 lakh", "live-g")

        body = (await client.get("/conversations/1")).json()
        assistant = [m for m in body["messages"] if m["role"] == "assistant"]
        traced = [call for m in body["messages"] for call in m["tool_calls"]]

        assert assistant, "the turn produced no reply"
        assert traced, "the turn called no tools"
        for call in traced:
            assert call["tool_name"]
            assert isinstance(call["args"], dict)
            assert isinstance(call["result"], dict)
            assert call["latency_ms"] is not None

    async def test_model_used_is_recorded(self, client, fast):
        """The canary. Failover is silent (D3), so this is the only assertion
        that reveals a suite quietly running on rung 3 after a 429."""
        await send(client, 1, "Hi, looking for a 3BHK in Bopal", "live-h")

        body = (await client.get("/conversations/1")).json()
        assistant = [m for m in body["messages"] if m["role"] == "assistant"][-1]
        assert assistant["model_used"] in llm.model_chain()
        assert assistant["provider"]
        if assistant["model_used"] != config.MODEL_PRIMARY:
            pytest.fail(
                f"served by {assistant['model_used']}, not the primary "
                f"{config.MODEL_PRIMARY} — check the day's quota before demoing."
            )

    async def test_the_transcript_replays_text_only(self, client, fast, session):
        """D11 — assistant-with-tool_calls and role:"tool" messages live and die
        inside one turn. Only `user` and `assistant` rows are ever written."""
        await send(client, 1, "3BHK in Bopal under 90 lakh", "live-i")

        roles = {row.role for row in session.exec(select(Message)).all()}
        assert roles <= {"user", "assistant"}


# --------------------------------------------------------------------------
# Failover (D2, D3)
# --------------------------------------------------------------------------

class TestFailover:
    """D2/D3 — the chain, and the difference between a rung that is *down* and a
    request that is *wrong*.

    A nonexistent model id is a 400, and `_should_fail_over` deliberately does
    not retry those: a payload one model rejects, the next will reject too.
    Only 429s, 5xx and timeouts move down the chain. Testing failover with a
    made-up model id therefore proves nothing — it exercises the graceful-hold
    path (D20) while looking like it proved failover.
    """

    @pytest.fixture
    def primary_times_out(self, monkeypatch):
        """The primary rung times out; every other rung is real."""
        real = llm._create

        def flaky(model, messages, tools):
            if model == config.MODEL_PRIMARY:
                raise openai.APITimeoutError(
                    request=httpx.Request("POST", "https://openrouter.ai/api/v1")
                )
            return real(model, messages, tools)

        monkeypatch.setattr(llm, "_create", flaky)
        llm.reset_circuit()

    async def test_every_rung_of_the_chain_can_answer_a_real_turn(self, client, fast):
        """One request per rung, three in total. Pinned model ids move —
        `-preview` suffixes get dropped, `:free` models get retired — and a rung
        that stopped existing is invisible until the day you need it."""
        for index, model in enumerate(llm.model_chain()):
            response, used, provider = await llm.llm_call(
                [{"role": "user", "content": "Reply with the single word: ok"}]
            )
            assert used == model, f"rung {index} answered as {used}"
            assert response.choices[0].message.content
            # Force the next iteration onto the next rung.
            llm._down_until[model] = time.monotonic() + llm.CIRCUIT_SECONDS

    async def test_a_timing_out_primary_completes_on_the_next_rung(
        self, client, fast, primary_times_out
    ):
        body = (await send(client, 1, "Hi", "live-k")).json()

        assert body["status"] == "active"
        assert body["reply"]
        assert body["reply"] != agent.HOLD_MESSAGE, "this should have failed over"

    async def test_the_assistant_row_names_the_rung_that_actually_spoke(
        self, client, fast, primary_times_out
    ):
        """Failover is silent by design, so the stored `model_used` is the only
        record that the reply did not come from the rung you think it did."""
        await send(client, 1, "Hi", "live-k2")

        body = (await client.get("/conversations/1")).json()
        assistant = [m for m in body["messages"] if m["role"] == "assistant"][-1]
        assert assistant["model_used"] == config.MODEL_FALLBACK
        assert assistant["provider"]

    async def test_a_mid_turn_failover_re_runs_no_tools(
        self, client, fast, primary_times_out, session
    ):
        """D3 — the failover wraps the completion call, never the loop. A turn
        half-written by one model finishes on the next with no tool
        re-execution and no duplicate `tool_calls` rows."""
        body = (await send(client, 1, "3BHK in Bopal under 90 lakh", "live-j")).json()
        assert body["reply"]

        rows = session.exec(select(ToolCall).order_by(ToolCall.id)).all()
        signatures = [(row.tool_name, str(row.args)) for row in rows]
        assert len(signatures) == len(set(signatures)), "a tool ran twice"

    async def test_a_rejected_payload_is_not_retried_down_the_chain(
        self, client, fast, monkeypatch
    ):
        """A 400 is not transient. Retrying it would spend the whole chain's
        quota on a request none of them can serve, and then still hold."""
        monkeypatch.setattr(llm, "model_chain", lambda: ["does-not-exist/model-x"])
        llm.reset_circuit()

        body = (await send(client, 1, "Hi there", "live-400")).json()
        assert body["status"] == "active"
        assert body["reply"] == agent.HOLD_MESSAGE


# --------------------------------------------------------------------------
# Latency (D20) — §13
# --------------------------------------------------------------------------

class TestLatency:
    """§13 — a full turn in under 8 seconds, measured **drain → reply**.

    Debounce is deliberate waiting, not latency, so the clock starts when the
    buffer drains. `messages.latency_ms` records exactly that, which is why the
    stored figure is what the assertion reads rather than a wall clock around the
    request.
    """

    async def test_a_full_turn_finishes_under_eight_seconds(self, client, fast, session):
        """A two-step turn — one tool call, then the reply — on a warm process.

        The warm-up turn is not padding. The first OpenRouter request a process
        makes pays DNS, TCP and TLS on top of the completion, which measured at
        4-6 extra seconds here; measuring that would be measuring the connection,
        not the loop. Every subsequent turn in the same process is the one a
        buyer in a real conversation experiences.
        """
        await send(client, 1, "Hi", "live-warm")

        body = (await send(client, 1, "3BHK in Bopal, 80 to 90 lakh", "live-l")).json()

        assert body["status"] in ("active", "escalated")
        assistant = session.exec(
            select(Message).where(Message.role == "assistant")
            .order_by(Message.id.desc())
        ).first()
        assert assistant.latency_ms < 8_000, f"drain->reply {assistant.latency_ms}ms"

    async def test_debounce_is_excluded_from_the_measurement(
        self, client, monkeypatch, session
    ):
        """D20 — the clock starts at drain. Waiting for a buyer to finish typing
        is deliberate; counting it would make the target meaningless."""
        monkeypatch.setattr(config, "DEBOUNCE_MS", 1_500)
        monkeypatch.setattr(config, "DEBOUNCE_MAX_MS", 4_000)

        started = time.perf_counter()
        await send(client, 1, "Hi there", "live-deb")
        wall = time.perf_counter() - started

        assistant = session.exec(
            select(Message).where(Message.role == "assistant")
            .order_by(Message.id.desc())
        ).first()
        assert wall >= 1.5, "the debounce window did not happen"
        assert assistant.latency_ms < (wall - 1.5) * 1000 + 500

    async def test_an_unreachable_chain_holds_the_line_and_stays_active(
        self, client, fast, monkeypatch
    ):
        """D20 — infrastructure trouble is not a reason to hand a lead to a
        human. The conversation stays active and retries on the next inbound;
        only a step-cap breach escalates."""
        monkeypatch.setattr(llm, "model_chain", lambda: ["does-not-exist/model-x"])
        llm.reset_circuit()

        body = (await send(client, 1, "Hi there", "live-hold")).json()
        assert body["status"] == "active"
        assert body["reply"] == agent.HOLD_MESSAGE

        with db.get_session() as session:
            from app.db import Conversation
            assert session.get(Conversation, 1).status == "active"
