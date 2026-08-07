"""Phase 5 §1 — the three endpoints, and the three §13 lines that cannot be live.

Phase 5 §3 names three definition-of-done lines that a live model can never
prove, for structural reasons:

| DoD line                                | Why it cannot be live                    |
|-----------------------------------------|------------------------------------------|
| Duplicate `wa_message_id` dropped       | rejected before the loop starts          |
| Escalated conversation gets no reply    | the status guard returns before the LLM  |
| Guard catches an injected fake date     | you cannot make a model hallucinate on cue |

All three are here, alongside the endpoint contracts themselves. The model is
scripted throughout — what is under test is the HTTP layer's behaviour, and a
real model would only add cost and variance to assertions that never touch it.
"""
import asyncio

import httpx
import pytest
from sqlmodel import Session, create_engine, select

from app import agent, config, db, guard, llm, main
from app.db import Conversation, GuardRejection, Message
from tests.test_agent import FakeLLM, calls, says

pytestmark = pytest.mark.usefixtures("seeded")

BRIEF = says(
    'WANTS 3BHK in Bopal\n'
    'WHY NOW Asked to negotiate the price.\n'
    'OPEN "Namaste, main AllSet se bol raha hoon."'
)


@pytest.fixture(autouse=True)
def clean_state():
    agent.reset_state()
    yield
    agent.reset_state()


@pytest.fixture
def fast(monkeypatch):
    """Debounce off — the scripted-demo setting (D10)."""
    monkeypatch.setattr(config, "DEBOUNCE_MS", 0)
    monkeypatch.setattr(config, "DEBOUNCE_MAX_MS", 10_000)


@pytest.fixture
def model(monkeypatch):
    def install(*script, **kwargs):
        fake = FakeLLM(script, **kwargs)
        monkeypatch.setattr(llm, "llm_call", fake)
        return fake
    return install


@pytest.fixture
async def client():
    """The app over an in-process ASGI transport.

    No socket, no port, and the same event loop as the test — which is what makes
    the coalescing case (D8) expressible at all: three concurrent POSTs have to
    await the same `asyncio.Event` the debounce timer will set.
    """
    transport = httpx.ASGITransport(app=main.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://api") as c:
        yield c


async def send(client, conversation_id, text, wa_message_id=None):
    response = await client.post(
        f"/conversations/{conversation_id}/messages",
        json={"text": text, "wa_message_id": wa_message_id},
    )
    return response


async def escalate_conversation(client, model, conversation_id=2):
    """Drive one conversation to `escalated` down the deterministic path (D18).

    The trigger needs no model, but the brief writer does — that call is the only
    LLM request on this path (D17).
    """
    model(BRIEF)
    response = await send(client, conversation_id, "price kam karo bhai", "wa-esc")
    assert response.json()["status"] == "escalated"
    return response


# --------------------------------------------------------------------------
# POST /conversations/{id}/messages
# --------------------------------------------------------------------------

class TestPostMessage:
    async def test_a_normal_turn_returns_reply_tool_calls_and_status(
        self, client, model, fast
    ):
        model(
            calls(("update_buyer_profile", {"buyer_id": 1, "updates": {"bhk_need": 3}})),
            says("Got it — what budget are you working with?"),
        )
        body = (await send(client, 1, "3BHK chahiye", "wa-1")).json()

        assert body["status"] == "active"
        assert body["reply"] == "Got it — what budget are you working with?"
        assert [c["tool_name"] for c in body["tool_calls"]] == ["update_buyer_profile"]

    async def test_tool_calls_carry_args_result_and_latency(self, client, model, fast):
        model(
            calls(("search_properties", {"bhk": 3, "localities": ["Bopal"]})),
            says("Here are a few."),
        )
        body = (await send(client, 1, "3BHK in Bopal", "wa-2")).json()

        call = body["tool_calls"][0]
        assert call["args"] == {"bhk": 3, "localities": ["Bopal"]}
        assert call["result"]["exact_matches"]
        assert call["latency_ms"] >= 0

    async def test_tool_calls_is_always_present_even_when_empty(
        self, client, model, fast
    ):
        model(says("Hello!"))
        body = (await send(client, 1, "hi", "wa-3")).json()
        assert body["tool_calls"] == []

    async def test_an_unknown_conversation_is_a_404_not_a_turn(self, client, model, fast):
        model(says("should never be called"))
        response = await send(client, 99, "hello", "wa-4")

        assert response.status_code == 404
        assert "99" in response.json()["detail"]

    async def test_blank_text_is_rejected_before_anything_is_buffered(
        self, client, model, fast
    ):
        model(says("should never be called"))
        assert (await send(client, 1, "   ", "wa-5")).status_code == 422
        assert (await client.post(
            "/conversations/1/messages", json={"text": "", "wa_message_id": "wa-6"}
        )).status_code == 422

    async def test_wa_message_id_is_optional(self, client, model, fast):
        model(says("Hello!"))
        response = await client.post("/conversations/1/messages", json={"text": "hi"})
        assert response.status_code == 200
        assert response.json()["reply"] == "Hello!"


class TestCoalescing:
    """D8 — three rapid messages, one turn, one reply.

    The live version of this line lives in `test_loop_live.py`; this one is the
    free regression that runs on every save.
    """

    async def test_three_rapid_messages_produce_one_reply(
        self, client, model, monkeypatch
    ):
        monkeypatch.setattr(config, "DEBOUNCE_MS", 200)
        monkeypatch.setattr(config, "DEBOUNCE_MAX_MS", 2_000)
        fake = model(says("Bopal me 3BHK — budget kitna hai?"))

        bodies = [
            response.json() for response in await asyncio.gather(
                send(client, 1, "Hi", "wa-a"),
                send(client, 1, "3bhk chahiye", "wa-b"),
                send(client, 1, "bopal me", "wa-c"),
            )
        ]

        replied = [b for b in bodies if b["status"] != "coalesced"]
        coalesced = [b for b in bodies if b["status"] == "coalesced"]
        assert len(replied) == 1 and len(coalesced) == 2
        assert all(b["reply"] is None for b in coalesced)
        assert fake.call_count == 1

    async def test_the_three_fragments_reach_the_model_as_one_message(
        self, client, model, monkeypatch, session
    ):
        monkeypatch.setattr(config, "DEBOUNCE_MS", 200)
        monkeypatch.setattr(config, "DEBOUNCE_MAX_MS", 2_000)
        model(says("Samajh gaya."))

        await asyncio.gather(
            send(client, 1, "Hi", "wa-d"),
            send(client, 1, "3bhk chahiye", "wa-e"),
            send(client, 1, "bopal me", "wa-f"),
        )

        stored = session.exec(
            select(Message).where(Message.conversation_id == 1)
            .where(Message.role == "user").order_by(Message.id.desc())
        ).first()
        assert stored.content == "Hi\n3bhk chahiye\nbopal me"


class TestDuplicateMessageId:
    """§13 — a duplicate `wa_message_id` is dropped silently.

    Structurally offline: the id is rejected before a turn is ever started, so
    there is no model call to make live.
    """

    async def test_the_second_send_is_dropped(self, client, model, fast):
        model(says("Hello!"))
        first = (await send(client, 1, "hi", "wa-dup")).json()
        second = (await send(client, 1, "hi", "wa-dup")).json()

        assert first["status"] == "active"
        assert second == {"reply": None, "tool_calls": [], "status": "duplicate"}

    async def test_dropped_silently_means_200_not_an_error(self, client, model, fast):
        model(says("Hello!"))
        await send(client, 1, "hi", "wa-dup2")
        assert (await send(client, 1, "hi", "wa-dup2")).status_code == 200

    async def test_no_second_row_and_no_second_turn(self, client, model, fast, session):
        fake = model(says("Hello!"))
        await send(client, 1, "hi", "wa-dup3")
        await send(client, 1, "hi", "wa-dup3")

        rows = session.exec(
            select(Message).where(Message.wa_message_id == "wa-dup3")
        ).all()
        assert len(rows) == 1
        assert fake.call_count == 1

    async def test_the_stored_row_catches_it_after_the_buffer_is_gone(
        self, client, model, fast
    ):
        """The in-memory `SEEN` set covers ids still sitting in a debounce
        window; the UNIQUE column is what survives a restart. Clearing the
        process state is how you test the second half."""
        model(says("Hello!"))
        await send(client, 1, "hi", "wa-dup4")
        agent.reset_state()

        assert (await send(client, 1, "hi", "wa-dup4")).json()["status"] == "duplicate"


class TestEscalatedConversationIsSilent:
    """§13 — a message to an escalated conversation gets no reply and shows up
    in `/broker/inbox`. Offline for the same structural reason: the status guard
    returns before the model is reached."""

    async def test_the_agent_says_nothing(self, client, model, fast):
        await escalate_conversation(client, model)
        body = (await send(client, 2, "acha theek hai, kab call karoge?", "wa-p1")).json()

        assert body["reply"] is None
        assert body["status"] == "escalated"

    async def test_the_model_is_never_called_again(self, client, model, fast):
        await escalate_conversation(client, model)
        fake = model(says("this must never be sent"))

        await send(client, 2, "hello?", "wa-p2")
        assert fake.call_count == 0

    async def test_the_message_is_stored_for_the_broker(self, client, model, fast, session):
        await escalate_conversation(client, model)
        await send(client, 2, "kab call karoge?", "wa-p3")

        rows = session.exec(
            select(Message).where(Message.wa_message_id == "wa-p3")
        ).all()
        assert len(rows) == 1
        assert rows[0].role == "user"

    async def test_it_appears_in_the_broker_inbox(self, client, model, fast):
        await escalate_conversation(client, model)
        await send(client, 2, "kab call karoge?", "wa-p4")

        lead = (await client.get("/broker/inbox")).json()["leads"][0]
        assert lead["conversation_id"] == 2
        assert [m["content"] for m in lead["unread_messages"]] == ["kab call karoge?"]
        assert lead["unread_count"] == 1

    async def test_the_handoff_line_itself_is_not_unread(self, client, model, fast):
        """The buyer hears exactly one line, then silence (D19). That line is an
        assistant row and must never be counted as something awaiting a reply."""
        await escalate_conversation(client, model)

        lead = (await client.get("/broker/inbox")).json()["leads"][0]
        assert lead["unread_messages"] == []


class TestGuardRejectionIsVisible:
    """§13 — the guard catches a deliberately injected fake date and the rejected
    output is logged. `FORCE_GUARD_FAIL=1` is the injection: you cannot make a
    model hallucinate on cue in front of an audience."""

    @pytest.fixture
    def forced(self, monkeypatch):
        monkeypatch.setattr(config, "FORCE_GUARD_FAIL", True)

    async def test_the_reply_never_carries_the_injected_facts(
        self, client, model, fast, forced
    ):
        model(says("Bopal me options hain."))
        body = (await send(client, 1, "Bopal me kya hai?", "wa-g1")).json()

        assert body["reply"] == guard.CLARIFYING_QUESTION
        assert "March 2027" not in body["reply"]
        assert "₹72 lakh" not in body["reply"]

    async def test_the_rejected_output_is_logged_in_full(
        self, client, model, fast, forced, session
    ):
        model(says("Bopal me options hain."))
        await send(client, 1, "Bopal me kya hai?", "wa-g2")

        rows = session.exec(select(GuardRejection).order_by(GuardRejection.id)).all()
        assert rows
        assert guard.FORCED_SUFFIX.strip() in rows[0].rejected_text
        assert "March 2027" in rows[0].offending_spans

    async def test_the_trace_endpoint_shows_it(self, client, model, fast, forced):
        model(says("Bopal me options hain."))
        await send(client, 1, "Bopal me kya hai?", "wa-g3")

        body = (await client.get("/conversations/1")).json()
        assert body["guard_rejections"]
        spans = body["guard_rejections"][0]["offending_spans"]
        assert "March 2027" in spans and "₹72 lakh" in spans

    async def test_a_clean_turn_logs_nothing(self, client, model, fast):
        model(says("Bopal me options hain."))
        await send(client, 1, "Bopal me kya hai?", "wa-g4")

        body = (await client.get("/conversations/1")).json()
        assert body["guard_rejections"] == []


# --------------------------------------------------------------------------
# GET /conversations/{id}
# --------------------------------------------------------------------------

class TestGetConversation:
    async def test_an_unknown_conversation_is_a_404(self, client):
        assert (await client.get("/conversations/99")).status_code == 404

    async def test_the_seeded_opening_message_is_there(self, client):
        body = (await client.get("/conversations/1")).json()
        assert body["conversation_id"] == 1
        assert body["status"] == "active"
        assert body["messages"][0]["content"] == "Hi, looking for a 3BHK in Bopal"

    async def test_the_transcript_is_in_order(self, client, model, fast):
        model(says("Kaisi help chahiye?"))
        await send(client, 1, "hello", "wa-t1")

        roles = [m["role"] for m in (await client.get("/conversations/1")).json()["messages"]]
        assert roles == ["user", "user", "assistant"]

    async def test_tool_calls_are_joined_onto_the_turn_that_made_them(
        self, client, model, fast
    ):
        model(
            calls(("search_properties", {"bhk": 3, "localities": ["Bopal"]})),
            says("Here are a few."),
        )
        await send(client, 1, "3BHK in Bopal", "wa-t2")

        messages = (await client.get("/conversations/1")).json()["messages"]
        traced = [m for m in messages if m["tool_calls"]]
        assert len(traced) == 1
        assert traced[0]["tool_calls"][0]["tool_name"] == "search_properties"

    async def test_assistant_rows_carry_the_telemetry(self, client, model, fast):
        model(says("Hello!"), model="google/gemini-3.1-flash-lite", provider="Google AI Studio")
        await send(client, 1, "hi", "wa-t3")

        assistant = [
            m for m in (await client.get("/conversations/1")).json()["messages"]
            if m["role"] == "assistant"
        ][-1]
        assert assistant["model_used"] == "google/gemini-3.1-flash-lite"
        assert assistant["provider"] == "Google AI Studio"
        assert assistant["latency_ms"] >= 0
        assert assistant["inbound_latency_ms"] >= 0

    async def test_user_rows_carry_no_telemetry(self, client):
        user_row = (await client.get("/conversations/1")).json()["messages"][0]
        assert "model_used" not in user_row
        assert "latency_ms" not in user_row

    async def test_the_buyer_profile_snapshot_is_included(self, client, model, fast):
        model(
            calls(("update_buyer_profile",
                   {"buyer_id": 1, "updates": {"budget_max": 9_000_000}})),
            says("Noted."),
        )
        await send(client, 1, "90 lakh tak", "wa-t4")

        buyer = (await client.get("/conversations/1")).json()["buyer"]
        assert buyer["name"] == "Priya"
        assert buyer["profile"]["budget"] == "up to ₹90 lakh"
        assert "preferred_localities" in buyer["unknown"]

    async def test_the_escalation_and_its_brief_are_included(self, client, model, fast):
        await escalate_conversation(client, model)

        body = (await client.get("/conversations/2")).json()
        assert body["status"] == "escalated"
        assert body["escalation"]["reason"] == "negotiation_request"
        assert body["escalation"]["urgency"] == "high"
        assert "Rakesh" in body["escalation"]["brief"]

    async def test_there_is_no_escalation_until_there_is_one(self, client):
        assert (await client.get("/conversations/1")).json()["escalation"] is None


# --------------------------------------------------------------------------
# GET /broker/inbox
# --------------------------------------------------------------------------

class TestBrokerInbox:
    async def test_it_is_empty_before_anything_escalates(self, client):
        assert (await client.get("/broker/inbox")).json() == {"count": 0, "leads": []}

    async def test_a_lead_carries_everything_the_broker_needs(self, client, model, fast):
        await escalate_conversation(client, model)

        lead = (await client.get("/broker/inbox")).json()["leads"][0]
        assert lead["buyer"] == {
            "id": 2, "name": "Rakesh", "phone": "+919825000002", "intent_tier": "cold",
        }
        assert lead["reason"] == "negotiation_request"
        assert lead["urgency"] == "high"
        assert lead["conversation_id"] == 2
        assert lead["escalated_at"]

    async def test_the_brief_is_readable_without_opening_the_transcript(
        self, client, model, fast
    ):
        await escalate_conversation(client, model)

        brief = (await client.get("/broker/inbox")).json()["leads"][0]["brief"]
        for label in ("WANTS", "WHY NOW", "OPEN", "KNOWN", "UNKNOWN", "TRIGGER"):
            assert label in brief
        assert "+919825000002" in brief

    async def test_leads_come_back_newest_first(self, client, model, fast):
        await escalate_conversation(client, model, conversation_id=2)
        agent.reset_state()
        model(BRIEF)
        await send(client, 3, "koi discount milega?", "wa-esc3")

        leads = (await client.get("/broker/inbox")).json()["leads"]
        assert [lead["conversation_id"] for lead in leads] == [3, 2]

    async def test_one_conversation_never_yields_two_leads(self, client, model, fast):
        await escalate_conversation(client, model)
        await send(client, 2, "price kam karo na bhai", "wa-esc-again")

        assert (await client.get("/broker/inbox")).json()["count"] == 1


# --------------------------------------------------------------------------
# Startup
# --------------------------------------------------------------------------

class TestStartup:
    """D9 and D24 both fail loudly at startup or not at all. `assert_ist` has its
    own tests in `test_config.py`; what matters here is that the app wires both
    of them into the lifespan rather than trusting the operator."""

    async def test_the_lifespan_asserts_both_contracts(self, monkeypatch):
        called = []
        monkeypatch.setattr(config, "assert_ist", lambda: called.append("ist"))
        monkeypatch.setattr(
            config, "assert_single_worker", lambda: called.append("worker")
        )
        async with main.lifespan(main.app):
            pass
        assert called == ["ist", "worker"]

    async def test_a_second_worker_stops_the_server_starting(self, monkeypatch):
        monkeypatch.setenv("WEB_CONCURRENCY", "4")
        with pytest.raises(RuntimeError, match="single-worker"):
            async with main.lifespan(main.app):
                pass

    async def test_startup_creates_the_tables(self, tmp_path, monkeypatch):
        """`create_all()` runs against whatever `db.engine` points at, so a
        server started on a database that does not exist yet comes up with an
        empty schema rather than a 500 on the first request."""
        fresh = create_engine(
            "sqlite:///" + (tmp_path / "fresh.db").as_posix(),
            connect_args={"check_same_thread": False},
        )
        monkeypatch.setattr(db, "engine", fresh)

        async with main.lifespan(main.app):
            pass

        with Session(fresh) as session:
            assert session.exec(select(Conversation)).all() == []
        fresh.dispose()
