"""Phase 5 §4 — the §13 definition-of-done lines that run live, end to end.

Six of the nine §13 lines run against a real model here. The other three are in
`test_api.py`, because a live model structurally cannot prove them: the duplicate
id is rejected before the loop starts, the escalated-conversation guard returns
before the model is reached, and you cannot make a model hallucinate on cue.

Beyond §13, this file also covers the five lines the design review added: the
labelled near-matches (D14), the qualified handoff, the brief a broker could act
on unread (D17), one handoff line then silence (D19), and the deterministic tier
firing without a model at all (D18).

Opt-in: `pytest --live`. Run it serially — see `test_loop_live.py` on quota.
"""
import httpx
import pytest
from sqlmodel import select

from app import agent, config, escalation, llm, main, triggers
from app.db import Conversation, Escalation, Message

pytestmark = [pytest.mark.live, pytest.mark.usefixtures("seeded")]


@pytest.fixture(autouse=True)
def clean_state():
    agent.reset_state()
    llm.reset_circuit()
    yield
    agent.reset_state()
    llm.reset_circuit()


@pytest.fixture(autouse=True)
def fast(monkeypatch):
    """Every test in this file runs at the scripted-demo setting (D10)."""
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
    response = await client.post(
        f"/conversations/{conversation_id}/messages",
        json={"text": text, "wa_message_id": wa_message_id},
    )
    assert response.status_code == 200, response.text
    return response.json()


def tool_names(body):
    return [call["tool_name"] for call in body["tool_calls"]]


def results_for(body, name):
    return [c["result"] for c in body["tool_calls"] if c["tool_name"] == name]


# --------------------------------------------------------------------------
# §13 — the scripted conversation
# --------------------------------------------------------------------------

class TestScriptedConversation:
    """§13 lines 1 and 2 — the scripted conversation runs end to end, and
    `tool_calls` is populated for every turn.

    **Five turns, and no budget turn.** Phase 5 §2's table has six, turn 2 being
    "80 to 90 lakh". That cannot run: D16 hands a lead over as soon as it has
    budget AND locality AND (BHK OR possession), and Priya's opening already
    states two of the three — so naming a budget escalates her on
    `qualification_complete` and turns 3 to 6 never happen. Withholding it keeps
    her unqualified, which is what lets the arc reach the booking and hand off on
    `site_visit_booked` instead: the strongest trigger, first in PRECEDENCE, and
    the one §2 wanted turn 6 to fire. Every capability in the table is still
    exercised.

    One test, five turns, because they are one conversation — turn 4's "it"
    refers to turn 3's property. Splitting them would re-run the earlier turns
    for each, at three times the quota for no extra coverage.
    """

    async def test_the_scripted_conversation_runs_end_to_end(self, client):
        traces = []

        # 1. Opening. Two facts stated, so both should be recorded and the first
        #    unknown asked about.
        turn = await send(client, 1, "Hi, looking for a 3BHK in Bopal", "dod-1")
        assert turn["status"] == "active"
        assert turn["reply"]
        assert "update_buyer_profile" in tool_names(turn)
        traces.append(turn)

        # 2. Search on partial qualification — every search argument is optional
        #    precisely so this turn is possible.
        turn = await send(
            client, 1,
            "I'll share my budget in a bit — can you show me what you have first?",
            "dod-2",
        )
        assert turn["status"] == "active"
        assert "search_properties" in tool_names(turn)
        cards = results_for(turn, "search_properties")[0]["exact_matches"]
        assert 1 <= len(cards) <= 5
        for card in cards:
            # D4 — every buyer-facing value arrives pre-formatted, which is what
            # lets the guard be a byte-exact substring match.
            assert card["price_display"].startswith("₹")
            assert card["possession_display"]
        traces.append(turn)

        titles = {card["id"]: card["title"] for card in cards}
        target_id = list(titles)[min(2, len(titles) - 1)]
        target = titles[target_id]

        # 3. Possession on a named property. The id has to come from the SHOWN
        #    block, because the transcript replay (D11) threw the cards away.
        turn = await send(
            client, 1, f"Tell me about {target} — when is possession?", "dod-3"
        )
        assert turn["status"] == "active"
        details = [c for c in turn["tool_calls"] if c["tool_name"] == "get_property_details"]
        assert details, tool_names(turn)
        assert "possession" in details[0]["args"].get("sections", [])
        assert details[0]["args"]["property_id"] == target_id, (
            f"asked about {target} (#{target_id}) and the agent looked up "
            f"#{details[0]['args']['property_id']} — a wrong-property answer that "
            f"the guard cannot catch, because the facts really did come from a tool"
        )
        traces.append(turn)

        # 4. A factual legal question. Not an escalation — this is what
        #    get_property_details(["legal"]) exists for.
        turn = await send(client, 1, "Is it RERA registered?", "dod-4")
        assert turn["status"] == "active", "a factual question must not escalate"
        legal = [
            call["result"].get("legal")
            for call in turn["tool_calls"]
            if call["tool_name"] == "get_property_details"
        ]
        assert any(section and section.get("rera_id") for section in legal), turn
        rera = [s["rera_id"] for s in legal if s][0]
        assert "DEMO" in rera, "every seeded RERA id carries a visible DEMO segment (D22)"
        assert rera in (turn["reply"] or ""), "the RERA id must be quoted verbatim"
        traces.append(turn)

        # 5. The booking, and the handoff the evaluator fires on it.
        turn = await send(client, 1, "Can I visit this Saturday at 5pm?", "dod-5")
        booked = results_for(turn, "book_site_visit")
        assert booked and booked[0].get("confirmed"), turn
        assert turn["status"] == "escalated"
        # D15/D19 — the confirmation the buyer needed survives the handoff.
        assert booked[0]["slot_display"] in turn["reply"]
        assert (
            escalation.HANDOFF_EN in turn["reply"]
            or escalation.HANDOFF_HI in turn["reply"]
        )
        traces.append(turn)

        # §13 line 2 — every turn traced something.
        assert all(t["tool_calls"] for t in traces), [tool_names(t) for t in traces]

        detail = (await client.get("/conversations/1")).json()
        assert detail["status"] == "escalated"
        assert detail["escalation"]["reason"] == "site_visit_booked"
        assert detail["escalation"]["urgency"] == "high"


# --------------------------------------------------------------------------
# §13 — the deterministic tier
# --------------------------------------------------------------------------

class TestDeterministicEscalation:
    """§13 — "price kam karo" escalates; "loan lunga, budget 65L" does not.

    The negative half is the expensive one to get wrong: a false positive
    permanently silences the agent for that buyer, so the lead is handed to a
    broker who was never going to be called.
    """

    async def test_price_kam_karo_escalates_without_a_search(self, client):
        body = await send(client, 2, "price kam karo bhai", "dod-neg-1")

        assert body["status"] == "escalated"
        assert "search_properties" not in tool_names(body)
        # The decision is made without a model (D18); the only LLM request on
        # this path is the brief writer.
        assert triggers.match("price kam karo bhai") == "negotiation_request"

    async def test_loan_lunga_is_an_ordinary_qualification_turn(self, client):
        assert triggers.match("loan lunga, budget 65L") is None

        body = await send(client, 2, "loan lunga, budget 65L", "dod-neg-2")
        assert body["status"] == "active"
        assert body["reply"]
        assert "update_buyer_profile" in tool_names(body)


# --------------------------------------------------------------------------
# §13 — the handoff and the brief
# --------------------------------------------------------------------------

class TestHandoffAndBrief:
    async def test_the_buyer_hears_exactly_one_handoff_line_then_silence(self, client):
        """D19 — one templated line, then permanent silence."""
        escalating = await send(client, 2, "price kam karo bhai", "dod-h1")
        assert escalating["status"] == "escalated"
        assert escalating["reply"]
        assert (
            escalation.HANDOFF_HI in escalating["reply"]
            or escalation.HANDOFF_EN in escalating["reply"]
        )

        silent = await send(client, 2, "haan theek hai, kab call aayega?", "dod-h2")
        assert silent["reply"] is None
        assert silent["status"] == "escalated"

    async def test_the_handoff_matches_the_buyer_s_language(self, client):
        body = await send(client, 2, "bhai price kam karo na, kuch to karo", "dod-h3")
        assert escalation.HANDOFF_HI in body["reply"]

    async def test_the_brief_is_actionable_unread(self, client, session):
        """§13 — "a brief a broker could act on unread". Structure is what a test
        can assert; whether it reads well is a human's call, and the manual
        script (docs/manual-test-plan.md) is where that check lives."""
        await send(client, 2, "3bhk chahiye bhai, budget 65 lakh, Bopal me", "dod-b1")
        await send(client, 2, "price kam karo bhai", "dod-b2")

        row = session.exec(select(Escalation)).first()
        assert row is not None
        brief = row.brief

        for label in ("WANTS", "WHY NOW", "OPEN", "KNOWN", "UNKNOWN", "TRIGGER"):
            assert label in brief, brief
        # Who to call, and what they already said.
        assert "Rakesh" in brief
        assert "+919825000002" in brief
        assert "LAST 3" in brief
        # And nothing the broker has to go and look up.
        assert row.reason and row.urgency in ("low", "medium", "high")

    async def test_one_conversation_yields_one_lead(self, client):
        await send(client, 2, "price kam karo bhai", "dod-b3")
        await send(client, 2, "koi discount milega?", "dod-b4")

        inbox = (await client.get("/broker/inbox")).json()
        assert inbox["count"] == 1
        assert inbox["leads"][0]["unread_messages"], "the second message is unanswered"


# --------------------------------------------------------------------------
# Beyond §13 — the design-review additions
# --------------------------------------------------------------------------

class TestZeroResultQuery:
    """D14 — a zero-result query returns labelled near-matches with the
    relaxation named. Anjali's opening is unsatisfiable against the seeded
    bands on purpose: Satellite 3BHKs start at ₹1.2 crore."""

    async def test_near_matches_come_back_labelled(self, client):
        body = await send(client, 3, "3BHK in Satellite under 80 lakh", "dod-z1")

        searches = results_for(body, "search_properties")
        assert searches, tool_names(body)
        result = searches[0]
        assert result["exact_matches"] == []
        assert result["near_matches"], result
        assert result["relaxed"], "the relaxation was not named"
        assert result["note"], "the gap was not explained"

    async def test_the_agent_still_replies_with_something_useful(self, client):
        body = await send(client, 3, "3BHK in Satellite under 80 lakh", "dod-z2")
        assert body["reply"]
        assert body["status"] in ("active", "escalated")


class TestQualificationEscalation:
    """D16 — the bar is budget **and** locality **and** (BHK **or** possession).
    Anjali states all three in one message, so the evaluator should hand her off
    on the turn that surfaces the near-matches."""

    async def test_a_fully_qualified_opening_hands_off(self, client, session):
        await send(client, 3, "3BHK in Satellite under 80 lakh", "dod-q1")

        convo = session.get(Conversation, 3)
        row = session.exec(
            select(Escalation).where(Escalation.conversation_id == 3)
        ).first()
        if convo.status == "escalated":
            assert row.reason in dict(escalation.PRECEDENCE)
            assert row.brief
        else:
            # The model did not record all three fields in one turn. That is a
            # prompt-adherence miss, not a broken evaluator — say which.
            pytest.fail(
                "qualification_complete did not fire on a message stating "
                "budget, locality and BHK; check what update_buyer_profile was "
                "actually called with."
            )


class TestNoDuplicateEscalationRows:
    async def test_both_authorities_firing_in_one_turn_write_one_row(
        self, client, session
    ):
        """The model's own `escalate_to_broker` and the post-turn evaluator can
        both fire on the same turn. The broker must not inherit the lead twice."""
        await send(client, 2, "3bhk chahiye bhai, budget 65 lakh, Bopal me", "dod-d1")
        await send(client, 2, "koi discount milega bhai?", "dod-d2")

        rows = session.exec(
            select(Escalation).where(Escalation.conversation_id == 2)
        ).all()
        assert len(rows) == 1

        assistant = session.exec(
            select(Message).where(Message.conversation_id == 2)
            .where(Message.role == "assistant")
        ).all()
        handoffs = [
            m for m in assistant
            if escalation.HANDOFF_HI in m.content or escalation.HANDOFF_EN in m.content
        ]
        assert len(handoffs) == 1, "the buyer heard the handoff line twice"
