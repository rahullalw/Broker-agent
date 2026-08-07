"""Phase 4 wired into the loop — guard, triggers, evaluator, brief, handoff.

`tests/test_agent.py` proves the loop's own mechanics. This file proves the four
safety pieces are actually reached from it, on every path an escalation can take.
"""
from datetime import datetime, timedelta

import pytest
from sqlmodel import select

from app import agent, config, escalation, guard, llm
from app.db import Conversation, Escalation, GuardRejection, ToolCall
from tests.test_agent import FakeLLM, calls, says

pytestmark = pytest.mark.usefixtures("seeded")


@pytest.fixture(autouse=True)
def clean_state():
    agent.reset_state()
    yield
    agent.reset_state()


@pytest.fixture
def fast(monkeypatch):
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
def slot():
    return (datetime.now() + timedelta(days=3)).replace(
        hour=17, minute=0, second=0, microsecond=0
    ).isoformat(sep=" ")


def escalations(session):
    return session.exec(select(Escalation).order_by(Escalation.id)).all()


def rejections(session):
    return session.exec(select(GuardRejection).order_by(GuardRejection.id)).all()


def tool_rows(session):
    return session.exec(select(ToolCall).order_by(ToolCall.id)).all()


def is_brief_writer(sent):
    return "handover notes" in sent[0]["content"]


def booking(slot, key="k1"):
    return ("book_site_visit", {"buyer_id": 1, "property_id": 5,
                                "slot": slot, "idempotency_key": key})


# --------------------------------------------------------------------------
# The deterministic tier (D18)
# --------------------------------------------------------------------------

class TestDeterministicTrigger:
    async def test_price_kam_karo_escalates(self, fast, model, session):
        model(says("WANTS x\nWHY NOW y\nOPEN z"))
        result = await agent.on_inbound(2, "wamid.1", "price kam karo bhai")
        assert result["status"] == "escalated"
        assert escalations(session)[0].reason == "negotiation_request"

    async def test_the_decision_is_made_before_the_agent_ever_answers(
        self, fast, model, session
    ):
        fake = model(says("WANTS x\nWHY NOW y\nOPEN z"))
        await agent.on_inbound(2, "wamid.1", "price kam karo bhai")
        # The only model call made was the brief-writer's.
        assert fake.call_count == 1
        assert is_brief_writer(fake.seen[0])
        assert tool_rows(session) == []

    async def test_loan_lunga_is_answered_normally(self, fast, model, session):
        model(says("Theek hai, aur kaunsi area?"))
        result = await agent.on_inbound(2, "wamid.1", "loan lunga, budget 65L")
        assert result["status"] == "active"
        assert result["reply"] == "Theek hai, aur kaunsi area?"
        assert escalations(session) == []

    async def test_the_buyer_gets_the_hinglish_handoff_line(self, fast, model):
        model(says("WANTS x\nWHY NOW y\nOPEN z"))
        result = await agent.on_inbound(2, "wamid.1", "price kam karo bhai")
        assert result["reply"] == escalation.HANDOFF_HI

    async def test_an_english_buyer_gets_the_english_line(self, fast, model):
        model(says("WANTS x\nWHY NOW y\nOPEN z"))
        result = await agent.on_inbound(1, "wamid.1", "Can we negotiate the price?")
        assert result["reply"] == escalation.HANDOFF_EN


# --------------------------------------------------------------------------
# The guard (D4–D7)
# --------------------------------------------------------------------------

class TestGuardInTheLoop:
    async def test_a_reply_quoting_a_tool_string_is_delivered(self, fast, model):
        model(
            calls(("search_properties", {"bhk": 3, "localities": ["Bopal"]})),
            says("Vrund Meadows is ₹86 lakh, ready to move."),
        )
        result = await agent.on_inbound(1, "wamid.1", "3BHK in Bopal")
        assert result["reply"] == "Vrund Meadows is ₹86 lakh, ready to move."

    async def test_the_buyers_own_figure_passes_with_no_tools(self, fast, model, session):
        model(says("Samajh gaya — 65 lakh budget, Bopal, 3BHK. Possession kab tak?"))
        result = await agent.on_inbound(1, "wamid.1", "budget 65 lakh, Bopal me 3BHK chahiye")
        assert "65 lakh" in result["reply"]
        assert rejections(session) == []

    async def test_an_invented_price_never_reaches_the_buyer(self, fast, model, session):
        model(
            calls(("search_properties", {"bhk": 3, "localities": ["Bopal"]})),
            says("This one is ₹72 lakh."),
            says("Vrund Meadows is ₹86 lakh."),
        )
        result = await agent.on_inbound(1, "wamid.1", "3BHK in Bopal")
        assert "₹72 lakh" not in result["reply"]
        assert "₹72 lakh" in rejections(session)[0].rejected_text

    async def test_a_reformatted_reply_is_repaired_and_delivered(self, fast, model, session):
        model(
            calls(("search_properties", {"bhk": 3, "localities": ["Bopal"]})),
            says("Vrund Meadows is ₹86 L."),
            says("Vrund Meadows is ₹86 lakh."),
        )
        result = await agent.on_inbound(1, "wamid.1", "3BHK in Bopal")
        assert result["reply"] == "Vrund Meadows is ₹86 lakh."
        assert len(rejections(session)) == 1, "the repair worked, so only one rejection"

    async def test_the_repair_prompt_shows_the_model_both_sides(self, fast, model):
        fake = model(
            calls(("search_properties", {"bhk": 3, "localities": ["Bopal"]})),
            says("Vrund Meadows is ₹86 L."),
            says("Vrund Meadows is ₹86 lakh."),
        )
        await agent.on_inbound(1, "wamid.1", "3BHK in Bopal")
        repair = fake.seen[-1][-1]
        assert repair["role"] == "user"
        assert "₹86 L" in repair["content"]
        assert "₹86 lakh" in repair["content"]

    async def test_a_second_failure_falls_back_to_a_question(self, fast, model, session):
        model(
            calls(("search_properties", {"bhk": 3, "localities": ["Bopal"]})),
            says("This one is ₹72 lakh."),
            says("Sorry — it is ₹73 lakh."),
        )
        result = await agent.on_inbound(1, "wamid.1", "3BHK in Bopal")
        assert result["reply"] == guard.CLARIFYING_QUESTION
        assert len(rejections(session)) == 2

    async def test_it_never_retries_more_than_once_per_turn(self, fast, model):
        fake = model(
            calls(("search_properties", {"bhk": 3})),
            says("This one is ₹72 lakh."),
        )
        await agent.on_inbound(1, "wamid.1", "3BHK")
        # search step, first reply, one repair. No more.
        assert fake.call_count == 3

    async def test_a_derived_percentage_never_reaches_the_buyer(self, fast, model):
        model(
            calls(("search_properties", {"bhk": 3, "localities": ["Bopal"]})),
            says("I can do a 5% discount."),
            says("I can do a 5% discount."),
        )
        result = await agent.on_inbound(1, "wamid.1", "3BHK in Bopal")
        assert "5%" not in result["reply"]

    async def test_the_rejected_text_is_never_stored_as_a_message(
        self, fast, model, session
    ):
        from app.db import Message
        model(
            calls(("search_properties", {"bhk": 3})),
            says("This one is ₹72 lakh."),
            says("Sorry — it is ₹73 lakh."),
        )
        await agent.on_inbound(1, "wamid.1", "3BHK")
        stored = session.exec(select(Message).where(Message.role == "assistant")).all()
        assert all("₹72 lakh" not in m.content for m in stored)


class TestForceGuardFailEndToEnd:
    @pytest.fixture
    def forced(self, monkeypatch):
        monkeypatch.setattr(config, "FORCE_GUARD_FAIL", True)

    async def test_the_injected_fact_is_caught_and_logged_in_full(
        self, forced, fast, model, session
    ):
        model(says("Bopal me options hain."))
        result = await agent.on_inbound(1, "wamid.1", "3BHK in Bopal")
        row = rejections(session)[0]
        assert row.rejected_text == (
            "Bopal me options hain. Price is ₹72 lakh, possession March 2027."
        )
        assert "₹72 lakh" in row.offending_spans
        assert "March 2027" in row.offending_spans
        assert "₹72 lakh" not in result["reply"]

    async def test_the_buyer_gets_the_clarifying_question(self, forced, fast, model):
        model(says("Bopal me options hain."))
        result = await agent.on_inbound(1, "wamid.1", "3BHK in Bopal")
        assert result["reply"] == guard.CLARIFYING_QUESTION

    async def test_the_conversation_stays_active(self, forced, fast, model, session):
        model(says("Bopal me options hain."))
        result = await agent.on_inbound(1, "wamid.1", "3BHK in Bopal")
        assert result["status"] == "active"


# --------------------------------------------------------------------------
# The evaluator (D15, D16)
# --------------------------------------------------------------------------

class TestEvaluatorInTheLoop:
    async def test_a_booking_escalates_post_turn(self, fast, model, session, slot):
        model(calls(booking(slot)), says("Booked."))
        result = await agent.on_inbound(1, "wamid.1", "Can I visit on Saturday at 5pm?")
        assert result["status"] == "escalated"
        assert escalations(session)[0].reason == "site_visit_booked"
        assert escalations(session)[0].urgency == "high"

    async def test_the_buyer_keeps_the_confirmation_and_gets_the_handoff(
        self, fast, model, slot
    ):
        model(calls(booking(slot)), says("Booked."))
        result = await agent.on_inbound(1, "wamid.1", "Can I visit on Saturday at 5pm?")
        assert "Booked." in result["reply"]
        assert escalation.HANDOFF_EN in result["reply"]

    async def test_budget_and_locality_alone_do_not_escalate(self, fast, model, session):
        model(
            calls(("update_buyer_profile", {"buyer_id": 1, "updates": {
                "budget_max": 9_000_000, "preferred_localities": ["Bopal"]}})),
            says("Kitne bedroom chahiye?"),
        )
        result = await agent.on_inbound(1, "wamid.1", "Bopal, 90 lakh tak")
        assert result["status"] == "active"
        assert escalations(session) == []

    async def test_a_complete_qualification_escalates(self, fast, model, session):
        model(
            calls(("update_buyer_profile", {"buyer_id": 1, "updates": {
                "budget_max": 9_000_000, "preferred_localities": ["Bopal"],
                "bhk_need": 3}})),
            says("Samajh gaya."),
        )
        result = await agent.on_inbound(1, "wamid.1", "Bopal, 90 lakh, 3BHK")
        assert result["status"] == "escalated"
        assert escalations(session)[0].reason == "qualification_complete"

    async def test_both_authorities_in_one_turn_write_one_row(
        self, fast, model, session, slot
    ):
        model(
            calls(booking(slot),
                  ("escalate_to_broker", {"conversation_id": 1,
                                          "reason": "buyer wants a human",
                                          "urgency": "high"})),
            says("WANTS x\nWHY NOW y\nOPEN z"),
        )
        await agent.on_inbound(1, "wamid.1", "book it and get me a person")
        assert len(escalations(session)) == 1

    async def test_the_next_inbound_is_met_with_silence(self, fast, model, session, slot):
        model(calls(booking(slot)), says("Booked."))
        await agent.on_inbound(1, "wamid.1", "Can I visit on Saturday at 5pm?")
        again = await agent.on_inbound(1, "wamid.2", "kitne baje?")
        assert again["reply"] is None
        assert again["status"] == "escalated"


class TestClarificationCounting:
    async def test_a_question_that_captured_nothing_counts(self, fast, model, session):
        model(says("Budget kitna hai?"))
        await agent.on_inbound(1, "wamid.1", "hmm")
        session.expire_all()
        assert session.get(Conversation, 1).clarification_count == 1

    async def test_capturing_something_resets_it(self, fast, model, session):
        model(says("Budget kitna hai?"))
        await agent.on_inbound(1, "wamid.1", "hmm")
        model(
            calls(("update_buyer_profile", {"buyer_id": 1, "updates": {"bhk_need": 3}})),
            says("Kaunsi area?"),
        )
        await agent.on_inbound(1, "wamid.2", "3bhk")
        session.expire_all()
        assert session.get(Conversation, 1).clarification_count == 0

    async def test_three_in_a_row_escalate(self, fast, model, session):
        for index in range(3):
            model(says("Budget kitna hai?"))
            result = await agent.on_inbound(1, f"wamid.{index}", "hmm")
        assert result["status"] == "escalated"
        assert escalations(session)[0].reason == "clarification_exhausted"


# --------------------------------------------------------------------------
# The brief (D17)
# --------------------------------------------------------------------------

class TestEveryPathWritesABrief:
    async def test_the_regex_path(self, fast, model, session):
        model(says("WANTS x\nWHY NOW y\nOPEN z"))
        await agent.on_inbound(2, "wamid.1", "price kam karo bhai")
        assert escalations(session)[0].brief

    async def test_the_model_tool_path(self, fast, model, session):
        model(
            calls(("escalate_to_broker", {"conversation_id": 1,
                                          "reason": "buyer wants a human",
                                          "urgency": "high"})),
            says("WANTS x\nWHY NOW y\nOPEN z"),
        )
        await agent.on_inbound(1, "wamid.1", "get me a person")
        assert escalations(session)[0].brief

    async def test_the_evaluator_path(self, fast, model, session):
        model(
            calls(("update_buyer_profile", {"buyer_id": 1, "updates": {
                "budget_max": 9_000_000, "preferred_localities": ["Bopal"],
                "bhk_need": 3}})),
            says("Samajh gaya."),
        )
        await agent.on_inbound(1, "wamid.1", "Bopal, 90 lakh, 3BHK")
        assert escalations(session)[0].brief

    async def test_the_step_cap_path(self, fast, model, session):
        model(calls(("search_properties", {"bhk": 3})))
        await agent.on_inbound(1, "wamid.1", "3BHK")
        assert escalations(session)[0].reason == "step_cap_breached"
        assert escalations(session)[0].brief

    async def test_every_brief_carries_all_the_sections(self, fast, model, session):
        model(says("WANTS x\nWHY NOW y\nOPEN z"))
        await agent.on_inbound(2, "wamid.1", "price kam karo bhai")
        text = escalations(session)[0].brief
        for section in ("WANTS", "WHY NOW", "OPEN", "KNOWN", "UNKNOWN", "TRIGGER", "LAST 3"):
            assert section in text

    async def test_the_brief_names_the_buyer_it_is_about(self, fast, model, session):
        model(says("WANTS x\nWHY NOW y\nOPEN z"))
        await agent.on_inbound(2, "wamid.1", "price kam karo bhai")
        assert "Rakesh" in escalations(session)[0].brief
