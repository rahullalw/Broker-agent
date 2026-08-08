"""Phase 4 §3 and §5 — the deterministic evaluator (D16) and the handoff (D19).

Most of §9's "model-judged" tier is not a judgement at all. Computing it after
each turn makes escalation **guaranteed** rather than hoped for, and it is what
catches the case the model itself misses.
"""
import pytest
from sqlmodel import select

from app import escalation
from app.db import Buyer, Conversation, Message, ToolCall
from app.tools import escalate_to_broker, update_buyer_profile

pytestmark = pytest.mark.usefixtures("seeded")


@pytest.fixture
def turn(session):
    """A user message to hang this turn's tool calls off."""
    message = Message(conversation_id=1, role="user", content="…")
    session.add(message)
    session.commit()
    session.refresh(message)
    return message.id


def log(session, message_id, tool_name, args, result):
    session.add(ToolCall(message_id=message_id, tool_name=tool_name,
                         args=args, result=result, latency_ms=1))
    session.commit()


def qualify(**fields):
    update_buyer_profile(1, fields)


class TestSiteVisitBooked:
    """D15 — book first, then escalate. The trigger is 'booked', not 'requested'."""

    def test_a_confirmed_booking_this_turn_fires(self, session, turn):
        log(session, turn, "book_site_visit",
            {"buyer_id": 1, "property_id": 5},
            {"confirmed": True, "slot_display": "Sunday, 9 August 2026, 5:00 pm",
             "property": "Shrishti Elara, Bopal"})
        assert escalation.evaluate(1, turn) == ("site_visit_booked", "high")

    def test_a_rejected_slot_does_not_fire(self, session, turn):
        log(session, turn, "book_site_visit", {"slot": "kal shaam"},
            {"error": "'kal shaam' is not a date."})
        assert escalation.evaluate(1, turn) is None

    def test_merely_searching_does_not_fire(self, session, turn):
        log(session, turn, "search_properties", {"bhk": 3}, {"exact_matches": []})
        assert escalation.evaluate(1, turn) is None

    def test_a_booking_from_an_earlier_turn_does_not_refire(self, session, turn):
        log(session, turn, "book_site_visit", {}, {"confirmed": True})
        later = Message(conversation_id=1, role="user", content="thanks")
        session.add(later)
        session.commit()
        session.refresh(later)
        assert escalation.evaluate(1, later.id) is None


class TestQualificationComplete:
    """The bar is deliberately raised past §9 — budget + locality alone fires at
    turn three, and a nurture agent that nurtures for ninety seconds is not one.
    Every field in the qualification order has to be answered."""

    def test_budget_and_locality_alone_do_not_fire(self, turn):
        qualify(budget_min=5_500_000, budget_max=6_500_000,
                preferred_localities=["Bopal"])
        assert escalation.evaluate(1, turn) is None

    def test_the_whole_qualification_order_fires(self, turn):
        qualify(budget_max=6_500_000, preferred_localities=["Bopal"], bhk_need=3,
                possession_need="2026-12", family_size=4)
        assert escalation.evaluate(1, turn) == ("qualification_complete", "medium")

    def test_everything_but_family_size_does_not_fire(self, turn):
        qualify(budget_max=6_500_000, preferred_localities=["Bopal"], bhk_need=3,
                possession_need="2026-12")
        assert escalation.evaluate(1, turn) is None

    def test_everything_but_possession_does_not_fire(self, turn):
        qualify(budget_max=6_500_000, preferred_localities=["Bopal"], bhk_need=3,
                family_size=4)
        assert escalation.evaluate(1, turn) is None

    def test_bhk_without_a_budget_does_not_fire(self, turn):
        qualify(preferred_localities=["Bopal"], bhk_need=3,
                possession_need="2026-12", family_size=4)
        assert escalation.evaluate(1, turn) is None

    def test_a_budget_without_a_locality_does_not_fire(self, turn):
        qualify(budget_max=6_500_000, bhk_need=3, possession_need="2026-12",
                family_size=4)
        assert escalation.evaluate(1, turn) is None

    def test_a_blank_profile_does_not_fire(self, turn):
        assert escalation.evaluate(1, turn) is None


class TestThirdFailedClarification:
    def test_three_fires(self, session, turn):
        convo = session.get(Conversation, 1)
        convo.clarification_count = 3
        session.add(convo)
        session.commit()
        assert escalation.evaluate(1, turn) == ("clarification_exhausted", "medium")

    def test_two_does_not(self, session, turn):
        convo = session.get(Conversation, 1)
        convo.clarification_count = 2
        session.add(convo)
        session.commit()
        assert escalation.evaluate(1, turn) is None


class TestPossessionAndPriceAsked:
    def test_both_sections_across_the_conversation_fire(self, session, turn):
        earlier = Message(conversation_id=1, role="user", content="possession?")
        session.add(earlier)
        session.commit()
        session.refresh(earlier)
        log(session, earlier.id, "get_property_details",
            {"property_id": 5, "sections": ["possession"]}, {"id": 5})
        log(session, turn, "get_property_details",
            {"property_id": 5, "sections": ["pricing"]}, {"id": 5})
        assert escalation.evaluate(1, turn) == ("possession_and_price_asked", "high")

    def test_both_in_one_call_fire(self, session, turn):
        log(session, turn, "get_property_details",
            {"property_id": 5, "sections": ["pricing", "possession"]}, {"id": 5})
        assert escalation.evaluate(1, turn) == ("possession_and_price_asked", "high")

    def test_possession_alone_does_not_fire(self, session, turn):
        log(session, turn, "get_property_details",
            {"property_id": 5, "sections": ["possession"]}, {"id": 5})
        assert escalation.evaluate(1, turn) is None

    def test_another_conversations_questions_do_not_count(self, session, turn):
        other = Message(conversation_id=2, role="user", content="price?")
        session.add(other)
        session.commit()
        session.refresh(other)
        log(session, other.id, "get_property_details",
            {"property_id": 5, "sections": ["pricing"]}, {"id": 5})
        log(session, turn, "get_property_details",
            {"property_id": 5, "sections": ["possession"]}, {"id": 5})
        assert escalation.evaluate(1, turn) is None


class TestPrecedence:
    def test_a_booking_outranks_a_completed_qualification(self, session, turn):
        qualify(budget_max=6_500_000, preferred_localities=["Bopal"], bhk_need=3)
        log(session, turn, "book_site_visit", {}, {"confirmed": True})
        assert escalation.evaluate(1, turn) == ("site_visit_booked", "high")

    def test_the_verdict_is_stable_across_repeated_calls(self, session, turn):
        qualify(budget_max=6_500_000, preferred_localities=["Bopal"], bhk_need=3)
        assert len({escalation.evaluate(1, turn) for _ in range(5)}) == 1


class TestDedupe:
    """One conversation, one escalation row — whichever authority fired."""

    def test_an_already_escalated_conversation_is_left_alone(self, session, turn):
        qualify(budget_max=6_500_000, preferred_localities=["Bopal"], bhk_need=3)
        escalate_to_broker(1, "buyer asked for a human", "high")
        assert escalation.evaluate(1, turn) is None

    def test_the_tool_refuses_to_write_a_second_row(self, session):
        first = escalate_to_broker(1, "buyer asked for a human", "high")
        second = escalate_to_broker(1, "site visit booked", "high")
        assert second["escalation_id"] == first["escalation_id"]
        assert len(session.exec(select(escalation.Escalation)).all()) == 1

    def test_the_original_reason_survives_a_second_attempt(self, session):
        escalate_to_broker(1, "buyer asked for a human", "high")
        escalate_to_broker(1, "site visit booked", "medium")
        row = session.exec(select(escalation.Escalation)).all()[0]
        assert row.reason == "buyer asked for a human"
        assert row.urgency == "high"


class TestClarificationCounter:
    """`clarification_count` had no producer. A turn that captured nothing and
    ended in a question is what a failed clarification looks like."""

    def test_a_question_that_captured_nothing_increments(self, session, turn):
        escalation.update_clarification_count(1, [], "Budget kitna hai?")
        session.expire_all()
        assert session.get(Conversation, 1).clarification_count == 1

    def test_a_successful_capture_resets_it(self, session, turn):
        escalation.update_clarification_count(1, [], "Budget kitna hai?")
        escalation.update_clarification_count(
            1,
            [{"tool_name": "update_buyer_profile", "result": {"profile": {}}}],
            "Bopal ya Shela?",
        )
        session.expire_all()
        assert session.get(Conversation, 1).clarification_count == 0

    def test_a_steering_error_does_not_count_as_a_capture(self, session, turn):
        escalation.update_clarification_count(
            1,
            [{"tool_name": "update_buyer_profile", "result": {"error": "no match"}}],
            "Kaunsi area?",
        )
        session.expire_all()
        assert session.get(Conversation, 1).clarification_count == 1

    def test_a_statement_rather_than_a_question_does_not_increment(self, session, turn):
        escalation.update_clarification_count(1, [], "Bopal me teen options hain.")
        session.expire_all()
        assert session.get(Conversation, 1).clarification_count == 0

    def test_three_failures_reach_the_evaluator_threshold(self, session, turn):
        for _ in range(3):
            escalation.update_clarification_count(1, [], "Budget kitna hai?")
        assert escalation.evaluate(1, turn) == ("clarification_exhausted", "medium")

    def test_a_turn_that_did_real_work_is_not_a_failed_clarification(
        self, session, turn
    ):
        """§10 tells the agent to end every message with a question, so a
        trailing "?" alone is true on nearly every turn. Counting those escalates
        any conversation that runs three turns without a profile write — even one
        where the agent searched, quoted possession and read back a RERA id."""
        escalation.update_clarification_count(
            1,
            [{"tool_name": "search_properties", "result": {"exact_matches": [{}]}}],
            "Do any of these work for you?",
        )
        session.expire_all()
        assert session.get(Conversation, 1).clarification_count == 0

    def test_a_full_browsing_arc_never_reaches_the_threshold(self, session, turn):
        # search -> possession -> legal, each ending in a question. This is the
        # scripted demo, and it must not hand off on turn three.
        for tool in ("search_properties", "get_property_details", "get_property_details"):
            escalation.update_clarification_count(
                1, [{"tool_name": tool, "result": {"id": 5}}], "Anything else?"
            )
        session.expire_all()
        assert session.get(Conversation, 1).clarification_count == 0
        assert escalation.evaluate(1, turn) is None

    def test_a_turn_where_every_tool_errored_still_counts(self, session, turn):
        # Nothing was achieved, so the clarification genuinely failed.
        escalation.update_clarification_count(
            1,
            [{"tool_name": "search_properties", "result": {"error": "No match for x"}}],
            "Which locality did you mean?",
        )
        session.expire_all()
        assert session.get(Conversation, 1).clarification_count == 1

    def test_progress_stops_the_count_rising_without_resetting_it(
        self, session, turn
    ):
        """Only the buyer answering resets the run. A useful turn just does not
        add to it."""
        escalation.update_clarification_count(1, [], "Budget kitna hai?")
        escalation.update_clarification_count(
            1, [{"tool_name": "search_properties", "result": {}}], "In Bopal or Shela?"
        )
        session.expire_all()
        assert session.get(Conversation, 1).clarification_count == 1


class TestHandoffLine:
    def test_hinglish_gets_the_hinglish_line(self):
        assert escalation.handoff_line("price kam karo bhai") == escalation.HANDOFF_HI

    def test_english_gets_the_english_line(self):
        assert escalation.handoff_line(
            "Can I visit this Saturday at 5pm?"
        ) == escalation.HANDOFF_EN

    def test_it_defaults_to_english(self):
        assert escalation.handoff_line("") == escalation.HANDOFF_EN
        assert escalation.handoff_line(None) == escalation.HANDOFF_EN

    @pytest.mark.parametrize("text", [
        "3bhk chahiye bhai, budget 65 lakh",
        "aadmi se baat karao",
        "possession kab tak milega?",
        "EMI kitna banega",
    ])
    def test_the_seeded_hinglish_messages_are_recognised(self, text):
        assert escalation.handoff_line(text) == escalation.HANDOFF_HI

    @pytest.mark.parametrize("text", [
        "Hi, looking for a 3BHK in Bopal",
        "Tell me about the second one",
        "Is it RERA registered?",
    ])
    def test_the_seeded_english_messages_are_recognised(self, text):
        assert escalation.handoff_line(text) == escalation.HANDOFF_EN

    def test_neither_line_carries_a_fact_the_guard_could_reject(self):
        from app import guard
        assert guard.detect(escalation.HANDOFF_HI) == []
        assert guard.detect(escalation.HANDOFF_EN) == []

    def test_the_lines_are_the_ones_the_spec_names(self):
        assert escalation.HANDOFF_EN == (
            "I'm connecting you with our broker, they'll reach out shortly."
        )
        assert escalation.HANDOFF_HI == (
            "Main aapko humare broker se connect kar raha hoon, "
            "wo thodi der mein aapse baat karenge."
        )
