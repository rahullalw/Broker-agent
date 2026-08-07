"""Phase 4 §4 — the brief (D17).

`escalations.brief` had no producer: the tool takes no `brief` argument and the
regex path never invokes the model at all. One mechanism serves both paths — a
dedicated writer call for the prose, a deterministic facts block appended
server-side, and the guard run over the prose so the brief cannot quote a price
no tool ever returned.
"""
from types import SimpleNamespace

import pytest
from sqlmodel import select

from app import brief, llm
from app.db import Escalation, GuardRejection, Message, SiteVisit, ToolCall
from app.tools import escalate_to_broker, update_buyer_profile

pytestmark = pytest.mark.usefixtures("seeded")

PROSE = (
    "WANTS 3BHK in Bopal, ready to move.\n"
    "WHY NOW Site visit confirmed for this weekend.\n"
    'OPEN "Hi Priya, I\'m Kunal from AllSet — I\'ll meet you at the site."'
)


def says(text):
    return SimpleNamespace(
        choices=[SimpleNamespace(
            finish_reason="stop",
            message=SimpleNamespace(content=text, tool_calls=None),
        )]
    )


@pytest.fixture
def writer(monkeypatch):
    def install(outcome=PROSE):
        calls = []

        async def fake(messages, tools=None):
            calls.append({"messages": messages, "tools": tools})
            if isinstance(outcome, Exception):
                raise outcome
            return says(outcome), "google/gemini-3.1-flash-lite", "Google AI Studio"

        monkeypatch.setattr(llm, "llm_call", fake)
        return calls
    return install


@pytest.fixture
def escalated(session):
    update_buyer_profile(1, {
        "budget_min": 8_000_000, "budget_max": 9_000_000,
        "preferred_localities": ["Bopal"], "bhk_need": 3,
        "possession_need": "2026-12", "intent_tier": "hot",
    })
    escalate_to_broker(1, "site_visit_booked", "high")
    return 1


def brief_row(session, conversation_id=1):
    return session.exec(
        select(Escalation).where(Escalation.conversation_id == conversation_id)
    ).first()


class TestTheWriterCall:
    async def test_the_brief_lands_on_the_escalation_row(self, writer, escalated, session):
        writer()
        await brief.fill_brief(1)
        session.expire_all()
        assert brief_row(session).brief

    async def test_it_is_returned_as_well_as_stored(self, writer, escalated, session):
        writer()
        returned = await brief.fill_brief(1)
        session.expire_all()
        assert returned == brief_row(session).brief

    async def test_the_writer_gets_no_tools(self, writer, escalated):
        calls = writer()
        await brief.fill_brief(1)
        assert calls[0]["tools"] in (None, [])

    async def test_the_writer_sees_the_transcript_and_the_trigger(self, writer, escalated):
        calls = writer()
        await brief.fill_brief(1)
        sent = "\n".join(m["content"] for m in calls[0]["messages"])
        assert "Hi, looking for a 3BHK in Bopal" in sent      # the seeded opening
        assert "site_visit_booked" in sent

    async def test_exactly_one_extra_request_is_spent(self, writer, escalated):
        calls = writer()
        await brief.fill_brief(1)
        assert len(calls) == 1


class TestSections:
    @pytest.fixture(autouse=True)
    async def written(self, writer, escalated):
        writer()
        self.text = await brief.fill_brief(1)

    def test_the_three_written_sections_are_present(self):
        for section in ("WANTS", "WHY NOW", "OPEN"):
            assert section in self.text

    def test_the_deterministic_facts_block_is_present(self):
        for section in ("KNOWN", "UNKNOWN", "TRIGGER", "LAST 3"):
            assert section in self.text

    def test_the_header_identifies_the_buyer(self):
        assert "Priya" in self.text
        assert "+919825000001" in self.text
        assert "intent: hot" in self.text

    def test_known_uses_display_strings_not_raw_rupees(self):
        assert "₹80–90 lakh" in self.text
        assert "8000000" not in self.text

    def test_unknown_names_what_is_still_missing(self):
        assert "family size" in self.text

    def test_the_trigger_and_urgency_are_stated(self):
        assert "site_visit_booked" in self.text
        assert "high" in self.text

    def test_the_last_three_messages_are_verbatim(self):
        assert "Hi, looking for a 3BHK in Bopal" in self.text


class TestSiteVisitLine:
    async def test_a_confirmed_visit_is_spelled_out(self, writer, escalated, session):
        from datetime import datetime, timedelta
        slot = (datetime.now() + timedelta(days=3)).replace(
            hour=17, minute=0, second=0, microsecond=0
        )
        session.add(SiteVisit(buyer_id=1, property_id=5, slot=slot, idempotency_key="k1"))
        session.commit()
        writer()
        text = await brief.fill_brief(1)
        assert "VISIT" in text
        assert "5:00 pm" in text
        assert "Shrishti Elara, Bopal" in text

    async def test_no_visit_means_no_visit_line(self, writer, escalated):
        writer()
        assert "VISIT" not in await brief.fill_brief(1)


class TestTheGuardRunsOverTheProse:
    async def test_an_ungrounded_price_is_not_allowed_into_the_brief(
        self, writer, escalated, session
    ):
        writer(
            "WANTS 3BHK in Bopal at ₹72 lakh.\n"
            "WHY NOW Site visit confirmed.\n"
            'OPEN "Hi Priya."'
        )
        text = await brief.fill_brief(1)
        assert "₹72 lakh" not in text

    async def test_the_rejected_prose_is_logged(self, writer, escalated, session):
        writer(
            "WANTS 3BHK in Bopal at ₹72 lakh.\n"
            "WHY NOW Site visit confirmed.\n"
            'OPEN "Hi Priya."'
        )
        await brief.fill_brief(1)
        row = session.exec(select(GuardRejection)).all()[0]
        assert "₹72 lakh" in row.rejected_text
        assert "₹72 lakh" in row.offending_spans

    async def test_a_figure_a_tool_really_returned_survives(
        self, writer, escalated, session
    ):
        message = Message(conversation_id=1, role="user", content="which ones?")
        session.add(message)
        session.commit()
        session.refresh(message)
        session.add(ToolCall(
            message_id=message.id, tool_name="search_properties", args={},
            result={"exact_matches": [{"price_display": "₹86 lakh"}]}, latency_ms=1,
        ))
        session.commit()
        writer(
            "WANTS 3BHK in Bopal around ₹86 lakh.\n"
            "WHY NOW Site visit confirmed.\n"
            'OPEN "Hi Priya."'
        )
        assert "₹86 lakh" in await brief.fill_brief(1)

    async def test_the_buyers_own_words_are_an_allowed_source(
        self, writer, escalated, session
    ):
        session.add(Message(conversation_id=1, role="user",
                            content="budget 85 lakh tak hai"))
        session.commit()
        writer(
            "WANTS 3BHK in Bopal, budget 85 lakh.\n"
            "WHY NOW Site visit confirmed.\n"
            'OPEN "Hi Priya."'
        )
        assert "85 lakh" in await brief.fill_brief(1)


class TestBothPathsAlwaysProduceABrief:
    async def test_a_writer_failure_still_yields_every_section(self, writer, escalated):
        writer(llm.LLMUnavailable("every model failed"))
        text = await brief.fill_brief(1)
        for section in ("WANTS", "WHY NOW", "OPEN", "KNOWN", "UNKNOWN", "TRIGGER", "LAST 3"):
            assert section in text

    async def test_unparseable_prose_falls_back_rather_than_shipping_garbage(
        self, writer, escalated
    ):
        writer("Sure! Here is a summary of the buyer for you.")
        text = await brief.fill_brief(1)
        assert "WANTS" in text and "WHY NOW" in text and "OPEN" in text
        assert "Here is a summary" not in text

    async def test_an_empty_reply_falls_back(self, writer, escalated):
        writer("")
        assert "WANTS" in await brief.fill_brief(1)

    async def test_the_regex_path_with_no_tool_calls_still_gets_a_brief(
        self, writer, session
    ):
        # "price kam karo bhai" escalates before any tool is ever called.
        session.add(Message(conversation_id=2, role="user", content="price kam karo bhai"))
        session.commit()
        escalate_to_broker(2, "negotiation_request", "high")
        writer(llm.LLMUnavailable("offline"))
        text = await brief.fill_brief(2)
        assert "negotiation_request" in text
        assert "price kam karo bhai" in text

    async def test_a_blank_profile_reads_as_unknown_not_as_a_crash(self, writer, session):
        escalate_to_broker(3, "human_requested", "high")
        writer()
        text = await brief.fill_brief(3)
        assert "UNKNOWN" in text
        assert "budget" in text


class TestIdempotence:
    async def test_rewriting_does_not_stack_two_briefs(self, writer, escalated, session):
        writer()
        first = await brief.fill_brief(1)
        second = await brief.fill_brief(1)
        assert second == first
        session.expire_all()
        assert brief_row(session).brief.count("WANTS") == 1

    async def test_an_unescalated_conversation_gets_nothing(self, writer, session):
        writer()
        assert await brief.fill_brief(2) is None
