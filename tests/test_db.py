"""Section 5 - eight tables, JSON list columns (D23), naive IST datetimes (D24)."""
import os
import subprocess
import sys
from datetime import datetime

import pytest
from sqlalchemy import inspect
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, create_engine, select

from app import db
from app.db import (
    Buyer, Conversation, Escalation, GuardRejection, Message, Property,
    SiteVisit, ToolCall,
)

EXPECTED_TABLES = {
    "properties", "buyers", "conversations", "messages",
    "tool_calls", "escalations", "site_visits", "guard_rejections",
}


@pytest.fixture
def engine():
    eng = create_engine("sqlite://")          # in-memory, isolated per test
    db.init_db(eng)
    return eng


@pytest.fixture
def session(engine):
    with Session(engine) as s:
        yield s


@pytest.fixture
def buyer(session):
    b = Buyer(name="Test Buyer", phone="+919000000001")
    session.add(b)
    session.commit()
    session.refresh(b)
    return b


@pytest.fixture
def prop(session):
    p = Property(
        title="Test Tower", locality="Bopal", bhk=3, price_inr=9_500_000,
        carpet_area_sqft=1240, possession_date=datetime(2026, 12, 1),
        status="under_construction",
        rera_id="PR/GJ/AHMEDABAD/AHMEDABAD CITY/AUDA/DEMO0001/EX1/2026",
        developer="Test Developers", tower="A", floor=7,
        amenities=["gym", "lift"], highlight_note="Test note.",
    )
    session.add(p)
    session.commit()
    session.refresh(p)
    return p


@pytest.fixture
def convo(session, buyer):
    c = Conversation(buyer_id=buyer.id)
    session.add(c)
    session.commit()
    session.refresh(c)
    return c


class TestSchema:
    def test_exactly_eight_tables(self, engine):
        assert set(inspect(engine).get_table_names()) == EXPECTED_TABLES

    def test_guard_rejections_is_its_own_table_not_a_message_column(self, engine):
        # Section 8 logs text that was never sent, so it must not need a messages row.
        columns = {c["name"] for c in inspect(engine).get_columns("guard_rejections")}
        assert {"conversation_id", "rejected_text", "offending_spans"} <= columns
        assert "message_id" not in columns

    def test_importing_app_db_creates_the_database_file(self, tmp_path):
        target = tmp_path / "allset.db"
        result = subprocess.run(
            [sys.executable, "-c", "import app.db; print('IMPORTED')"],
            capture_output=True, text=True, cwd=os.getcwd(),
            env={**os.environ, "DATABASE_URL": "sqlite:///" + target.as_posix()},
        )
        assert result.returncode == 0, result.stderr
        assert target.exists()
        created = create_engine("sqlite:///" + target.as_posix())
        assert set(inspect(created).get_table_names()) == EXPECTED_TABLES


class TestProperty:
    def test_amenities_round_trip_as_a_json_list(self, session, prop):
        fetched = session.get(Property, prop.id)
        assert fetched.amenities == ["gym", "lift"]
        assert isinstance(fetched.amenities, list)

    def test_empty_amenities_defaults_to_a_list_not_null(self, session):
        p = Property(
            title="Bare", locality="Shela", bhk=2, price_inr=4_800_000,
            carpet_area_sqft=900, possession_date=datetime(2024, 1, 1),
            status="ready_to_move", rera_id="DEMO", developer="D", tower="B",
            floor=1, highlight_note="x",
        )
        session.add(p)
        session.commit()
        session.refresh(p)
        assert p.amenities == []

    def test_possession_date_round_trips_as_a_naive_datetime(self, session, prop):
        fetched = session.get(Property, prop.id)
        assert fetched.possession_date == datetime(2026, 12, 1)
        assert fetched.possession_date.tzinfo is None, "D24 - naive IST, never aware"

    def test_price_is_stored_as_raw_rupees(self, session, prop):
        assert session.get(Property, prop.id).price_inr == 9_500_000


class TestBuyer:
    def test_qualification_fields_default_to_unknown(self, session, buyer):
        # D12 - "unknown" is the starting state and drives the profile block.
        fetched = session.get(Buyer, buyer.id)
        assert fetched.budget_min is None
        assert fetched.budget_max is None
        assert fetched.preferred_localities is None
        assert fetched.bhk_need is None
        assert fetched.possession_need is None
        assert fetched.family_size is None

    def test_intent_tier_starts_cold(self, session, buyer):
        assert session.get(Buyer, buyer.id).intent_tier == "cold"

    def test_preferred_localities_round_trip_as_json(self, session, buyer):
        buyer.preferred_localities = ["Bopal", "Shela"]
        session.add(buyer)
        session.commit()
        session.refresh(buyer)
        assert buyer.preferred_localities == ["Bopal", "Shela"]


class TestConversation:
    def test_defaults(self, session, convo):
        assert convo.status == "active"
        assert convo.clarification_count == 0   # D16 third-failed-clarification trigger
        assert isinstance(convo.created_at, datetime)

    def test_created_at_is_naive_local_time(self, session, convo):
        assert convo.created_at.tzinfo is None
        assert abs((datetime.now() - convo.created_at).total_seconds()) < 60

    def test_orphan_buyer_id_is_rejected(self, session):
        # SQLite ignores foreign keys unless the pragma is on; the fixed ids in
        # D21 are only meaningful if the constraint actually bites.
        session.add(Conversation(buyer_id=9999))
        with pytest.raises(IntegrityError):
            session.commit()


class TestMessage:
    def _msg(self, conversation_id, **kw):
        return Message(conversation_id=conversation_id, role="user", content="hi", **kw)

    def test_assistant_telemetry_columns_are_nullable_on_user_rows(self, session, convo):
        m = self._msg(convo.id)
        session.add(m)
        session.commit()
        session.refresh(m)
        assert m.latency_ms is None
        assert m.model_used is None
        assert m.provider is None

    def test_assistant_row_carries_serving_model_and_latency(self, session, convo):
        # D3 logs the serving model per step; D20 the latency.
        m = Message(conversation_id=convo.id, role="assistant", content="hello",
                    latency_ms=1234, model_used="google/gemini-2.5-flash",
                    provider="openrouter")
        session.add(m)
        session.commit()
        session.refresh(m)
        assert m.latency_ms == 1234
        assert m.model_used == "google/gemini-2.5-flash"
        assert m.provider == "openrouter"

    def test_wa_message_id_is_unique(self, session, convo):
        session.add(self._msg(convo.id, wa_message_id="wamid.ABC"))
        session.commit()
        session.add(self._msg(convo.id, wa_message_id="wamid.ABC"))
        with pytest.raises(IntegrityError):
            session.commit()

    def test_many_rows_may_have_no_wa_message_id(self, session, convo):
        session.add_all([self._msg(convo.id), self._msg(convo.id), self._msg(convo.id)])
        session.commit()
        assert len(session.exec(select(Message)).all()) == 3


class TestToolCall:
    def test_args_and_result_round_trip_as_json_objects(self, session, convo):
        m = Message(conversation_id=convo.id, role="assistant", content="ok")
        session.add(m)
        session.commit()
        session.refresh(m)

        call = ToolCall(
            message_id=m.id, tool_name="search_properties",
            args={"locality": "Bopal", "bhk": 3},
            result={"matches": [{"id": 1}], "relaxed": None},
            latency_ms=12,
        )
        session.add(call)
        session.commit()
        session.refresh(call)
        assert call.args["bhk"] == 3
        assert call.result["matches"][0]["id"] == 1


class TestEscalationAndSiteVisit:
    def test_escalation_stores_reason_urgency_and_brief(self, session, convo):
        e = Escalation(conversation_id=convo.id, reason="price_negotiation",
                       urgency="high", brief="Buyer asked for a discount.")
        session.add(e)
        session.commit()
        session.refresh(e)
        assert e.urgency == "high"
        assert isinstance(e.created_at, datetime)

    def test_site_visit_idempotency_key_is_unique(self, session, buyer, prop):
        for _ in range(2):
            session.add(SiteVisit(buyer_id=buyer.id, property_id=prop.id,
                                  slot=datetime(2026, 8, 12, 11, 0),
                                  idempotency_key="b1-p1-20260812T1100"))
        with pytest.raises(IntegrityError):
            session.commit()

    def test_guard_rejection_stores_full_text_and_spans(self, session, convo):
        g = GuardRejection(conversation_id=convo.id,
                           rejected_text="I can offer 80 lakh, a 10% discount.",
                           offending_spans=["80 lakh", "10%"])
        session.add(g)
        session.commit()
        session.refresh(g)
        assert g.offending_spans == ["80 lakh", "10%"]
        assert "10%" in g.rejected_text


class TestNoUtcnow:
    def test_app_package_never_calls_utcnow(self):
        # D24 guard - the cheap grep the spec asks CI to run.
        hits = []
        for root, _, files in os.walk("app"):
            for name in files:
                if not name.endswith(".py"):
                    continue
                path = os.path.join(root, name)
                with open(path, encoding="utf-8") as fh:
                    text = fh.read()
                if "utcnow" in text or "timezone.utc" in text:
                    hits.append(path)
        assert hits == []
