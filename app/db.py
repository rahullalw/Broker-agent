"""SQLModel tables and the engine (Section 5).

SQLite has no ARRAY type, so every list column is JSON (D23). It also has no
native datetime: values round-trip as strings and every comparison is
naive-local, which is only correct because config.assert_ist() guarantees local
time is IST (D24).
"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import Column, event
from sqlalchemy.engine import Engine
from sqlmodel import JSON, Field, Session, SQLModel, create_engine

from app import config
from app.enums import CONVO_STATUS  # noqa: F401  (vocabulary lives in enums.py)

config.assert_ist()


@event.listens_for(Engine, "connect")
def _enable_sqlite_foreign_keys(dbapi_connection, connection_record) -> None:
    """SQLite ignores REFERENCES unless asked not to. Without this the foreign
    keys below are decoration, and the fixed seed ids (D21) prove nothing."""
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


class Property(SQLModel, table=True):
    __tablename__ = "properties"

    id: int | None = Field(default=None, primary_key=True)
    title: str
    locality: str                      # in LOCALITIES
    bhk: int
    price_inr: int                     # raw rupees - never shown to the buyer
    carpet_area_sqft: int
    possession_date: datetime          # naive IST; a past date means ready to move
    status: str                        # in STATUS
    rera_id: str                       # real Gujarat format, DEMO content (D22)
    developer: str
    tower: str
    floor: int
    amenities: list[str] = Field(default_factory=list, sa_column=Column(JSON, nullable=False))
    highlight_note: str


class Buyer(SQLModel, table=True):
    __tablename__ = "buyers"

    id: int | None = Field(default=None, primary_key=True)
    name: str
    phone: str
    # Every qualification field is nullable: "unknown" is the starting state and
    # is what the per-turn known/unknown block is built from (D12).
    budget_min: int | None = None
    budget_max: int | None = None
    preferred_localities: list[str] | None = Field(
        default=None, sa_column=Column(JSON, nullable=True)
    )
    bhk_need: int | None = None
    possession_need: datetime | None = None
    family_size: int | None = None
    intent_tier: str = "cold"          # in INTENT_TIER


class Conversation(SQLModel, table=True):
    __tablename__ = "conversations"

    id: int | None = Field(default=None, primary_key=True)
    buyer_id: int = Field(foreign_key="buyers.id")
    status: str = "active"             # in CONVO_STATUS
    # Feeds the deterministic evaluator's "third failed clarification" trigger (D16).
    clarification_count: int = 0
    created_at: datetime = Field(default_factory=datetime.now)


class Message(SQLModel, table=True):
    __tablename__ = "messages"

    id: int | None = Field(default=None, primary_key=True)
    conversation_id: int = Field(foreign_key="conversations.id")
    role: str                          # only "user" or "assistant" are ever written (D11)
    content: str
    created_at: datetime = Field(default_factory=datetime.now)
    wa_message_id: str | None = Field(default=None, unique=True, index=True)
    # Populated on assistant rows only (D3, D20).
    # `latency_ms` is drain -> reply, the enforced 8s target: debounce is
    # deliberate waiting, not latency. `inbound_latency_ms` is last inbound ->
    # reply, which is what the buyer actually experienced.
    latency_ms: int | None = None
    inbound_latency_ms: int | None = None
    model_used: str | None = None
    provider: str | None = None


class ToolCall(SQLModel, table=True):
    __tablename__ = "tool_calls"

    id: int | None = Field(default=None, primary_key=True)
    message_id: int = Field(foreign_key="messages.id")
    tool_name: str
    # Not logging: this is the trace panel and the grounding audit trail.
    args: dict = Field(default_factory=dict, sa_column=Column(JSON, nullable=False))
    result: dict = Field(default_factory=dict, sa_column=Column(JSON, nullable=False))
    latency_ms: int | None = None


class Escalation(SQLModel, table=True):
    __tablename__ = "escalations"

    id: int | None = Field(default=None, primary_key=True)
    conversation_id: int = Field(foreign_key="conversations.id")
    reason: str
    urgency: str                       # in URGENCY
    brief: str
    created_at: datetime = Field(default_factory=datetime.now)


class SiteVisit(SQLModel, table=True):
    __tablename__ = "site_visits"

    id: int | None = Field(default=None, primary_key=True)
    buyer_id: int = Field(foreign_key="buyers.id")
    property_id: int = Field(foreign_key="properties.id")
    slot: datetime
    idempotency_key: str = Field(unique=True, index=True)


class GuardRejection(SQLModel, table=True):
    """Section 8 requires logging rejected model output in full. It lives here
    rather than on messages because these rows were never sent to anyone."""

    __tablename__ = "guard_rejections"

    id: int | None = Field(default=None, primary_key=True)
    conversation_id: int = Field(foreign_key="conversations.id")
    rejected_text: str
    offending_spans: list[str] = Field(
        default_factory=list, sa_column=Column(JSON, nullable=False)
    )
    created_at: datetime = Field(default_factory=datetime.now)


engine = create_engine(config.DATABASE_URL, connect_args={"check_same_thread": False})


def init_db(target: Engine | None = None) -> Engine:
    """Create every table. Safe to call repeatedly."""
    target = target or engine
    SQLModel.metadata.create_all(target)
    return target


def get_session() -> Session:
    return Session(engine)


init_db()
