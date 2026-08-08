"""The three endpoints (Phase 5 §1, §11 of the spec).

    POST /conversations/{id}/messages   -> {reply | null, tool_calls[], status}
    GET  /conversations/{id}            -> transcript + full trace
    GET  /broker/inbox                  -> escalated leads + briefs + post-escalation
                                           buyer messages

Three, and no more. A frontend that renders a chat window, a trace panel and a
broker inbox needs nothing else, and every extra route is a surface the demo has
to defend.

**The POST handler is `async def` and it blocks.** It awaits the shared
`asyncio.Event` every request for a conversation waits on (D8), so a coalesced
message returns only once the winning turn has finished. Worst case is
`DEBOUNCE_MAX_MS` plus a full step-capped turn, so client timeouts have to be set
generously — `demo.sh` uses 120s.

**DB sessions stay synchronous inside that async handler.** SQLite queries here
are sub-millisecond, so the event-loop block is real but negligible; the call
that actually takes time is the LLM request, and `llm.llm_call` already pushes
that onto a thread. Making the ORM async would buy nothing and cost the whole
`db.get_session()` contract the tools are written against.
"""
from __future__ import annotations

from collections import defaultdict
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlmodel import select

from app import agent, config, db
from app.db import (
    Buyer,
    Conversation,
    Escalation,
    GuardRejection,
    Message,
    ToolCall,
)
from app.tools import update_buyer_profile


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Both v0 seams fail loudly here rather than quietly at 3pm (D9, D24).

    `assert_ist` runs again even though importing `app.db` already ran it: this
    is the line an operator sees in the server log, and a startup that survives
    it is a startup whose clock and process model are both what the code assumes.
    """
    config.assert_ist()
    config.assert_single_worker()
    db.init_db()
    yield


app = FastAPI(
    title="AllSet — autonomous nurture agent",
    version="0.1.0",
    lifespan=lifespan,
)


STATIC_DIR = Path(__file__).parent / "static"


@app.get("/demo")
def demo_ui() -> FileResponse:
    """A single-page console: chat, trace panel and broker inbox in one tab.

    Talks to the three endpoints above with plain `fetch`, no build step. Not
    part of the product surface — a demo convenience the API doesn't depend on.
    """
    return FileResponse(STATIC_DIR / "demo.html")


class InboundMessage(BaseModel):
    """One simulated WhatsApp inbound.

    `wa_message_id` is optional so an exploratory curl is one field shorter, but
    the dedup path (D8) only exists when it is sent — the demo always sends it.
    """

    text: str = Field(min_length=1)
    wa_message_id: str | None = None


# --------------------------------------------------------------------------
# POST /conversations/{id}/messages
# --------------------------------------------------------------------------

@app.post("/conversations/{conversation_id}/messages")
async def post_message(conversation_id: int, body: InboundMessage) -> dict:
    """Six outcomes, one shape.

    | Situation                     | reply            | status      |
    |-------------------------------|------------------|-------------|
    | Normal turn                   | the reply        | active      |
    | Coalesced into another turn   | null             | coalesced   |
    | Duplicate wa_message_id       | null             | duplicate   |
    | Conversation already escalated| null             | escalated   |
    | The escalating turn itself    | the handoff line | escalated   |
    | Infra failure after failover  | a holding line   | active      |

    `tool_calls` is always present, `[]` where there is no trace — the frontend
    should never have to branch on a missing key.
    """
    if not body.text.strip():
        raise HTTPException(status_code=422, detail="text cannot be empty or blank.")

    with db.get_session() as session:
        if session.get(Conversation, conversation_id) is None:
            raise HTTPException(
                status_code=404,
                detail=f"No conversation with id {conversation_id}.",
            )

    result = await agent.on_inbound(conversation_id, body.wa_message_id, body.text)
    return {
        "reply": result.get("reply"),
        "tool_calls": result.get("tool_calls") or [],
        "status": result["status"],
    }


# --------------------------------------------------------------------------
# GET /conversations/{id}
# --------------------------------------------------------------------------

def _tool_call_view(row: ToolCall) -> dict:
    return {
        "tool_name": row.tool_name,
        "args": row.args,
        "result": row.result,
        "latency_ms": row.latency_ms,
    }


def _message_view(row: Message, traces: dict[int, list[dict]]) -> dict:
    """One transcript row and everything the turn recorded against it.

    `tool_calls` hangs off the **user** row, not the assistant one — `agent._run`
    opens the turn by saving the buyer's message and traces every dispatch
    against that id, so the trace panel reads chronologically: what the buyer
    said, what the agent did about it, what it then replied.
    """
    view = {
        "id": row.id,
        "role": row.role,
        "content": row.content,
        "created_at": row.created_at,
        "wa_message_id": row.wa_message_id,
        "tool_calls": traces.get(row.id, []),
    }
    if row.role == "assistant":
        # Telemetry only ever lands on assistant rows (D3, D20). `latency_ms` is
        # the enforced one: drain -> reply, the 8s target. `inbound_latency_ms`
        # is last inbound -> reply, which is what the buyer actually felt.
        view |= {
            "latency_ms": row.latency_ms,
            "inbound_latency_ms": row.inbound_latency_ms,
            "model_used": row.model_used,
            "provider": row.provider,
        }
    return view


@app.get("/conversations/{conversation_id}")
def get_conversation(conversation_id: int) -> dict:
    """The demo's trace panel: transcript joined to `tool_calls`, plus every
    guard rejection and the escalation if there is one.

    Guard rejections are here rather than in a fourth endpoint because they are
    the interesting half of the grounding story — a rejection is output that was
    generated and never sent, and there is nowhere else to see it.
    """
    with db.get_session() as session:
        convo = session.get(Conversation, conversation_id)
        if convo is None:
            raise HTTPException(
                status_code=404,
                detail=f"No conversation with id {conversation_id}.",
            )
        buyer = session.get(Buyer, convo.buyer_id)

        messages = session.exec(
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.id)
        ).all()

        traces: dict[int, list[dict]] = defaultdict(list)
        for row in session.exec(
            select(ToolCall)
            .join(Message, ToolCall.message_id == Message.id)
            .where(Message.conversation_id == conversation_id)
            .order_by(ToolCall.id)
        ).all():
            traces[row.message_id].append(_tool_call_view(row))

        rejections = session.exec(
            select(GuardRejection)
            .where(GuardRejection.conversation_id == conversation_id)
            .order_by(GuardRejection.id)
        ).all()

        escalation = session.exec(
            select(Escalation).where(Escalation.conversation_id == conversation_id)
        ).first()

        payload = {
            "conversation_id": convo.id,
            "status": convo.status,
            "clarification_count": convo.clarification_count,
            "created_at": convo.created_at,
            "buyer": {
                "id": buyer.id,
                "name": buyer.name,
                "phone": buyer.phone,
                "intent_tier": buyer.intent_tier,
            },
            "messages": [_message_view(row, traces) for row in messages],
            "guard_rejections": [
                {
                    "id": row.id,
                    "rejected_text": row.rejected_text,
                    "offending_spans": row.offending_spans,
                    "created_at": row.created_at,
                }
                for row in rejections
            ],
            "escalation": None if escalation is None else {
                "id": escalation.id,
                "reason": escalation.reason,
                "urgency": escalation.urgency,
                "brief": escalation.brief,
                "created_at": escalation.created_at,
            },
        }

    # Outside the session on purpose: this opens its own. An empty `updates` is a
    # read (D13), so the panel shows the same known/unknown state the turn saw.
    snapshot = update_buyer_profile(payload["buyer"]["id"], {})
    payload["buyer"]["profile"] = snapshot.get("profile") or {}
    payload["buyer"]["unknown"] = snapshot.get("unknown") or []
    return payload


# --------------------------------------------------------------------------
# GET /broker/inbox
# --------------------------------------------------------------------------

def _post_escalation_messages(session, conversation_id: int) -> list[Message]:
    """Buyer messages that arrived after the handoff line.

    Keyed off the **last assistant message id**, not a timestamp. After
    escalation the agent never speaks in this conversation again (D19), so the
    final assistant row is the handoff by construction and everything the buyer
    sent after it is post-escalation. Comparing `created_at` against the
    escalation row would work too, right up to the turn where both land in the
    same microsecond.
    """
    handoff_id = session.exec(
        select(Message.id)
        .where(Message.conversation_id == conversation_id)
        .where(Message.role == "assistant")
        .order_by(Message.id.desc())
    ).first()

    query = (
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .where(Message.role == "user")
    )
    if handoff_id is not None:
        query = query.where(Message.id > handoff_id)
    return session.exec(query.order_by(Message.id)).all()


@app.get("/broker/inbox")
def broker_inbox() -> dict:
    """Escalated leads, newest first, each with the brief and the messages the
    agent deliberately did not answer.

    Those unanswered messages are the reason this endpoint exists. Everything
    else here is also in `GET /conversations/{id}`; the queue of buyers waiting
    on a human, with what they said while waiting, is not.
    """
    leads = []
    with db.get_session() as session:
        rows = session.exec(
            select(Escalation).order_by(
                Escalation.created_at.desc(), Escalation.id.desc()
            )
        ).all()

        for escalation in rows:
            convo = session.get(Conversation, escalation.conversation_id)
            buyer = session.get(Buyer, convo.buyer_id)
            unread = _post_escalation_messages(session, escalation.conversation_id)
            leads.append({
                "escalation_id": escalation.id,
                "conversation_id": escalation.conversation_id,
                "escalated_at": escalation.created_at,
                "reason": escalation.reason,
                "urgency": escalation.urgency,
                "buyer": {
                    "id": buyer.id,
                    "name": buyer.name,
                    "phone": buyer.phone,
                    "intent_tier": buyer.intent_tier,
                },
                "brief": escalation.brief,
                "unread_count": len(unread),
                "unread_messages": [
                    {
                        "id": row.id,
                        "content": row.content,
                        "created_at": row.created_at,
                        "wa_message_id": row.wa_message_id,
                    }
                    for row in unread
                ],
            })

    return {"count": len(leads), "leads": leads}
