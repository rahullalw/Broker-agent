"""Inbound handling, debounce, the per-conversation lock and the turn (Phase 3 §3).

**Process model (D9).** Single worker: `uvicorn app.main:app --workers 1`. The
lock, the buffers and the events are in-process dicts keyed by
`conversation_id`. There is deliberately no Postgres advisory lock — it was a
cross-process primitive guarding in-memory state, which cannot work. State dies
on restart and mid-debounce messages are lost; that is acceptable for v0 and it
is the explicit v1 seam.

**Debounce (D10).** Real buyers type "Hi" / "3bhk chahiye" / "bopal me" as three
messages in five seconds. Each inbound resets a sliding `DEBOUNCE_MS` timer; if
the buffer reaches `DEBOUNCE_MAX_MS` the turn fires anyway, so a fast typer
cannot starve it.

**Shared future (D8).** Every POST for a conversation awaits the same event. The
request that arrived last returns the reply; earlier ones return
`{"reply": None, "status": "coalesced"}`.

**Text-only transcript (D11).** Assistant-with-tool_calls and `role: "tool"`
messages exist only in the in-memory list for the duration of the turn. Nothing
pairs a `tool_call_id` across turns, so there are no poisoned rows to maintain.
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field

import openai
from sqlmodel import select

from app import brief, config, db, escalation, guard, llm, triggers
from app.db import Conversation, Message, ToolCall
from app.dispatch import dispatch
from app.prompt import build_system_prompt
from app.schemas import TOOLS
from app.tools import escalate_to_broker, update_buyer_profile

STEP_CAP = 8
TRANSCRIPT_TURNS = 20

# Infrastructure trouble is not a reason to hand a lead to a human (D20), so the
# buyer gets a holding line and the conversation stays active for the next try.
HOLD_MESSAGE = "One moment — let me check that and come right back to you."


# --------------------------------------------------------------------------
# In-process state
# --------------------------------------------------------------------------

@dataclass
class Batch:
    """One debounce window: the fragments, and everyone waiting on them."""

    texts: list[str] = field(default_factory=list)
    event: asyncio.Event = field(default_factory=asyncio.Event)
    wa_message_id: str | None = None
    first_at: float = 0.0
    last_at: float = 0.0
    last_token: int = 0
    winner: int | None = None
    result: dict | None = None


@dataclass
class ConversationState:
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    current: Batch = field(default_factory=Batch)
    timer: asyncio.Task | None = None
    seq: int = 0


STATE: dict[int, ConversationState] = {}

# WhatsApp resends. The unique constraint on messages.wa_message_id is the
# durable half; this set covers ids still sitting in a debounce buffer, which
# have no row yet.
SEEN: set[str] = set()


def reset_state() -> None:
    """Test hook — the process-level state has no other lifecycle."""
    STATE.clear()
    SEEN.clear()


def state_for(conversation_id: int) -> ConversationState:
    return STATE.setdefault(conversation_id, ConversationState())


def lock_for(conversation_id: int) -> asyncio.Lock:
    return state_for(conversation_id).lock


# --------------------------------------------------------------------------
# Inbound (D8, D10)
# --------------------------------------------------------------------------

def duplicate(wa_message_id: str | None) -> bool:
    if wa_message_id is None:
        return False
    if wa_message_id in SEEN:
        return True
    with db.get_session() as session:
        already = session.exec(
            select(Message).where(Message.wa_message_id == wa_message_id)
        ).first()
    return already is not None


async def _debounce(conversation_id: int, batch: Batch) -> None:
    loop = asyncio.get_running_loop()
    while True:
        sliding = batch.last_at + config.DEBOUNCE_MS / 1000
        hard_cap = batch.first_at + config.DEBOUNCE_MAX_MS / 1000
        delay = min(sliding, hard_cap) - loop.time()
        if delay <= 0:
            break
        await asyncio.sleep(delay)

    state_for(conversation_id).timer = None
    await run_turn(conversation_id)


def _ensure_timer(conversation_id: int, state: ConversationState) -> None:
    if state.timer is None or state.timer.done():
        state.timer = asyncio.create_task(_debounce(conversation_id, state.current))


async def on_inbound(conversation_id: int, wa_message_id: str | None, text: str) -> dict:
    if duplicate(wa_message_id):
        return {"reply": None, "status": "duplicate"}
    if wa_message_id is not None:
        SEEN.add(wa_message_id)

    state = state_for(conversation_id)
    batch = state.current
    state.seq += 1
    token = state.seq

    now = asyncio.get_running_loop().time()
    if not batch.texts:
        batch.first_at = now
        batch.wa_message_id = wa_message_id
    batch.texts.append(text)
    batch.last_at = now
    batch.last_token = token

    _ensure_timer(conversation_id, state)
    await batch.event.wait()

    if token == batch.winner:
        return batch.result
    return {"reply": None, "status": "coalesced"}


# --------------------------------------------------------------------------
# Persistence helpers
# --------------------------------------------------------------------------

def save_message(conversation_id: int, role: str, content: str, **columns) -> int:
    with db.get_session() as session:
        message = Message(
            conversation_id=conversation_id, role=role, content=content, **columns
        )
        session.add(message)
        session.commit()
        session.refresh(message)
        return message.id


def trace_for(message_id: int) -> list[dict]:
    with db.get_session() as session:
        rows = session.exec(
            select(ToolCall).where(ToolCall.message_id == message_id).order_by(ToolCall.id)
        ).all()
    return [
        {"tool_name": r.tool_name, "args": r.args, "result": r.result,
         "latency_ms": r.latency_ms}
        for r in rows
    ]


# Enough for the buyer to refer back to something, few enough that the block
# stays a line or two.
SHOWN_LIMIT = 8


def shown_properties(conversation_id: int) -> list[dict]:
    """Every property this conversation has surfaced, most recent search first.

    Rebuilt from `tool_calls` rather than remembered, for the same reason the
    profile is (D12): the transcript is text-only, so nothing the tools returned
    survives into the next turn. Cards are deduplicated on id and the newest
    search wins, because that is the list the buyer is looking at.
    """
    with db.get_session() as session:
        rows = session.exec(
            select(ToolCall)
            .join(Message, ToolCall.message_id == Message.id)
            .where(Message.conversation_id == conversation_id)
            .where(ToolCall.tool_name == "search_properties")
            .order_by(ToolCall.id.desc())
        ).all()

    seen: dict[int, dict] = {}
    for row in rows:
        result = row.result or {}
        cards = (result.get("exact_matches") or []) + (result.get("near_matches") or [])
        for card in cards:
            if card.get("id") is not None and card["id"] not in seen:
                seen[card["id"]] = {
                    "id": card["id"],
                    "title": card.get("title", ""),
                    "locality": card.get("locality", ""),
                }
        if len(seen) >= SHOWN_LIMIT:
            break
    return list(seen.values())[:SHOWN_LIMIT]


def build_messages(conversation_id: int, buyer_id: int) -> list[dict]:
    """System prompt plus the last 20 `user`/`assistant` text rows (D11).

    The dynamic profile block is read straight off `update_buyer_profile`, so
    the prompt and the tool cannot disagree about what is still unknown (D12).
    The shown-properties block is the same idea for property ids.
    """
    system = build_system_prompt(
        update_buyer_profile(buyer_id, {}), buyer_id, conversation_id,
        shown=shown_properties(conversation_id),
    )
    with db.get_session() as session:
        rows = session.exec(
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .where(Message.role.in_(("user", "assistant")))
            .order_by(Message.id.desc())
            .limit(TRANSCRIPT_TURNS)
        ).all()
    return [{"role": "system", "content": system}] + [
        {"role": row.role, "content": row.content} for row in reversed(rows)
    ]


# --------------------------------------------------------------------------
# The turn
# --------------------------------------------------------------------------

def _elapsed_ms(since: float) -> int:
    return int((time.perf_counter() - since) * 1000)


async def _reply(
    conversation_id: int,
    text: str,
    t0: float,
    batch: Batch,
    message_id: int,
    model_used: str | None,
    provider: str | None,
    status: str,
) -> dict:
    """Writes the assistant row and shapes the POST response."""
    loop = asyncio.get_running_loop()
    save_message(
        conversation_id, "assistant", text,
        latency_ms=_elapsed_ms(t0),
        inbound_latency_ms=int((loop.time() - batch.last_at) * 1000),
        model_used=model_used,
        provider=provider,
    )
    return {"reply": text, "tool_calls": trace_for(message_id), "status": status}


async def guard_and_send(conversation_id, response, messages, batch, message_id,
                         model_used, provider):
    """The grounding post-check and its one repair turn (D4).

    Returns `(text, model_used, provider)` — the failover can serve the repair
    from a different rung than the reply it is repairing, and the assistant row
    has to record whichever one actually spoke last.
    """
    sources = guard.sources_text(
        [call["result"] for call in trace_for(message_id)], batch.texts
    )
    text = guard.maybe_force_fail(response.choices[0].message.content or "")
    spans = guard.offending_spans(text, sources)
    if not spans:
        return text, model_used, provider

    guard.log_rejection(conversation_id, text, spans)

    # One repair turn, and never more than one per turn.
    messages = messages + [
        {"role": "assistant", "content": text},
        {"role": "user", "content": guard.repair_prompt(spans, sources)},
    ]
    try:
        repaired, model_used, provider = await llm.llm_call(messages, TOOLS)
    except (llm.LLMUnavailable, openai.OpenAIError):
        return guard.CLARIFYING_QUESTION, model_used, provider

    text = guard.maybe_force_fail(repaired.choices[0].message.content or "")
    if not text.strip():
        return guard.CLARIFYING_QUESTION, model_used, provider

    spans = guard.offending_spans(text, sources)
    if not spans:
        return text, model_used, provider

    guard.log_rejection(conversation_id, text, spans)
    return guard.CLARIFYING_QUESTION, model_used, provider


async def finish_escalated(conversation_id, t0, batch, message_id,
                           model_used, provider, reply=None) -> dict:
    """The escalating turn returns the templated handoff line; after this the
    agent never speaks in this conversation again (D19).

    The brief is written here rather than in the tool, because the same brief
    must be produced on the deterministic path where the tool is never called
    (D17).
    """
    await brief.fill_brief(conversation_id)
    line = escalation.handoff_line("\n".join(batch.texts))
    return await _reply(
        conversation_id, f"{reply}\n\n{line}" if reply else line,
        t0, batch, message_id, model_used, provider, "escalated",
    )


async def escalate(conversation_id, reason, urgency, t0, batch, message_id,
                   model_used=None, provider=None, reply=None) -> dict:
    escalate_to_broker(conversation_id, reason, urgency)
    return await finish_escalated(
        conversation_id, t0, batch, message_id, model_used, provider, reply
    )


async def _run(conversation_id: int, batch: Batch) -> dict:
    with db.get_session() as session:
        convo = session.get(Conversation, conversation_id)
        if convo is None:
            return {"reply": None, "tool_calls": [], "status": "unknown_conversation"}
        status, buyer_id = convo.status, convo.buyer_id

    user_message = "\n".join(batch.texts)

    if status != "active":
        # A human owns this lead now: store the message for the broker inbox and
        # stay silent. Two voices answering one buyer is the fastest way to lose
        # trust.
        save_message(conversation_id, "user", user_message,
                     wa_message_id=batch.wa_message_id)
        return {"reply": None, "tool_calls": [], "status": status}

    message_id = save_message(
        conversation_id, "user", user_message, wa_message_id=batch.wa_message_id
    )
    t0 = time.perf_counter()          # the latency clock starts at drain (D20)

    # The deterministic tier (D18). Not "pre-LLM" — the brief-writer call runs on
    # this path too — but the escalation *decision* is made without a model.
    trigger = triggers.match(user_message)
    if trigger:
        return await escalate(
            conversation_id, trigger, triggers.TRIGGER_URGENCY,
            t0, batch, message_id,
        )

    messages = build_messages(conversation_id, buyer_id)
    model_used = provider = None

    for _ in range(STEP_CAP):
        try:
            response, model_used, provider = await llm.llm_call(messages, TOOLS)
        except (llm.LLMUnavailable, openai.OpenAIError):
            # Infrastructure, not intent: hold the line, stay active, retry on
            # the next inbound (D20).
            return await _reply(conversation_id, HOLD_MESSAGE, t0, batch,
                                message_id, model_used, provider, "active")

        choice = response.choices[0]
        if choice.finish_reason != "tool_calls":
            break
        messages.append(choice.message.model_dump())

        escalated = False
        for call in choice.message.tool_calls:
            result = dispatch(call, message_id)
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": json.dumps(result, ensure_ascii=False),
            })
            if call.function.name == "escalate_to_broker" and "error" not in result:
                escalated = True

        # After the whole batch, never mid-batch: returning early would leave a
        # tool_call unanswered, which the OpenAI format forbids.
        if escalated:
            return await finish_escalated(
                conversation_id, t0, batch, message_id, model_used, provider
            )
    else:
        # The agent genuinely cannot converge; a human should own it.
        return await escalate(conversation_id, "step_cap_breached", "medium",
                              t0, batch, message_id, model_used, provider)

    text, model_used, provider = await guard_and_send(
        conversation_id, response, messages, batch, message_id, model_used, provider
    )

    # The post-turn evaluator (D16). Computing these makes escalation guaranteed
    # rather than hoped for — it catches the handoff the model itself missed.
    trace = trace_for(message_id)
    escalation.update_clarification_count(conversation_id, trace, text)
    verdict = escalation.evaluate(conversation_id, message_id)
    if verdict is not None:
        reason, urgency = verdict
        # The reply is carried through rather than replaced: on this path the
        # model has already said something the buyer needs — "your visit is
        # confirmed" — and swallowing it to send only the handoff line would
        # lose the very thing that triggered the handoff.
        return await escalate(conversation_id, reason, urgency, t0, batch,
                              message_id, model_used, provider, reply=text)

    return await _reply(conversation_id, text, t0, batch, message_id,
                        model_used, provider, "active")


async def run_turn(conversation_id: int) -> dict | None:
    state = state_for(conversation_id)
    async with state.lock:
        batch = state.current
        if not batch.texts:
            return None

        # Anything arriving from here on belongs to the next window.
        state.current = Batch()
        batch.winner = batch.last_token

        try:
            batch.result = await _run(conversation_id, batch)
        finally:
            if batch.result is None:
                batch.result = {"reply": None, "tool_calls": [], "status": "error"}
            batch.event.set()
        return batch.result
