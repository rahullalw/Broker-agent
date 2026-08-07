"""The deterministic evaluator and the handoff (Phase 4 §3 and §5, D15, D16, D19).

Most of §9's "model-judged" tier is not a judgement at all. Computing it after
each turn makes escalation **guaranteed** rather than hoped for.

`intent_tier` is deliberately **not** a trigger — it was circular. The model sets
it, so "escalate when `intent_tier == hot`" reduces to "escalate when the model
decides to", which its own `escalate_to_broker` call already expresses. The field
stays; it is useful context in the brief.
"""
from __future__ import annotations

import re

from sqlmodel import select

from app import db
from app.db import Buyer, Conversation, Escalation, Message, ToolCall

# Strongest signal first, so a turn that trips two always reports the same one.
# `qualification_complete` sits below the two "the buyer is asking now" triggers
# because it is the weakest evidence of intent among them.
PRECEDENCE = (
    ("site_visit_booked", "high"),
    ("possession_and_price_asked", "high"),
    ("qualification_complete", "medium"),
    ("clarification_exhausted", "medium"),
)

CLARIFICATION_LIMIT = 3

HANDOFF_HI = (
    "Main aapko humare broker se connect kar raha hoon, "
    "wo thodi der mein aapse baat karenge."
)
HANDOFF_EN = "I'm connecting you with our broker, they'll reach out shortly."

# A crude check, deliberately. Whole words only, and none of them are also
# English words — "me" and "to" mean different things in the two languages and
# would flip half the English messages.
HINGLISH_MARKERS = re.compile(
    r"\b(hai|hain|haan|nahi|nahin|kya|kyu|kaise|kitna|kitni|kab|kaun|kaunsa|"
    r"chahiye|karo|karao|kardo|kar|karna|karni|karunga|batao|bataye|"
    r"milega|milegi|hoga|hogi|banega|banegi|lagega|lagegi|sakta|sakti|sakte|"
    r"raha|rahi|rahe|hua|gaya|diya|liya|"
    r"bhai|bhaiya|yaar|aap|aapko|mujhe|mera|meri|tak|thoda|thodi|"
    r"kuch|koi|bhi|phir|kam|zyada|jyada|"
    r"acha|accha|theek|sahi|dekh|dekhna|dikhao|chalo|abhi|baat|se|ka|ki|ke|"
    r"me|mein|pe|par|lunga|dena|dedo|wala|wali)\b",
    re.IGNORECASE,
)

# "me", "se", "ka" and friends are ambiguous on their own, so a single marker is
# not enough — two independent hits is what makes it Hinglish rather than an
# English sentence containing the word "me".
HINGLISH_THRESHOLD = 2


def handoff_line(text: str | None) -> str:
    """One templated line, then permanent silence.

    Templated means no LLM call, no latency, no guard risk — it contains zero
    facts — and it is identical on the regex and model paths.
    """
    if not text:
        return HANDOFF_EN
    hits = {match.group(0).lower() for match in HINGLISH_MARKERS.finditer(text)}
    return HANDOFF_HI if len(hits) >= HINGLISH_THRESHOLD else HANDOFF_EN


def already_escalated(conversation_id: int) -> bool:
    with db.get_session() as session:
        convo = session.get(Conversation, conversation_id)
        if convo is not None and convo.status == "escalated":
            return True
        row = session.exec(
            select(Escalation).where(Escalation.conversation_id == conversation_id)
        ).first()
    return row is not None


def _tool_calls(session, conversation_id: int, message_id: int | None = None):
    query = (
        select(ToolCall)
        .join(Message, ToolCall.message_id == Message.id)
        .where(Message.conversation_id == conversation_id)
    )
    if message_id is not None:
        query = query.where(ToolCall.message_id == message_id)
    return session.exec(query).all()


def _booked_this_turn(session, conversation_id: int, message_id: int) -> bool:
    return any(
        row.tool_name == "book_site_visit" and row.result.get("confirmed")
        for row in _tool_calls(session, conversation_id, message_id)
    )


def _possession_and_price_asked(session, conversation_id: int) -> bool:
    asked: set[str] = set()
    for row in _tool_calls(session, conversation_id):
        if row.tool_name == "get_property_details":
            asked.update(row.args.get("sections") or [])
    return {"pricing", "possession"} <= asked


def _qualification_complete(buyer: Buyer) -> bool:
    """Budget **and** locality **and** (BHK **or** possession).

    Taken literally, §9's "budget confirmed and locality locked" fires at turn
    three — the first two steps of the qualification order. Requiring one more
    lets the agent complete a real arc and surface properties before handing off.
    """
    has_budget = buyer.budget_min is not None or buyer.budget_max is not None
    has_locality = bool(buyer.preferred_localities)
    has_shape = buyer.bhk_need is not None or buyer.possession_need is not None
    return has_budget and has_locality and has_shape


def evaluate(conversation_id: int, message_id: int) -> tuple[str, str] | None:
    """`(reason, urgency)` if this turn should hand off, else None."""
    if already_escalated(conversation_id):
        return None            # one conversation, one escalation row

    with db.get_session() as session:
        convo = session.get(Conversation, conversation_id)
        if convo is None:
            return None
        buyer = session.get(Buyer, convo.buyer_id)

        fired = {
            "site_visit_booked": _booked_this_turn(session, conversation_id, message_id),
            "possession_and_price_asked": _possession_and_price_asked(
                session, conversation_id
            ),
            "qualification_complete": buyer is not None and _qualification_complete(buyer),
            "clarification_exhausted": convo.clarification_count >= CLARIFICATION_LIMIT,
        }

    for reason, urgency in PRECEDENCE:
        if fired[reason]:
            return reason, urgency
    return None


def _succeeded(result: dict, name: str | None = None) -> bool:
    if name is not None and result.get("tool_name") != name:
        return False
    return "error" not in (result.get("result") or {})


def update_clarification_count(
    conversation_id: int, tool_results: list[dict], reply_text: str
) -> int:
    """Maintain the counter D16's third trigger reads.

    The spec names the trigger but not what makes a clarification "failed", so
    this is the working definition: the agent asked a question **and the turn
    achieved nothing at all**. Any successful profile write resets it — the buyer
    answered, so the run of failures is over.

    Both halves matter. §10 tells the agent to end every message with a question,
    so `asked_again` alone is true on essentially every turn; counting those
    would escalate any conversation that ran three turns without a profile
    write — including one where the agent searched, quoted possession and read
    back a RERA id. A turn that called a tool successfully advanced the
    conversation, whatever it ended with. It is not a failed clarification.
    """
    captured = any(
        _succeeded(result, "update_buyer_profile") for result in tool_results
    )
    progressed = any(_succeeded(result) for result in tool_results)
    asked_again = (reply_text or "").strip().endswith("?") and not progressed

    with db.get_session() as session:
        convo = session.get(Conversation, conversation_id)
        if convo is None:
            return 0
        if captured:
            convo.clarification_count = 0
        elif asked_again:
            convo.clarification_count += 1
        session.add(convo)
        session.commit()
        return convo.clarification_count
