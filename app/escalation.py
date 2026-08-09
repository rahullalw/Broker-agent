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

# Two questions count as "the same question" at this much word overlap. Loose
# enough to catch a rephrase, tight enough that budget->locality is a new ask.
REPEAT_OVERLAP = 0.6

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


def _progressed_this_turn(session, conversation_id: int, message_id: int) -> bool:
    """Did this turn actually do something — search, fetch details, book, write
    the profile — as opposed to only talking?"""
    return any(
        "error" not in (row.result or {})
        for row in _tool_calls(session, conversation_id, message_id)
    )


def _possession_and_price_asked(session, conversation_id: int) -> bool:
    asked: set[str] = set()
    for row in _tool_calls(session, conversation_id):
        if row.tool_name == "get_property_details":
            asked.update(row.args.get("sections") or [])
    return {"pricing", "possession"} <= asked


def _qualification_complete(buyer: Buyer) -> bool:
    """Every field in the prompt's qualification order, and nothing less:
    budget, locality, BHK, possession timeline, family size.

    Taken literally, §9's "budget confirmed and locality locked" fires at turn
    three — the first two steps of the qualification order — and a nurture agent
    that nurtures for ninety seconds is not one. Requiring the whole list lets
    the conversation run as long as it needs to: the agent asks one question per
    message, so this cannot fire before the buyer has actually answered all five,
    however many turns that takes.

    **This is only half the trigger** — `evaluate` also requires the turn to have
    achieved nothing (`_progressed_this_turn`). A complete profile is not itself
    the moment to hand off: the field that completes it is written by the same
    turn that captured it, so escalating on state alone cuts the agent off
    mid-sentence, one turn *before* it searches. The buyer is told "main abhi
    aapke liye options check karta hoon" and then the agent goes silent forever,
    having shown nothing.

    Read as a fallback instead, it composes with the arc rather than truncating
    it. Capture, search, details and booking all count as progress, so the
    handoff lands where it belongs — on `site_visit_booked`, or on the first turn
    where a fully-qualified buyer has nothing left for the agent to do.
    """
    return (
        (buyer.budget_min is not None or buyer.budget_max is not None)
        and bool(buyer.preferred_localities)
        and buyer.bhk_need is not None
        and buyer.possession_need is not None
        and buyer.family_size is not None
    )


def evaluate(conversation_id: int, message_id: int) -> tuple[str, str] | None:
    """`(reason, urgency)` if this turn should hand off, else None."""
    if already_escalated(conversation_id):
        return None            # one conversation, one escalation row

    with db.get_session() as session:
        convo = session.get(Conversation, conversation_id)
        if convo is None:
            return None
        buyer = session.get(Buyer, convo.buyer_id)

        # A booking is the terminal action, so it hands off on its own turn.
        # The other two are *state* checks — conditions that stay true once
        # reached — and firing them the moment they go true cuts the agent off
        # mid-arc, on the very turn it did the work that satisfied them. They
        # wait for a turn that achieved nothing instead. See
        # `_qualification_complete` for the full argument.
        progressed = _progressed_this_turn(session, conversation_id, message_id)

        fired = {
            "site_visit_booked": _booked_this_turn(session, conversation_id, message_id),
            "possession_and_price_asked": (
                _possession_and_price_asked(session, conversation_id)
                and not progressed
            ),
            "qualification_complete": (
                buyer is not None
                and _qualification_complete(buyer)
                and not progressed
            ),
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


def _last_question(text: str) -> str | None:
    """The final question in a reply, normalised to bare words.

    Comparing whole replies is useless — the wrapper sentence changes every turn
    ("Koi baat nahi, ..." / "Zaroor, ...") while the ask underneath stays the
    same. The question is the only part that says whether the agent moved on.
    """
    questions = re.findall(r"[^.!?\n]*\?", text or "")
    if not questions:
        return None
    words = re.findall(r"[a-z0-9]+", questions[-1].lower())
    return " ".join(words) or None


def _same_question(current: str | None, previous: str | None) -> bool:
    """Word-overlap, not equality — a model rephrasing the identical ask is
    still the identical ask ("kya aap budget bata sakte hain?" /
    "aap apna budget bata sakte hain?")."""
    if not current or not previous:
        return False
    a, b = set(current.split()), set(previous.split())
    return len(a & b) / len(a | b) >= REPEAT_OVERLAP


def _previous_assistant_text(session, conversation_id: int) -> str | None:
    """The reply *before* this turn's. Safe to read: the caller runs before
    `_reply` writes the current assistant row."""
    row = session.exec(
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .where(Message.role == "assistant")
        .order_by(Message.id.desc())
    ).first()
    return row.content if row else None


def update_clarification_count(
    conversation_id: int, tool_results: list[dict], reply_text: str
) -> int:
    """Maintain the counter D16's third trigger reads.

    The spec names the trigger but not what makes a clarification "failed", so
    this is the working definition: the agent **asked the same question again**
    and the turn still achieved nothing. Any successful tool call resets it — the
    conversation moved, whatever the reply ended with.

    Requiring the repeat is the whole point. §10 tells the agent to end every
    message with a question, and the opening arc has nothing to write to the
    profile yet: "hi" → asks budget, "mujhe ghar lena hai" → asks budget,
    "budget socha nahi hai" → asks locality. Counting bare question-marks made
    that cooperative three-turn opening indistinguishable from an agent stuck in
    a loop, and escalated it at turn three. A *new* question means the buyer
    answered and the agent advanced; only a repeat is evidence of being stuck.
    """
    progressed = any(_succeeded(result) for result in tool_results)

    with db.get_session() as session:
        convo = session.get(Conversation, conversation_id)
        if convo is None:
            return 0

        if progressed:
            convo.clarification_count = 0
        else:
            asked = _last_question(reply_text)
            previous = _last_question(_previous_assistant_text(session, conversation_id))
            if _same_question(asked, previous):
                convo.clarification_count += 1
            else:
                convo.clarification_count = 0

        session.add(convo)
        session.commit()
        return convo.clarification_count
