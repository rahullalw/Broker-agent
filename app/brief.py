"""The escalation brief (Phase 4 §4, D17).

`escalations.brief` had no producer: `escalate_to_broker` takes no `brief`
argument, and the deterministic tier never invokes the model at all. One
mechanism serves both paths.

**Written half** — a dedicated LLM call with its own small prompt, given the
transcript and the trigger. Latency is free: escalation is terminal and the buyer
is getting no further replies. Quota is not free — this is one extra request per
escalation against the 15/min BYOK budget (Phase 3 §1).

**Deterministic half** — a facts block appended server-side. It is not
model-written, so it is not guarded; it is assembled from display strings the
tools already produced.

The guard runs over the written half so the brief cannot quote a price no tool
ever returned. If the writer fails, is unparseable, or is rejected, the prose
falls back to templates — an escalation must never be blocked by its own brief.
"""
from __future__ import annotations

from sqlmodel import select

from app import db, guard, llm
from app.db import Buyer, Conversation, Escalation, Message, Property, SiteVisit, ToolCall
from app.format import slot_display
from app.tools import update_buyer_profile

SECTIONS = ("WANTS", "WHY NOW", "OPEN")
LABEL_WIDTH = 8

WRITER_PROMPT = """\
You write handover notes for property brokers in Ahmedabad. A conversation with
a buyer has just been handed from an AI assistant to a human broker. Write the
note the broker reads before calling them.

Output exactly three lines, each starting with its label and nothing else:

WANTS <what the buyer is looking for, one line>
WHY NOW <what triggered the handover, one line>
OPEN "<the first line the broker should send, in the buyer's own language>"

Quote figures, dates and RERA ids exactly as they appear in the transcript.
State no figure that is not already there. No preamble, no closing remark."""

REASON_PHRASES = {
    "site_visit_booked": "Site visit confirmed.",
    "qualification_complete": "Budget, locality and requirement are all captured.",
    "possession_and_price_asked": "Asked about both possession and pricing.",
    "clarification_exhausted": "Three clarifications in a row went unanswered.",
    "negotiation_request": "Buyer asked to negotiate the price.",
    "loan_advice_request": "Buyer asked for loan advice.",
    "legal_dispute": "Buyer raised a legal or dispute matter.",
    "human_requested": "Buyer asked to speak to a person.",
    "step_cap_breached": "The assistant could not converge and stopped.",
}

UNKNOWN_LABELS = {
    "budget": "budget",
    "preferred_localities": "locality",
    "bhk_need": "BHK",
    "possession_need": "possession timeline",
    "family_size": "family size",
}


# --------------------------------------------------------------------------
# Reading the conversation
# --------------------------------------------------------------------------

def _transcript(session, conversation_id: int) -> list[Message]:
    return session.exec(
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .where(Message.role.in_(("user", "assistant")))
        .order_by(Message.id)
    ).all()


def _tool_results(session, conversation_id: int) -> list[dict]:
    rows = session.exec(
        select(ToolCall)
        .join(Message, ToolCall.message_id == Message.id)
        .where(Message.conversation_id == conversation_id)
        .order_by(ToolCall.id)
    ).all()
    return [row.result for row in rows]


def _visit_line(session, buyer_id: int) -> str | None:
    visit = session.exec(
        select(SiteVisit).where(SiteVisit.buyer_id == buyer_id)
        .order_by(SiteVisit.id.desc())
    ).first()
    if visit is None:
        return None
    prop = session.get(Property, visit.property_id)
    return f"{slot_display(visit.slot)} — {prop.title}, {prop.locality}"


# --------------------------------------------------------------------------
# The written half
# --------------------------------------------------------------------------

def _parse(text: str) -> dict[str, str] | None:
    """Three labelled lines, or nothing. A model that ignored the format is not
    a brief — it is prose the broker has to read twice."""
    found: dict[str, str] = {}
    for line in (text or "").splitlines():
        stripped = line.strip()
        for section in SECTIONS:
            if stripped.upper().startswith(section):
                body = stripped[len(section):].strip(" :\t")
                if body:
                    found[section] = body
    return found if len(found) == len(SECTIONS) else None


def _fallback(profile: dict, reason: str, visit: str | None) -> dict[str, str]:
    """Templated prose, assembled from display strings the tools produced."""
    wants = ", ".join(filter(None, [
        f"{profile['bhk_need']}BHK" if profile.get("bhk_need") else None,
        f"in {'/'.join(profile['localities'])}" if profile.get("localities") else None,
        profile.get("budget"),
        f"possession by {profile['possession_need']}" if profile.get("possession_need") else None,
    ]))
    why = REASON_PHRASES.get(reason, f"Handed over on {reason}.")
    if visit and reason == "site_visit_booked":
        why = f"{why[:-1]} — {visit}."
    return {
        "WANTS": wants or "Not yet qualified — see the transcript.",
        "WHY NOW": why,
        "OPEN": '"Hi, this is your AllSet broker — I have your requirement, '
                'let me take it from here."',
    }


async def _write_prose(
    conversation_id: int, transcript: list[Message], reason: str, urgency: str
) -> str | None:
    lines = "\n".join(f"{m.role}: {m.content}" for m in transcript)
    messages = [
        {"role": "system", "content": WRITER_PROMPT},
        {"role": "user", "content":
            f"Trigger: {reason} (urgency: {urgency})\n\nTranscript:\n{lines}"},
    ]
    try:
        response, _, _ = await llm.llm_call(messages)
    except Exception:
        # An escalation must never be blocked by its own brief.
        return None
    return response.choices[0].message.content


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------

def _row(label: str, value: str) -> str:
    return f"{label:<{LABEL_WIDTH}}{value}"


async def fill_brief(conversation_id: int) -> str | None:
    """Write the brief for this conversation's escalation and store it.

    Returns None if the conversation was never escalated. Re-running is a no-op:
    the broker's brief does not change under them.
    """
    with db.get_session() as session:
        row = session.exec(
            select(Escalation).where(Escalation.conversation_id == conversation_id)
        ).first()
        if row is None:
            return None
        if row.brief:
            return row.brief

        convo = session.get(Conversation, conversation_id)
        buyer = session.get(Buyer, convo.buyer_id)
        transcript = _transcript(session, conversation_id)
        tool_results = _tool_results(session, conversation_id)
        buyer_messages = [m.content for m in transcript if m.role == "user"]
        visit = _visit_line(session, buyer.id)
        reason, urgency = row.reason, row.urgency
        buyer_name, buyer_phone, intent = buyer.name, buyer.phone, buyer.intent_tier

    snapshot = update_buyer_profile(buyer.id, {})
    profile = snapshot.get("profile") or {}
    unknown = snapshot.get("unknown") or []

    prose = _parse(await _write_prose(conversation_id, transcript, reason, urgency) or "")

    if prose is not None:
        # The guard runs over the written half only. Same allowed sources as a
        # turn (D5), widened to the whole conversation because that is the span
        # the brief covers.
        sources = guard.sources_text(tool_results, buyer_messages)
        offending = guard.offending_spans("\n".join(prose.values()), sources)
        if offending:
            guard.log_rejection(
                conversation_id, "\n".join(prose.values()), offending
            )
            prose = None

    if prose is None:
        prose = _fallback(profile, reason, visit)

    known = " · ".join(filter(None, [
        profile.get("budget"),
        "/".join(profile.get("localities", [])) or None,
        f"{profile['bhk_need']}BHK" if profile.get("bhk_need") else None,
        f"possession by {profile['possession_need']}" if profile.get("possession_need") else None,
        f"family of {profile['family_size']}" if profile.get("family_size") else None,
    ])) or "nothing captured"

    body = [
        f"{buyer_name} · {buyer_phone} · intent: {intent}",
        "",
        _row("WANTS", prose["WANTS"]),
        _row("WHY NOW", prose["WHY NOW"]),
        _row("OPEN", prose["OPEN"]),
        "",
        _row("KNOWN", known),
        _row("UNKNOWN", ", ".join(UNKNOWN_LABELS.get(f, f) for f in unknown) or "nothing"),
        _row("TRIGGER", f"{reason} (urgency: {urgency})"),
    ]
    if visit:
        body.append(_row("VISIT", visit))
    body.append(_row("LAST 3", ""))
    body += [f"{' ' * LABEL_WIDTH}{m.role}: {m.content}" for m in transcript[-3:]]

    text = "\n".join(body)

    with db.get_session() as session:
        row = session.exec(
            select(Escalation).where(Escalation.conversation_id == conversation_id)
        ).first()
        row.brief = text
        session.add(row)
        session.commit()
    return text
