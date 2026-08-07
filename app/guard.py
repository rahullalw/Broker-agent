"""The grounding post-check (Phase 4 §1, D4–D7).

> Any price, area, possession date, RERA ID, or approval status in an outgoing
> message must appear in an allowed source from this same turn.

This is the single most valuable thing in the codebase and it is a byte-exact
substring match. It can be that simple only because tools emit idiomatic display
strings and the prompt tells the model to quote them verbatim — no
canonicalisation layer, no false positives on `₹65 lakh` vs `6500000`.

**Allowed sources (D5):** this turn's `tool_calls.result` rows, and this turn's
buyer message. The second exists because the most natural message in the whole
qualification flow calls no tools at all:

    Buyer: "budget 65 lakh, Bopal me 3BHK chahiye"
    Agent: "Samajh gaya — 65 lakh budget, Bopal, 3BHK. Possession kab tak chahiye?"

Echoing a buyer's own stated figure is confirmation, not hallucination.

**Not allowed:** the stored buyer profile. It is agent-written, so admitting it
would let a value the model got wrong three turns ago certify itself. D12 handles
that instead by having the agent *pass* stored values as tool arguments rather
than restate them, and `search_properties`' `fit_reason` hands the number back
guard-legally.
"""
from __future__ import annotations

import json
import re

from app import config, db
from app.db import GuardRejection
from app.format import MONTHS

# D6 — §8's original "₹ amounts, dates, RERA patterns" is too narrow. It misses
# exactly the numbers a helpful model volunteers.
PATTERNS = (
    re.compile(r"₹\s?[\d,.]+"),                                    # ₹65, ₹1.25
    re.compile(r"[\d,.]+\s*(crore|lakh|lac|Cr|L)\b", re.IGNORECASE),  # bare 65 lakh, 1.2 Cr
    re.compile(r"[\d.]+\s*%"),                                     # a computed 5%
    re.compile(r"[\d,]+\s*(sq\.?\s?ft|sqft)", re.IGNORECASE),      # 1,240 sq ft
    re.compile(rf"\b({'|'.join(MONTHS)})\s+\d{{4}}\b"),            # December 2026
    re.compile(r"(PR/GJ/|GJ/RERA/)[A-Z0-9/ ]+"),                   # RERA ids
)

# Appended post-generation so the guard can be triggered on cue. You cannot make
# a model hallucinate to order in front of an audience.
FORCED_SUFFIX = " Price is ₹72 lakh, possession March 2027."

# The last resort when even the repair turn fails. Carries no facts of its own,
# so it can never itself be rejected.
CLARIFYING_QUESTION = (
    "Sorry, I want to get that exactly right — could you tell me again what "
    "you'd like to know?"
)


def detect(text: str) -> list[str]:
    """Every guarded span in `text`, in order of appearance, deduplicated.

    Overlapping and touching matches are merged first. `₹86 lakh` trips two
    patterns — `₹86` and `86 lakh` — and checking those halves separately would
    let a reply pass whose halves appear in different tool results. Merging is
    strictly stronger (a merged span present in the sources implies every part
    is), and it is what lets the repair turn show the model `₹86 lakh` rather
    than two fragments of it.
    """
    text = text or ""
    ranges = sorted(
        (found.start(), found.end())
        for pattern in PATTERNS
        for found in pattern.finditer(text)
    )

    merged: list[list[int]] = []
    for start, end in ranges:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])

    spans = [text[start:end].strip() for start, end in merged]
    return list(dict.fromkeys(spans))


def sources_text(tool_results: list[dict], buyer_messages: list[str]) -> str:
    """The concatenated allowed sources for this turn. The stored profile is
    deliberately not among them."""
    parts = [json.dumps(result, ensure_ascii=False) for result in tool_results]
    parts += [message for message in buyer_messages if message]
    return "\n".join(parts)


def offending_spans(text: str, sources: str) -> list[str]:
    """Spans that do not appear byte-for-byte in the allowed sources."""
    return [span for span in detect(text) if span not in sources]


def maybe_force_fail(text: str) -> str:
    # Read at call time so the demo can flip it without a restart.
    return text + FORCED_SUFFIX if config.FORCE_GUARD_FAIL else text


def _display_strings(sources: str) -> list[str]:
    """The quotable facts hiding in the raw sources, for the repair prompt. The
    model gets shown the strings, not the JSON it has to dig them out of."""
    return detect(sources)


def repair_prompt(spans: list[str], sources: str) -> str:
    """One repair turn (D4). Shows what was written, what the tool returned, and
    asks for the same reply with the exact strings."""
    wrote = ", ".join(f"`{span}`" for span in spans)
    returned = ", ".join(f"`{span}`" for span in _display_strings(sources)) or "nothing"
    return (
        f"You wrote {wrote}. The tools returned {returned}. "
        f"Re-send your reply quoting the tool's exact strings verbatim — same "
        f"symbol, same unit, same spelling — and state no other figures."
    )


def log_rejection(conversation_id: int, rejected_text: str, spans: list[str]) -> None:
    """Log the full rejected output. It is the prompt-debugging goldmine and the
    demo artifact, and it lives here rather than on `messages` because these
    rows were never sent to anyone."""
    with db.get_session() as session:
        session.add(GuardRejection(
            conversation_id=conversation_id,
            rejected_text=rejected_text,
            offending_spans=list(spans),
        ))
        session.commit()
