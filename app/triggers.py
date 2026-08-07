"""The deterministic escalation tier (Phase 4 §2, D18).

**Wording correction to §9:** this tier is not "pre-LLM" — the brief-writer call
runs on this path too. What it guarantees, and what actually matters, is that the
escalation *decision* is deterministic. No model, no temperature, no drift.

Every pattern fires on a **request for action or advice**, never on a mention.
That distinction is the whole design. `discount` appears in a buyer reporting a
discount someone else gave them; `rate` appears in `accurate` and `corporate`;
`kam` appears in `kam se kam`. A false positive permanently silences the agent
for that buyer, so the patterns are shaped around the ask, not the noun, and
every one of them is anchored with word boundaries.

Runs on the **coalesced** buffer text, not per fragment — a buyer who types
"bhai" and "price kam karo" as two messages is making one request.
"""
from __future__ import annotations

import re

CATEGORIES = (
    "negotiation_request",
    "loan_advice_request",
    "legal_dispute",
    "human_requested",
)

# The buyer is asking for something only a person can give. There is no useful
# gradation here, so the tier does not pretend to compute one.
TRIGGER_URGENCY = "high"

# Hinglish verb tails that turn a noun into an ask. `diya`/`tha` are absent on
# purpose: those are what reporting a past discount looks like.
_ASKING = r"(milega|milegi|mil\s+sakta|hoga|hogi|ho\s+sakta|de\s?do|dedo|dena|do|karo|kar\s?do|kardo|chahiye|possible|batao)"

PATTERNS: dict[str, tuple[str, ...]] = {
    "negotiation_request": (
        r"\bprice\s+kam\b",
        r"\brate\s+kam\b",
        r"\b(kuch|thoda|thodi|zyada|aur)\s+kam\s+(hoga|hogi|ho\s+sakta|karo|kar\s?do|kardo)\b",
        r"\bbest\s+rate\b",
        r"\bmol\s*bhav\b",
        r"\bfinal\s+price\b",
        r"\bnegotiat(e|ed|ion|ing|iable)\b",
        # A discount only counts when it is being asked for.
        r"\b(koi|kuch|thoda|thodi|any|some)\s+discount\b",
        r"\bdiscount\s+" + _ASKING + r"\b",
        r"\bdiscount\s*\?",
    ),
    "loan_advice_request": (
        # Not the word `loan` alone: "I'll take a home loan, budget 65L" is a
        # normal qualification message, and escalating it hands the broker half
        # of all buyers for nothing.
        r"\bemi\s+(kitna|kitni|kya|kaise|kaisa|batao|hoga|hogi|banega|bane|calculate)\b",
        r"\b(kitna|kitni|kya)\s+emi\b",
        r"\bloan\s+(dila|dilwa|dilva)\w*\b",
        r"\bloan\s+(approve|approval|milega|milegi|process|kaise|kitna|dega|degi)\b",
        r"\binterest\s+rate\b",
        r"\b(kaunsa|kaun\s?sa|konsa|which)\s+bank\b",
    ),
    "legal_dispute": (
        r"\bdhokha?\b",
        r"\blegal\s+notice\b",
        r"\b(case|complaint|shikayat)\s+(kar|file|daal|dal)\w*\b",
        r"\bfile\s+a\s+case\b",
        r"\brera\s+(complaint|shikayat)\b",
    ),
    "human_requested": (
        r"\b(talk|speak|connect)\s+to\s+(a\s+|the\s+)?(real\s+)?(person|human|agent|someone|somebody|broker|manager)\b",
        r"\b(aadmi|admi|insaan|banda|koi)\s+se\s+baat\b",
        r"\bmanager\s+(se|ko|chahiye|bulao|call)\b",
    ),
}

_COMPILED: dict[str, tuple[re.Pattern, ...]] = {
    category: tuple(re.compile(p, re.IGNORECASE) for p in patterns)
    for category, patterns in PATTERNS.items()
}


def compiled_patterns() -> list[re.Pattern]:
    return [p for group in _COMPILED.values() for p in group]


def match(text: str) -> str | None:
    """The category the buyer's message falls into, or None.

    Categories are checked in `CATEGORIES` order so a message that trips two
    always reports the same one.
    """
    if not text:
        return None
    for category in CATEGORIES:
        for pattern in _COMPILED[category]:
            if pattern.search(text):
                return category
    return None
