"""OpenAI-format tool schemas (Phase 2 §7).

This is the only description of the five tools the model ever sees, so the closed
vocabularies live here as `enum`s (D13) — an enum the agent cannot violate
removes an entire class of failure. Python-side validation in `tools.py` says the
same things again, on purpose: the schema shapes the model's output, the Python
check produces the steering error string when the model ignores it anyway.

There are deliberately no schemas for negotiating a price, quoting a loan,
giving a legal opinion or committing to a possession date. Blocking by
capability beats blocking by instruction.
"""
from __future__ import annotations

from app.enums import INTENT_TIER, LOCALITIES, URGENCY
from app.tools import SECTIONS

# Phase 2 §7: try `strict: true`, but verify it works with Gemini before
# depending on it. It is off until that verification happens — strict mode also
# requires every property to appear in `required`, which would force the model to
# send an explicit null for each unused search filter. `tools.py` validates
# regardless, so nothing downstream depends on this flag.
STRICT = False


def _tool(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "strict": STRICT,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }


_LOCALITY_ARRAY = {
    "type": "array",
    "items": {"type": "string", "enum": list(LOCALITIES)},
}

_MONTH = {
    "type": "string",
    "description": "Target month as YYYY-MM, e.g. 2026-12.",
}


TOOLS = [
    _tool(
        "search_properties",
        "Find up to five properties that fit the buyer. Every argument is "
        "optional, so it can be called on partial qualification. Never returns "
        "nothing: if too few properties match, near-misses come back labelled "
        "with what was relaxed. Results carry ready-to-quote display strings.",
        {
            "budget_min": {
                "type": "integer",
                "description": "Lower bound in whole rupees, e.g. 5500000.",
            },
            "budget_max": {
                "type": "integer",
                "description": "Upper bound in whole rupees, e.g. 6500000.",
            },
            "localities": {
                **_LOCALITY_ARRAY,
                "description": "Localities in the buyer's order of preference.",
            },
            "bhk": {
                "type": "integer",
                "description": "Bedrooms needed. Never relaxed.",
            },
            "possession_before": {
                **_MONTH,
                "description": "Latest acceptable possession month, YYYY-MM.",
            },
        },
        [],
    ),
    _tool(
        "get_property_details",
        "Look up specific sections of one property. Ask only for the sections "
        "the buyer asked about; every fact comes back pre-formatted.",
        {
            "property_id": {
                "type": "integer",
                "description": "Id from a search_properties result.",
            },
            "sections": {
                "type": "array",
                "items": {"type": "string", "enum": list(SECTIONS)},
                "description": "Which sections to return.",
            },
        },
        ["property_id", "sections"],
    ),
    _tool(
        "update_buyer_profile",
        "Record what the buyer has told you and read back what is still "
        "unknown. Call it as soon as a fact is stated. An empty updates object "
        "is a read. The returned unknown list drives the next question.",
        {
            "buyer_id": {"type": "integer", "description": "Id of the buyer."},
            "updates": {
                "type": "object",
                "description": "Only the fields the buyer actually stated.",
                "properties": {
                    "budget_min": {
                        "type": "integer",
                        "description": "Lower bound in whole rupees, e.g. 5500000.",
                    },
                    "budget_max": {
                        "type": "integer",
                        "description": "Upper bound in whole rupees, e.g. 6500000.",
                    },
                    "preferred_localities": {
                        **_LOCALITY_ARRAY,
                        "description": "Localities in the buyer's order of preference.",
                    },
                    "bhk_need": {"type": "integer", "description": "Bedrooms needed."},
                    "possession_need": {
                        **_MONTH,
                        "description": "Month the buyer needs possession by, YYYY-MM.",
                    },
                    "family_size": {
                        "type": "integer",
                        "description": "People who will live in the home.",
                    },
                    "intent_tier": {
                        "type": "string",
                        "enum": list(INTENT_TIER),
                        "description": "How ready to transact the buyer sounds.",
                    },
                },
                "required": [],
                "additionalProperties": False,
            },
        },
        ["buyer_id", "updates"],
    ),
    _tool(
        "book_site_visit",
        "Book a visit to one property. The slot must be an exact date and time "
        "within the next 14 days; vague timings are rejected.",
        {
            "buyer_id": {"type": "integer", "description": "Id of the buyer."},
            "property_id": {
                "type": "integer",
                "description": "Id from a search_properties result.",
            },
            "slot": {
                "type": "string",
                "description": (
                    "Exact local date and time, e.g. 2026-08-09 17:00. Ask the "
                    "buyer for a real date rather than guessing one."
                ),
            },
            "idempotency_key": {
                "type": "string",
                "description": (
                    "Stable key for this booking attempt. Reuse it on a retry so "
                    "the buyer is never booked twice."
                ),
            },
        },
        ["buyer_id", "property_id", "slot", "idempotency_key"],
    ),
    _tool(
        "escalate_to_broker",
        "Hand the conversation to a human broker. Terminal: after this you say "
        "nothing further in this conversation.",
        {
            "conversation_id": {
                "type": "integer",
                "description": "Id of this conversation.",
            },
            "reason": {
                "type": "string",
                "description": "One line on what the broker is inheriting.",
            },
            "urgency": {
                "type": "string",
                "enum": list(URGENCY),
                "description": "How fast the broker needs to pick this up.",
            },
        },
        ["conversation_id", "reason", "urgency"],
    ),
]
