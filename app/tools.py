"""The five tools, as plain Python functions (Phase 2).

Three rules shape every function here.

**Compact JSON, never raw rows.** Five results, not fifty. The model reads these
results as prose; a fifty-row dump is a fifty-row hallucination surface.

**Facts ship pre-formatted (D4).** Every buyer-facing value is a display string
from `format.py`. That is what lets the Phase 4 guard be a byte-exact substring
match instead of a canonicalisation layer — the model quotes what the tool said,
character for character, because there is nothing else to quote.

**Errors steer, they never raise.** A tool that throws hands the loop a crash
where the model needed a sentence telling it what to send instead. Every failure
returns `{"error": "..."}` naming the valid alternatives.

Tools that deliberately do not exist: price negotiation, loan advice, legal
opinion, possession commitment. Blocking by capability beats blocking by
instruction.
"""
from __future__ import annotations

from datetime import datetime, timedelta

from sqlmodel import select

from app import db
from app.db import Buyer, Conversation, Escalation, Property, SiteVisit
from app.enums import ADJACENT, INTENT_TIER, LOCALITIES, URGENCY
from app.format import (
    amenities_display,
    area_display,
    budget_display,
    month_display,
    possession_display,
    price_display,
    slot_display,
    status_display,
)

SECTIONS = ("pricing", "possession", "legal", "amenities")

# The Phase 2 §3 example, fixed prose rather than a computed or inferred legal
# status. Every seeded property carries a real-format RERA id under an AUDA
# layout, so one committed sentence is the honest note for all of them — the
# alternative is the agent deriving legal standing, which it must never do.
APPROVAL_NOTE = "RERA registered. AUDA approved layout."

# D14 relaxation policy, in order. BHK is absent on purpose: a 3BHK buyer is
# never shown 2BHKs.
BUDGET_RELAXATION = 125  # percent

MIN_RESULTS = 3
MAX_RESULTS = 5

# §10's qualification order — budget -> locality -> BHK -> possession -> family
# size. `unknown` and `next_question_hint` both read off this tuple, so the
# agent's next question and the prompt's dynamic block cannot drift apart.
QUALIFICATION_ORDER = (
    "budget", "preferred_localities", "bhk_need", "possession_need", "family_size",
)


# --------------------------------------------------------------------------
# Steering errors
# --------------------------------------------------------------------------

def _error(message: str) -> dict:
    return {"error": message}


def _join(items) -> str:
    """['Shela', 'Bopal'] -> 'Shela and Bopal' — for prose, not for enums."""
    items = list(items)
    if len(items) <= 1:
        return "".join(items)
    return f"{', '.join(items[:-1])} and {items[-1]}"


def _bad_locality(value: str) -> dict:
    return _error(
        f"No match for locality {value!r}. Valid: {', '.join(LOCALITIES)}."
    )


def _parse_month(value: str) -> datetime:
    """'2026-12' -> datetime(2026, 12, 1). Raises ValueError to be steered on."""
    year, _, month = value.partition("-")
    return datetime(int(year), int(month), 1)


def _bad_month(field: str, value) -> dict:
    return _error(
        f"{value!r} is not a month for {field}. Send YYYY-MM, e.g. 2026-12."
    )


def _as_int(value):
    """Models emit 6500000.0 as readily as 6500000. Accept both, reject prose."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


# --------------------------------------------------------------------------
# search_properties (D14)
# --------------------------------------------------------------------------

def _month_ceiling(month: datetime) -> datetime:
    """First instant after `month`, so 'possession_before' includes that month."""
    if month.month == 12:
        return datetime(month.year + 1, 1, 1)
    return datetime(month.year, month.month + 1, 1)


def _in_budget(price: int, minimum: int | None, maximum: int | None) -> bool:
    if minimum is not None and price < minimum:
        return False
    if maximum is not None and price > maximum:
        return False
    return True


def _fit_reason(prop: Property, budget_max: int | None, before: datetime | None) -> str:
    """Server-templated, never model-written.

    Useful side effect: the numbers in here land in `tool_calls.result`, so the
    agent may quote them and the guard will agree (D5).
    """
    parts = []

    if budget_max is not None and prop.price_inr != budget_max:
        gap = budget_max - prop.price_inr
        side = "under" if gap > 0 else "over"
        parts.append(f"{price_display(abs(gap))} {side} your budget")

    parts.append(f"in {prop.locality}")

    # Possession earns a line only when the buyer asked about it — otherwise
    # every card carries a date nobody enquired about.
    if before is not None:
        if prop.status == "ready_to_move":
            parts.append("ready to move")
        else:
            parts.append(f"possession {month_display(prop.possession_date)}")

    return " · ".join(parts)


def _card(prop: Property, budget_max: int | None, before: datetime | None) -> dict:
    return {
        "id": prop.id,
        "title": prop.title,
        "locality": prop.locality,
        "bhk": prop.bhk,
        "price_display": price_display(prop.price_inr),
        "possession_display": possession_display(prop.possession_date, prop.status),
        "fit_reason": _fit_reason(prop, budget_max, before),
    }


def _matches(
    rows: list[Property],
    budget_min: int | None,
    budget_max: int | None,
    localities: list[str] | None,
    bhk: int | None,
    before: datetime | None,
) -> list[Property]:
    return [
        p for p in rows
        if (bhk is None or p.bhk == bhk)
        and (not localities or p.locality in localities)
        and _in_budget(p.price_inr, budget_min, budget_max)
        and (before is None or p.possession_date < _month_ceiling(before))
    ]


def _rank(
    rows: list[Property],
    budget_min: int | None,
    budget_max: int | None,
    localities: list[str] | None,
    before: datetime | None,
) -> list[Property]:
    """Deterministic (D14). Reproducibility matters more than cleverness, and
    `id` is the final tie-break always, so the same query returns the same cards."""
    stated = list(localities or ())

    def key(p: Property):
        return (
            0 if _in_budget(p.price_inr, budget_min, budget_max) else 1,
            abs(p.price_inr - budget_max) if budget_max is not None else 0,
            stated.index(p.locality) if p.locality in stated else len(stated),
            0 if before is None or p.possession_date < _month_ceiling(before) else 1,
            p.id,
        )

    return sorted(rows, key=key)


def _relaxation_note(
    rows: list[Property],
    exact: list[Property],
    bhk: int | None,
    localities: list[str] | None,
    budget_min: int | None,
    budget_max: int | None,
    before: datetime | None,
) -> str:
    """Explains the gap in the buyer's own terms, from the real inventory."""
    criteria = f"{bhk}BHK" if bhk else "properties"
    if localities:
        criteria += f" in {_join(localities)}"
    if budget_min is not None and budget_max is not None:
        criteria += f" in the {budget_display(budget_min, budget_max)} range"
    elif budget_max is not None:
        criteria += f" under {price_display(budget_max)}"
    elif budget_min is not None:
        criteria += f" over {price_display(budget_min)}"
    if before is not None:
        criteria += f" with possession by {month_display(before)}"

    if exact:
        plural = "" if len(exact) == 1 else "es"
        headline = f"Only {len(exact)} exact match{plural} for {criteria}."
    else:
        headline = f"No {criteria}."

    clauses = []
    if localities and budget_max is not None:
        # Nothing in the stated localities fits the budget: name the real floor.
        here = _matches(rows, None, None, localities, bhk, before)
        if here and not _matches(rows, budget_min, budget_max, localities, bhk, before):
            floor = min(p.price_inr for p in here)
            clauses.append(f"Closest in {_join(localities)} starts at {price_display(floor)}")

    if budget_max is not None:
        elsewhere = [
            locality for locality in LOCALITIES
            if locality not in (localities or ())
            and _matches(rows, budget_min, budget_max, [locality], bhk, before)
        ]
        if elsewhere:
            verb = "has" if len(elsewhere) == 1 else "have"
            size = f"{bhk}BHK" if bhk else "matching"
            clauses.append(
                f"within {price_display(budget_max)}, {_join(elsewhere)} {verb} "
                f"{size} options"
            )

    if not clauses:
        return headline
    sentence = "; ".join(clauses)
    return f"{headline} {sentence[0].upper()}{sentence[1:]}."


def search_properties(
    budget_min: int | None = None,
    budget_max: int | None = None,
    localities: list[str] | None = None,
    bhk: int | None = None,
    possession_before: str | None = None,
) -> dict:
    """≤5 compact cards. Never empty: near-misses come back labelled (D14).

    Every argument is optional — the agent must be able to search on partial
    qualification, because that is the only kind it has for the first few turns.
    """
    with db.get_session() as session:
        rows = list(session.exec(select(Property)).all())

        if budget_min is not None and (budget_min := _as_int(budget_min)) is None:
            return _error("budget_min must be a whole number of rupees, e.g. 5500000.")
        if budget_max is not None and (budget_max := _as_int(budget_max)) is None:
            return _error("budget_max must be a whole number of rupees, e.g. 6500000.")
        if budget_min is not None and budget_max is not None and budget_min > budget_max:
            return _error(
                f"budget_min {price_display(budget_min)} is above budget_max "
                f"{price_display(budget_max)}. Send the lower figure as budget_min."
            )

        if localities is not None:
            if isinstance(localities, str):
                localities = [localities]
            for locality in localities:
                if locality not in LOCALITIES:
                    return _bad_locality(locality)

        if bhk is not None:
            available = sorted({p.bhk for p in rows})
            if (bhk := _as_int(bhk)) is None or bhk not in available:
                return _error(
                    f"No inventory with bhk={bhk}. Valid: "
                    f"{', '.join(str(b) for b in available)}."
                )

        before = None
        if possession_before is not None:
            try:
                before = _parse_month(possession_before)
            except (ValueError, TypeError):
                return _bad_month("possession_before", possession_before)

        exact = _rank(
            _matches(rows, budget_min, budget_max, localities, bhk, before),
            budget_min, budget_max, localities, before,
        )[:MAX_RESULTS]

        near: list[Property] = []
        relaxations: list[str] = []
        if len(exact) < MIN_RESULTS:
            # Fixed policy: budget first, locality second, BHK never.
            widened_max = budget_max
            widened_localities = list(localities or ())

            for rung in ("budget", "locality"):
                if len(exact) + len(near) >= MIN_RESULTS:
                    break
                if rung == "budget":
                    if budget_max is None:
                        continue
                    widened_max = budget_max * BUDGET_RELAXATION // 100
                    relaxations.append(f"budget raised to {price_display(widened_max)}")
                else:
                    added = [
                        neighbour
                        for locality in (localities or ())
                        for neighbour in ADJACENT[locality]
                        if neighbour not in widened_localities
                    ]
                    if not added:
                        continue
                    widened_localities += added
                    relaxations.append(f"localities widened to {_join(added)}")

                found = _matches(
                    rows, budget_min, widened_max, widened_localities, bhk, before
                )
                near = _rank(
                    [p for p in found if p not in exact],
                    budget_min, budget_max, localities, before,
                )[:MIN_RESULTS - len(exact)]

        return {
            "exact_matches": [_card(p, budget_max, before) for p in exact],
            "near_matches": [_card(p, budget_max, before) for p in near],
            "relaxed": "; ".join(relaxations) if relaxations else None,
            "note": (
                _relaxation_note(
                    rows, exact, bhk, localities, budget_min, budget_max, before
                )
                if relaxations else None
            ),
        }


# --------------------------------------------------------------------------
# get_property_details
# --------------------------------------------------------------------------

def get_property_details(property_id: int, sections: list[str]) -> dict:
    """Only the sections that were asked for — never the whole row."""
    if isinstance(sections, str):
        sections = [sections]
    if not sections:
        return _error(
            f"sections cannot be empty. Ask for one or more of: {', '.join(SECTIONS)}."
        )
    for section in sections:
        if section not in SECTIONS:
            return _error(
                f"No match for section {section!r}. Valid: {', '.join(SECTIONS)}."
            )

    with db.get_session() as session:
        prop = session.get(Property, property_id)
        if prop is None:
            return _error(
                f"No property with id {property_id}. Run search_properties first "
                f"and use an id from its results."
            )

        # id and title identify the row so the agent can name it in the reply;
        # neither is a guarded fact.
        result = {"id": prop.id, "title": prop.title}

        if "pricing" in sections:
            result["pricing"] = {
                "price_display": price_display(prop.price_inr),
                "carpet_area_display": area_display(prop.carpet_area_sqft),
                "tower": prop.tower,
                "floor": prop.floor,
            }
        if "possession" in sections:
            result["possession"] = {
                "possession_display": possession_display(prop.possession_date, prop.status),
                "status_display": status_display(prop.status),
            }
        if "legal" in sections:
            result["legal"] = {
                "rera_id": prop.rera_id,          # verbatim, always
                "developer": prop.developer,
                "approval_note": APPROVAL_NOTE,
            }
        if "amenities" in sections:
            result["amenities"] = {"amenities": amenities_display(prop.amenities)}

        return result


# --------------------------------------------------------------------------
# update_buyer_profile (D13)
# --------------------------------------------------------------------------

def _validated_updates(updates: dict) -> tuple[dict, dict | None]:
    """Returns (column values, steering error). Nothing is written on error —
    a half-applied profile is worse than no profile."""
    columns: dict = {}

    for field, value in updates.items():
        if field in ("budget_min", "budget_max", "bhk_need", "family_size"):
            number = _as_int(value)
            if number is None or number <= 0:
                return {}, _error(
                    f"{field} must be a positive whole number, got {value!r}."
                )
            columns[field] = number

        elif field == "preferred_localities":
            values = [value] if isinstance(value, str) else value
            if not values:
                return {}, _error(
                    f"preferred_localities cannot be empty. Valid: {', '.join(LOCALITIES)}."
                )
            for locality in values:
                if locality not in LOCALITIES:
                    return {}, _bad_locality(locality)
            columns[field] = list(values)

        elif field == "possession_need":
            try:
                columns[field] = _parse_month(value)
            except (ValueError, TypeError, AttributeError):
                return {}, _bad_month("possession_need", value)

        elif field == "intent_tier":
            if value not in INTENT_TIER:
                return {}, _error(
                    f"No match for intent_tier {value!r}. Valid: {', '.join(INTENT_TIER)}."
                )
            columns[field] = value

        else:
            return {}, _error(
                f"No match for field {field!r}. Valid: budget_min, budget_max, "
                f"preferred_localities, bhk_need, possession_need, family_size, "
                f"intent_tier."
            )

    return columns, None


def _profile_view(buyer: Buyer) -> dict:
    """Only what is known, all of it pre-formatted (D4)."""
    profile: dict = {}
    budget = budget_display(buyer.budget_min, buyer.budget_max)
    if budget is not None:
        profile["budget"] = budget
    if buyer.preferred_localities:
        profile["localities"] = list(buyer.preferred_localities)
    if buyer.bhk_need is not None:
        profile["bhk_need"] = buyer.bhk_need
    if buyer.possession_need is not None:
        profile["possession_need"] = month_display(buyer.possession_need)
    if buyer.family_size is not None:
        profile["family_size"] = buyer.family_size
    profile["intent_tier"] = buyer.intent_tier
    return profile


def _unknown_fields(buyer: Buyer) -> list[str]:
    known = {
        "budget": buyer.budget_min is not None or buyer.budget_max is not None,
        "preferred_localities": bool(buyer.preferred_localities),
        "bhk_need": buyer.bhk_need is not None,
        "possession_need": buyer.possession_need is not None,
        "family_size": buyer.family_size is not None,
    }
    return [field for field in QUALIFICATION_ORDER if not known[field]]


def update_buyer_profile(buyer_id: int, updates: dict) -> dict:
    """The updated profile **and** the explicit still-unknown list.

    The unknown list is what makes the agent ask good questions instead of
    inventing them. An empty `updates` object is legal and acts as a read.
    """
    if updates is None:
        updates = {}
    if not isinstance(updates, dict):
        return _error("updates must be an object, e.g. {\"budget_max\": 6500000}.")

    columns, error = _validated_updates(updates)
    if error is not None:
        return error

    with db.get_session() as session:
        buyer = session.get(Buyer, buyer_id)
        if buyer is None:
            return _error(f"No buyer with id {buyer_id}.")

        merged_min = columns.get("budget_min", buyer.budget_min)
        merged_max = columns.get("budget_max", buyer.budget_max)
        if merged_min is not None and merged_max is not None and merged_min > merged_max:
            return _error(
                f"budget_min {price_display(merged_min)} is above budget_max "
                f"{price_display(merged_max)}. Send the lower figure as budget_min."
            )

        for field, value in columns.items():
            setattr(buyer, field, value)
        session.add(buyer)
        session.commit()
        session.refresh(buyer)

        unknown = _unknown_fields(buyer)
        return {
            "profile": _profile_view(buyer),
            "unknown": unknown,
            "next_question_hint": unknown[0] if unknown else None,
        }


# --------------------------------------------------------------------------
# book_site_visit (D15, D24)
# --------------------------------------------------------------------------

BOOKING_WINDOW = timedelta(days=14)


def _confirmation(visit: SiteVisit, prop: Property) -> dict:
    return {
        "confirmed": True,
        "slot_display": slot_display(visit.slot),
        "property": f"{prop.title}, {prop.locality}",
    }


def book_site_visit(
    buyer_id: int, property_id: int, slot: str, idempotency_key: str
) -> dict:
    """A successful booking is an escalation trigger (D15) — but not here. The
    tool books; Phase 4's evaluator picks it up post-turn."""
    with db.get_session() as session:
        buyer = session.get(Buyer, buyer_id)
        if buyer is None:
            return _error(f"No buyer with id {buyer_id}.")
        prop = session.get(Property, property_id)
        if prop is None:
            return _error(
                f"No property with id {property_id}. Run search_properties first "
                f"and use an id from its results."
            )

        try:
            when = datetime.fromisoformat(str(slot))
        except ValueError:
            return _error(
                f"{slot!r} is not a date. Send an exact date and time, "
                f"e.g. 2026-08-09 17:00."
            )

        # D24: naive IST throughout. Strip whatever offset the model attached and
        # keep the wall clock — converting it would move a 5pm visit to 11:30am.
        when = when.replace(tzinfo=None)

        now = datetime.now()
        if when <= now:
            return _error(
                f"{slot_display(when)} is in the past. Send a date and time in the "
                f"future, within the next 14 days."
            )
        if when > now + BOOKING_WINDOW:
            return _error(
                f"{slot_display(when)} is more than 14 days away. Site visits are "
                f"booked within 14 days — send a nearer date."
            )

        existing = session.exec(
            select(SiteVisit).where(SiteVisit.idempotency_key == idempotency_key)
        ).first()
        if existing is not None:
            # A repeat returns the existing booking unchanged, never a second row.
            return _confirmation(existing, session.get(Property, existing.property_id))

        visit = SiteVisit(
            buyer_id=buyer_id, property_id=property_id,
            slot=when, idempotency_key=idempotency_key,
        )
        session.add(visit)
        session.commit()
        session.refresh(visit)
        return _confirmation(visit, prop)


# --------------------------------------------------------------------------
# escalate_to_broker
# --------------------------------------------------------------------------

def escalate_to_broker(conversation_id: int, reason: str, urgency: str) -> dict:
    """Terminal — ends the loop.

    The `brief` is written empty on purpose. Phase 4's brief-writer fills it,
    because the same brief must be produced on the regex path where this tool is
    never called (D17).
    """
    if urgency not in URGENCY:
        return _error(
            f"No match for urgency {urgency!r}. Valid: {', '.join(URGENCY)}."
        )
    if not reason or not str(reason).strip():
        return _error("reason cannot be empty. Say in one line what the broker inherits.")

    with db.get_session() as session:
        convo = session.get(Conversation, conversation_id)
        if convo is None:
            return _error(f"No conversation with id {conversation_id}.")

        # One conversation, one escalation row (Phase 4 §3). Both authorities can
        # fire in the same turn — the model's own call and the post-turn
        # evaluator — and the broker must not inherit the same lead twice. The
        # first reason wins: it is the one the brief was written against.
        existing = session.exec(
            select(Escalation).where(Escalation.conversation_id == conversation_id)
        ).first()
        if existing is not None:
            return {
                "escalated": True,
                "escalation_id": existing.id,
                "reason": existing.reason,
                "urgency": existing.urgency,
            }

        escalation = Escalation(
            conversation_id=conversation_id,
            reason=str(reason).strip(),
            urgency=urgency,
            brief="",
        )
        session.add(escalation)
        convo.status = "escalated"
        session.add(convo)
        session.commit()
        session.refresh(escalation)

        return {
            "escalated": True,
            "escalation_id": escalation.id,
            "reason": escalation.reason,
            "urgency": escalation.urgency,
        }


TOOL_MAPPING = {
    "search_properties": search_properties,
    "get_property_details": get_property_details,
    "update_buyer_profile": update_buyer_profile,
    "book_site_visit": book_site_visit,
    "escalate_to_broker": escalate_to_broker,
}
