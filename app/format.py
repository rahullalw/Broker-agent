"""Buyer-facing display strings (D4).

These are the most load-bearing strings in the codebase. The grounding guard
compares model output against them byte-for-byte, so they must already read the
way an Ahmedabad broker writes — otherwise the model "helpfully" reformats them
and good turns start getting rejected.

Arithmetic runs through Decimal, not float: 62.555 is not representable in
binary and would round down to 62.55, which is the wrong price.
"""
from __future__ import annotations

from datetime import datetime
from decimal import ROUND_HALF_UP, Decimal

from app.enums import STATUS

LAKH = Decimal(100_000)
CRORE = Decimal(10_000_000)
_CENTS = Decimal("0.01")

# Spelled out rather than strftime("%B"), which follows the process locale and
# would quietly emit "dezembro" on a differently configured machine.
MONTHS = (
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
)

# Same reasoning as MONTHS: strftime("%A") follows the process locale.
WEEKDAYS = (
    "Monday", "Tuesday", "Wednesday", "Thursday",
    "Friday", "Saturday", "Sunday",
)

STATUS_DISPLAY = {
    "ready_to_move": "Ready to move",
    "under_construction": "Under construction",
}


def _quantize(amount: Decimal) -> Decimal:
    """Two decimals, rounded half-up the way a person would."""
    return amount.quantize(_CENTS, rounding=ROUND_HALF_UP)


def _render(amount: Decimal) -> str:
    """Drop trailing zeros: 1.20 -> '1.2', 85.00 -> '85'."""
    return format(amount.normalize(), "f")


def price_display(rupees: int) -> str:
    """8_500_000 -> '₹85 lakh' ; 12_500_000 -> '₹1.25 crore'"""
    if rupees <= 0:
        raise ValueError(f"price must be a positive rupee amount, got {rupees!r}")

    lakhs = _quantize(Decimal(rupees) / LAKH)
    if lakhs >= 100:
        # Nobody says "₹100 lakh" — and rounding can push 99.99 lakh up to it.
        return f"₹{_render(_quantize(Decimal(rupees) / CRORE))} crore"
    return f"₹{_render(lakhs)} lakh"


def possession_display(d: datetime, status: str) -> str:
    """'Ready to move' if status == ready_to_move else 'December 2026'"""
    if status not in STATUS:
        raise ValueError(f"unknown possession status {status!r}, expected one of {STATUS}")
    if status == "ready_to_move":
        return "Ready to move"
    return month_display(d)


def area_display(sqft: int) -> str:
    """1240 -> '1,240 sq ft'"""
    if sqft <= 0:
        raise ValueError(f"carpet area must be positive, got {sqft!r}")
    return f"{sqft:,} sq ft"


def month_display(d: datetime) -> str:
    """datetime(2026, 12, 1) -> 'December 2026'"""
    return f"{MONTHS[d.month - 1]} {d.year}"


def status_display(status: str) -> str:
    """'ready_to_move' -> 'Ready to move'"""
    if status not in STATUS_DISPLAY:
        raise ValueError(f"unknown possession status {status!r}, expected one of {STATUS}")
    return STATUS_DISPLAY[status]


def budget_display(minimum: int | None, maximum: int | None) -> str | None:
    """(5_500_000, 6_500_000) -> '₹55–65 lakh'

    An en dash, and the unit named once when both ends share it — that is how a
    broker writes a range, so the model has no reason to rewrite it.
    """
    if minimum is None and maximum is None:
        return None
    if minimum is None:
        return f"up to {price_display(maximum)}"
    if maximum is None:
        return f"from {price_display(minimum)}"

    low, high = price_display(minimum), price_display(maximum)
    if low == high:
        return low

    low_amount, low_unit = low.rsplit(" ", 1)
    high_amount, high_unit = high.rsplit(" ", 1)
    if low_unit == high_unit:
        # "₹55" + "–" + "65 lakh"; high_amount keeps its own ₹ stripped.
        return f"{low_amount}–{high_amount.lstrip('₹')} {high_unit}"
    return f"{low}–{high}"


def slot_display(d: datetime) -> str:
    """datetime(2026, 8, 9, 17, 0) -> 'Sunday, 9 August 2026, 5:00 pm'

    The weekday is computed. A buyer reads the day name before the date, and an
    off-by-one there is the kind of error that loses a site visit.
    """
    hour = d.hour % 12 or 12
    meridiem = "am" if d.hour < 12 else "pm"
    return (
        f"{WEEKDAYS[d.weekday()]}, {d.day} {MONTHS[d.month - 1]} {d.year}, "
        f"{hour}:{d.minute:02d} {meridiem}"
    )


def amenities_display(amenities: list[str]) -> str:
    """['clubhouse', 'covered_parking'] -> 'clubhouse, covered parking'"""
    return ", ".join(a.replace("_", " ") for a in amenities)
