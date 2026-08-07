"""The seeded world (Section 7).

Everything here is a committed literal. Nothing is generated at runtime, because
the demo has to produce the same 25 properties, the same ids and the same prices
on every machine and every run (D21).

Prices are hand-checked against the Section 5 bands:

    Bopal        8   2BHK  48-70L    3BHK  85L-1.3Cr
    South Bopal  6                   3BHK  95L-1.5Cr
    Shela        6   2BHK  42-62L    3BHK  75L-1.1Cr
    Satellite    5                   3BHK  1.2-2.2Cr

RERA ids use the real Gujarat format with unmistakably fake content - the DEMO
segment is mandatory on all 25, and every developer and tower name is invented
(D22).
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime

from sqlmodel import Session, select

from app import config, db
from app.db import Buyer, Conversation, Message, Property
from app.format import area_display, possession_display, price_display


def _rera(serial: int, year: int) -> str:
    return f"PR/GJ/AHMEDABAD/AHMEDABAD CITY/AUDA/DEMO{serial:04d}/EX1/{year}"


# --------------------------------------------------------------------------
# Properties
# --------------------------------------------------------------------------

PROPERTIES = (
    # ---- Bopal: 4x 2BHK (48-70L), 4x 3BHK (85L-1.3Cr) --------------------
    {
        "id": 1, "title": "Aarohi Serene", "locality": "Bopal", "bhk": 2,
        "price_inr": 5_200_000, "carpet_area_sqft": 985,
        "possession_date": datetime(2025, 6, 1), "status": "ready_to_move",
        "rera_id": _rera(101, 2024), "developer": "Aarohi Realty",
        "tower": "A", "floor": 6,
        "amenities": ["clubhouse", "gym", "covered_parking", "lift", "power_backup"],
        "highlight_note": "Quiet corner block, walking distance to Bopal circle.",
    },
    {
        "id": 2, "title": "Vrund Crest", "locality": "Bopal", "bhk": 2,
        "price_inr": 4_800_000, "carpet_area_sqft": 940,
        "possession_date": datetime(2024, 11, 1), "status": "ready_to_move",
        "rera_id": _rera(102, 2023), "developer": "Vrund Buildcon",
        "tower": "B", "floor": 3,
        "amenities": ["covered_parking", "lift", "24x7_security", "kids_play_area"],
        "highlight_note": "Lowest entry price in Bopal, possession certificate in hand.",
    },
    {
        "id": 3, "title": "Nirvaan Bloom", "locality": "Bopal", "bhk": 2,
        "price_inr": 6_200_000, "carpet_area_sqft": 1_060,
        "possession_date": datetime(2027, 3, 1), "status": "under_construction",
        "rera_id": _rera(103, 2025), "developer": "Nirvaan Group",
        "tower": "C", "floor": 9,
        "amenities": ["clubhouse", "swimming_pool", "gym", "landscaped_garden",
                      "covered_parking", "lift"],
        "highlight_note": "Larger 2BHK layout with a separate utility balcony.",
    },
    {
        "id": 4, "title": "Sanidhya Verve", "locality": "Bopal", "bhk": 2,
        "price_inr": 6_850_000, "carpet_area_sqft": 1_120,
        "possession_date": datetime(2026, 12, 1), "status": "under_construction",
        "rera_id": _rera(104, 2025), "developer": "Sanidhya Infra",
        "tower": "A", "floor": 12,
        "amenities": ["clubhouse", "gym", "jogging_track", "power_backup",
                      "covered_parking", "24x7_security"],
        "highlight_note": "Top-floor units face the open AUDA plot on the west.",
    },
    {
        "id": 5, "title": "Shrishti Elara", "locality": "Bopal", "bhk": 3,
        "price_inr": 9_200_000, "carpet_area_sqft": 1_280,
        "possession_date": datetime(2026, 2, 1), "status": "ready_to_move",
        "rera_id": _rera(105, 2024), "developer": "Shrishti Buildcon",
        "tower": "D", "floor": 8,
        "amenities": ["clubhouse", "gym", "swimming_pool", "covered_parking",
                      "landscaped_garden", "lift", "24x7_security"],
        "highlight_note": "Ready 3BHK with all three bedrooms on the outer wall.",
    },
    {
        "id": 6, "title": "Aarohi Skyline", "locality": "Bopal", "bhk": 3,
        "price_inr": 10_500_000, "carpet_area_sqft": 1_395,
        "possession_date": datetime(2027, 6, 1), "status": "under_construction",
        "rera_id": _rera(106, 2025), "developer": "Aarohi Realty",
        "tower": "B", "floor": 14,
        "amenities": ["clubhouse", "swimming_pool", "gym", "indoor_games",
                      "jogging_track", "power_backup", "lift"],
        "highlight_note": "Two covered parking bays included at this configuration.",
    },
    {
        "id": 7, "title": "Vrund Meadows", "locality": "Bopal", "bhk": 3,
        "price_inr": 8_600_000, "carpet_area_sqft": 1_240,
        "possession_date": datetime(2025, 9, 1), "status": "ready_to_move",
        "rera_id": _rera(107, 2023), "developer": "Vrund Buildcon",
        "tower": "A", "floor": 4,
        "amenities": ["covered_parking", "lift", "power_backup", "community_hall",
                      "landscaped_garden"],
        "highlight_note": "Best value ready 3BHK in Bopal, low maintenance society.",
    },
    {
        "id": 8, "title": "Kalash Grandeur", "locality": "Bopal", "bhk": 3,
        "price_inr": 12_800_000, "carpet_area_sqft": 1_520,
        "possession_date": datetime(2027, 12, 1), "status": "under_construction",
        "rera_id": _rera(108, 2026), "developer": "Kalash Realty",
        "tower": "E", "floor": 17,
        "amenities": ["clubhouse", "swimming_pool", "gym", "kids_play_area",
                      "landscaped_garden", "24x7_security", "power_backup",
                      "covered_parking", "lift"],
        "highlight_note": "Premium tower with private lift lobby per floor.",
    },

    # ---- South Bopal: 6x 3BHK (95L-1.5Cr) --------------------------------
    {
        "id": 9, "title": "Nandan Aurum", "locality": "South Bopal", "bhk": 3,
        "price_inr": 9_800_000, "carpet_area_sqft": 1_310,
        "possession_date": datetime(2026, 1, 1), "status": "ready_to_move",
        "rera_id": _rera(109, 2024), "developer": "Nandan Estates",
        "tower": "A", "floor": 7,
        "amenities": ["clubhouse", "gym", "covered_parking", "lift",
                      "24x7_security", "landscaped_garden"],
        "highlight_note": "Entry-level South Bopal 3BHK, ready and fully occupied.",
    },
    {
        "id": 10, "title": "Silvermist Residency", "locality": "South Bopal", "bhk": 3,
        "price_inr": 11_500_000, "carpet_area_sqft": 1_420,
        "possession_date": datetime(2027, 3, 1), "status": "under_construction",
        "rera_id": _rera(110, 2025), "developer": "Silvermist Developers",
        "tower": "B", "floor": 11,
        "amenities": ["clubhouse", "swimming_pool", "gym", "jogging_track",
                      "kids_play_area", "power_backup", "lift"],
        "highlight_note": "Three-side open plot with a central landscaped podium.",
    },
    {
        "id": 11, "title": "Pravaah Heights", "locality": "South Bopal", "bhk": 3,
        "price_inr": 13_200_000, "carpet_area_sqft": 1_540,
        "possession_date": datetime(2027, 9, 1), "status": "under_construction",
        "rera_id": _rera(111, 2026), "developer": "Pravaah Infra",
        "tower": "C", "floor": 15,
        "amenities": ["clubhouse", "swimming_pool", "gym", "indoor_games",
                      "community_hall", "24x7_security", "covered_parking"],
        "highlight_note": "Corner units get a wraparound balcony on two sides.",
    },
    {
        "id": 12, "title": "Nandan Solitaire", "locality": "South Bopal", "bhk": 3,
        "price_inr": 14_800_000, "carpet_area_sqft": 1_660,
        "possession_date": datetime(2025, 12, 1), "status": "ready_to_move",
        "rera_id": _rera(112, 2024), "developer": "Nandan Estates",
        "tower": "A", "floor": 18,
        "amenities": ["clubhouse", "swimming_pool", "gym", "landscaped_garden",
                      "jogging_track", "power_backup", "lift", "24x7_security"],
        "highlight_note": "Largest ready 3BHK on this side of the SP Ring Road.",
    },
    {
        "id": 13, "title": "Ojas Greenscape", "locality": "South Bopal", "bhk": 3,
        "price_inr": 9_600_000, "carpet_area_sqft": 1_295,
        "possession_date": datetime(2024, 8, 1), "status": "ready_to_move",
        "rera_id": _rera(113, 2023), "developer": "Ojas Realty",
        "tower": "D", "floor": 5,
        "amenities": ["covered_parking", "lift", "landscaped_garden",
                      "kids_play_area", "power_backup"],
        "highlight_note": "Older society, mature trees, very low monthly outgoings.",
    },
    {
        "id": 14, "title": "Silvermist Enclave", "locality": "South Bopal", "bhk": 3,
        "price_inr": 12_400_000, "carpet_area_sqft": 1_480,
        "possession_date": datetime(2026, 12, 1), "status": "under_construction",
        "rera_id": _rera(114, 2025), "developer": "Silvermist Developers",
        "tower": "A", "floor": 10,
        "amenities": ["clubhouse", "gym", "swimming_pool", "covered_parking",
                      "24x7_security", "lift", "indoor_games"],
        "highlight_note": "Handover within the year, finishing work already started.",
    },

    # ---- Shela: 3x 2BHK (42-62L), 3x 3BHK (75L-1.1Cr) --------------------
    {
        "id": 15, "title": "Tvisha Bloom", "locality": "Shela", "bhk": 2,
        "price_inr": 4_300_000, "carpet_area_sqft": 860,
        "possession_date": datetime(2025, 3, 1), "status": "ready_to_move",
        "rera_id": _rera(115, 2023), "developer": "Tvisha Buildcon",
        "tower": "A", "floor": 2,
        "amenities": ["covered_parking", "lift", "24x7_security"],
        "highlight_note": "Most affordable ready unit across all four localities.",
    },
    {
        "id": 16, "title": "Anantam Vista", "locality": "Shela", "bhk": 2,
        "price_inr": 5_100_000, "carpet_area_sqft": 930,
        "possession_date": datetime(2027, 6, 1), "status": "under_construction",
        "rera_id": _rera(116, 2025), "developer": "Anantam Group",
        "tower": "B", "floor": 8,
        "amenities": ["clubhouse", "gym", "kids_play_area", "covered_parking",
                      "lift", "power_backup"],
        "highlight_note": "Compact 2BHK with a dedicated study nook off the hall.",
    },
    {
        "id": 17, "title": "Tvisha Orchid", "locality": "Shela", "bhk": 2,
        "price_inr": 5_950_000, "carpet_area_sqft": 1_010,
        "possession_date": datetime(2026, 5, 1), "status": "ready_to_move",
        "rera_id": _rera(117, 2024), "developer": "Tvisha Buildcon",
        "tower": "C", "floor": 6,
        "amenities": ["clubhouse", "gym", "landscaped_garden", "covered_parking",
                      "lift", "community_hall"],
        "highlight_note": "Largest 2BHK carpet in Shela, ready to shift into.",
    },
    {
        "id": 18, "title": "Anantam Crest", "locality": "Shela", "bhk": 3,
        "price_inr": 7_800_000, "carpet_area_sqft": 1_185,
        "possession_date": datetime(2024, 12, 1), "status": "ready_to_move",
        "rera_id": _rera(118, 2023), "developer": "Anantam Group",
        "tower": "A", "floor": 4,
        "amenities": ["covered_parking", "lift", "power_backup", "kids_play_area"],
        "highlight_note": "Cheapest ready 3BHK on the books, good for first upgrades.",
    },
    {
        "id": 19, "title": "Rushil Panorama", "locality": "Shela", "bhk": 3,
        "price_inr": 9_400_000, "carpet_area_sqft": 1_330,
        "possession_date": datetime(2027, 3, 1), "status": "under_construction",
        "rera_id": _rera(119, 2025), "developer": "Rushil Estates",
        "tower": "B", "floor": 12,
        "amenities": ["clubhouse", "swimming_pool", "gym", "jogging_track",
                      "landscaped_garden", "covered_parking", "lift"],
        "highlight_note": "Wide-frontage flats, every bedroom takes an outer wall.",
    },
    {
        "id": 20, "title": "Rushil Woods", "locality": "Shela", "bhk": 3,
        "price_inr": 10_800_000, "carpet_area_sqft": 1_450,
        "possession_date": datetime(2028, 3, 1), "status": "under_construction",
        "rera_id": _rera(120, 2026), "developer": "Rushil Estates",
        "tower": "C", "floor": 16,
        "amenities": ["clubhouse", "swimming_pool", "gym", "indoor_games",
                      "community_hall", "24x7_security", "power_backup", "lift"],
        "highlight_note": "Longest runway to possession, best payment plan on offer.",
    },

    # ---- Satellite: 5x 3BHK (1.2-2.2Cr) ----------------------------------
    {
        "id": 21, "title": "Meridian Ashray", "locality": "Satellite", "bhk": 3,
        "price_inr": 12_500_000, "carpet_area_sqft": 1_470,
        "possession_date": datetime(2025, 7, 1), "status": "ready_to_move",
        "rera_id": _rera(121, 2024), "developer": "Meridian Buildcon",
        "tower": "A", "floor": 9,
        "amenities": ["clubhouse", "gym", "covered_parking", "lift",
                      "24x7_security", "power_backup"],
        "highlight_note": "Entry point into Satellite, walk to Jodhpur cross roads.",
    },
    {
        "id": 22, "title": "Zenith Palladio", "locality": "Satellite", "bhk": 3,
        "price_inr": 16_200_000, "carpet_area_sqft": 1_680,
        "possession_date": datetime(2027, 6, 1), "status": "under_construction",
        "rera_id": _rera(122, 2025), "developer": "Zenith Realty",
        "tower": "B", "floor": 13,
        "amenities": ["clubhouse", "swimming_pool", "gym", "indoor_games",
                      "jogging_track", "landscaped_garden", "covered_parking"],
        "highlight_note": "Double-height lobby and a rooftop deck on the 20th.",
    },
    {
        "id": 23, "title": "Meridian Trilogy", "locality": "Satellite", "bhk": 3,
        "price_inr": 14_500_000, "carpet_area_sqft": 1_590,
        "possession_date": datetime(2026, 3, 1), "status": "ready_to_move",
        "rera_id": _rera(123, 2024), "developer": "Meridian Buildcon",
        "tower": "C", "floor": 11,
        "amenities": ["clubhouse", "swimming_pool", "gym", "covered_parking",
                      "lift", "24x7_security", "community_hall"],
        "highlight_note": "Ready possession with a clear title and no bank lien.",
    },
    {
        "id": 24, "title": "Aveza Crown", "locality": "Satellite", "bhk": 3,
        "price_inr": 21_500_000, "carpet_area_sqft": 1_920,
        "possession_date": datetime(2028, 6, 1), "status": "under_construction",
        "rera_id": _rera(124, 2026), "developer": "Aveza Group",
        "tower": "A", "floor": 21,
        "amenities": ["clubhouse", "swimming_pool", "gym", "indoor_games",
                      "jogging_track", "landscaped_garden", "community_hall",
                      "24x7_security", "power_backup", "covered_parking"],
        "highlight_note": "Most expensive listing, four-side open with a city view.",
    },
    {
        "id": 25, "title": "Zenith Corniche", "locality": "Satellite", "bhk": 3,
        "price_inr": 18_800_000, "carpet_area_sqft": 1_780,
        "possession_date": datetime(2024, 10, 1), "status": "ready_to_move",
        "rera_id": _rera(125, 2023), "developer": "Zenith Realty",
        "tower": "D", "floor": 15,
        "amenities": ["clubhouse", "gym", "swimming_pool", "landscaped_garden",
                      "covered_parking", "lift", "24x7_security"],
        "highlight_note": "Established address, resale-heavy tower with quick moves.",
    },
)


# --------------------------------------------------------------------------
# Buyers and conversations (D21)
# --------------------------------------------------------------------------

# Fixed ids so the demo curl commands never change. Every qualification field
# starts unknown on purpose: the known/unknown block (D12) and the whole
# qualification flow are only interesting if the agent has to fill them in.
BUYERS = (
    {
        "id": 1, "name": "Priya", "phone": "+919825000001", "intent_tier": "cold",
        "budget_min": None, "budget_max": None, "preferred_localities": None,
        "bhk_need": None, "possession_need": None, "family_size": None,
    },
    {
        "id": 2, "name": "Rakesh", "phone": "+919825000002", "intent_tier": "cold",
        "budget_min": None, "budget_max": None, "preferred_localities": None,
        "bhk_need": None, "possession_need": None, "family_size": None,
    },
    {
        "id": 3, "name": "Anjali", "phone": "+919825000003", "intent_tier": "cold",
        "budget_min": None, "budget_max": None, "preferred_localities": None,
        "bhk_need": None, "possession_need": None, "family_size": None,
    },
)

# Fixed opening messages so the live demo cannot wander.
#   1 Priya   - English, mid-budget: happy path, search -> details -> book -> escalate
#   2 Rakesh  - Hinglish negotiator: qualification, then a price ask -> regex escalation
#   3 Anjali  - deliberately unsatisfiable against the seeded bands -> D14 relaxation
OPENINGS = (
    "Hi, looking for a 3BHK in Bopal",
    "3bhk chahiye bhai, budget 65 lakh",
    "3BHK in Satellite under 80 lakh",
)

CONVERSATIONS = (
    {"id": 1, "buyer_id": 1},
    {"id": 2, "buyer_id": 2},
    {"id": 3, "buyer_id": 3},
)


# --------------------------------------------------------------------------
# Seeding
# --------------------------------------------------------------------------

def is_seeded(session: Session) -> bool:
    return session.exec(select(Property).limit(1)).first() is not None


def reset() -> None:
    """Delete the database file and rebuild the schema.

    Deleting the file is the whole reset story now that this is SQLite (D21).
    The engine is disposed first because Windows will not unlink a file that
    still has an open handle in the connection pool.
    """
    db.engine.dispose()
    path = config.db_path()
    for suffix in ("", "-wal", "-shm"):
        candidate = path.with_name(path.name + suffix)
        candidate.unlink(missing_ok=True)
    db.init_db()


def seed_all(session: Session) -> None:
    """Insert the committed dataset. Ids are explicit, not autoincremented, so
    that search ranking tie-breaks (D14) are reproducible."""
    session.add_all([Property(**row) for row in PROPERTIES])
    session.add_all([Buyer(**row) for row in BUYERS])
    session.add_all([Conversation(**row) for row in CONVERSATIONS])
    session.commit()

    # Only user and assistant rows are ever written (D11); these are user rows,
    # so they carry no model or latency telemetry.
    session.add_all([
        Message(conversation_id=convo["id"], role="user", content=opening)
        for convo, opening in zip(CONVERSATIONS, OPENINGS)
    ])
    session.commit()


# --------------------------------------------------------------------------
# Summary table
# --------------------------------------------------------------------------

def summary_lines() -> list[str]:
    from app.enums import LOCALITIES

    lines = [
        "",
        f"AllSet seeded -> {config.db_path()}",
        "",
        f"  {'Locality':<13}{'2BHK':>5}{'3BHK':>6}{'Total':>7}{'Ready':>7}{'Under c.':>10}"
        f"   {'Price range':<26}",
        "  " + "-" * 77,
    ]

    for locality in LOCALITIES:
        rows = [p for p in PROPERTIES if p["locality"] == locality]
        prices = [p["price_inr"] for p in rows]
        span = f"{price_display(min(prices))} - {price_display(max(prices))}"
        lines.append(
            f"  {locality:<13}"
            f"{sum(1 for p in rows if p['bhk'] == 2):>5}"
            f"{sum(1 for p in rows if p['bhk'] == 3):>6}"
            f"{len(rows):>7}"
            f"{sum(1 for p in rows if p['status'] == 'ready_to_move'):>7}"
            f"{sum(1 for p in rows if p['status'] == 'under_construction'):>10}"
            f"   {span:<26}"
        )

    lines += [
        "  " + "-" * 77,
        f"  {'All':<13}"
        f"{sum(1 for p in PROPERTIES if p['bhk'] == 2):>5}"
        f"{sum(1 for p in PROPERTIES if p['bhk'] == 3):>6}"
        f"{len(PROPERTIES):>7}"
        f"{sum(1 for p in PROPERTIES if p['status'] == 'ready_to_move'):>7}"
        f"{sum(1 for p in PROPERTIES if p['status'] == 'under_construction'):>10}",
        "",
        "  Buyers and their opening messages",
        "  " + "-" * 77,
    ]
    for buyer, opening in zip(BUYERS, OPENINGS):
        lines.append(f"  {buyer['id']}  {buyer['name']:<8} conversation {buyer['id']}"
                     f"  \"{opening}\"")

    sample = PROPERTIES[4]
    lines += [
        "",
        "  Sample display strings (D4)",
        "  " + "-" * 77,
        f"  {sample['title']:<20} {price_display(sample['price_inr'])}"
        f"  ·  {area_display(sample['carpet_area_sqft'])}"
        f"  ·  {possession_display(sample['possession_date'], sample['status'])}",
        f"  {'RERA':<20} {sample['rera_id']}",
        "",
        f"  Tables: properties {len(PROPERTIES)}  ·  buyers {len(BUYERS)}"
        f"  ·  conversations {len(CONVERSATIONS)}  ·  messages {len(OPENINGS)}",
        "",
    ]
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m app.seed",
        description="Seed the AllSet demo database.",
    )
    parser.add_argument(
        "--reset", action="store_true",
        help="delete the database file, recreate the schema and reseed",
    )
    args = parser.parse_args(argv)

    # The display strings are non-ASCII by mandate (D4). When stdout is a pipe
    # rather than a console, Windows defaults it to cp1252 and printing the
    # rupee sign kills the process, so pin the encoding before printing.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")

    if args.reset:
        reset()
    else:
        db.init_db()
        with db.get_session() as session:
            if is_seeded(session):
                print(
                    "Database already contains properties. "
                    "Re-run with --reset to delete and rebuild it.",
                    file=sys.stderr,
                )
                return 1

    with db.get_session() as session:
        seed_all(session)

    print("\n".join(summary_lines()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
