"""Section 7 - the seeded world the whole demo runs on.

The dataset is committed as literals so the demo is reproducible (no runtime
generation). These tests are the hand-check the spec asks for.
"""
import os
import re
import subprocess
import sys
from datetime import datetime

import pytest
from sqlalchemy import inspect
from sqlmodel import Session, create_engine, select

from app import seed
from app.db import Buyer, Conversation, Message, Property
from app.enums import AMENITIES, LOCALITIES, STATUS

# Section 5 distribution. Inclusive rupee bounds.
BANDS = {
    ("Bopal", 2): (4_800_000, 7_000_000),
    ("Bopal", 3): (8_500_000, 13_000_000),
    ("South Bopal", 3): (9_500_000, 15_000_000),
    ("Shela", 2): (4_200_000, 6_200_000),
    ("Shela", 3): (7_500_000, 11_000_000),
    ("Satellite", 3): (12_000_000, 22_000_000),
}

LOCALITY_COUNTS = {"Bopal": 8, "South Bopal": 6, "Shela": 6, "Satellite": 5}

RERA_PATTERN = re.compile(
    r"^PR/GJ/AHMEDABAD/AHMEDABAD CITY/AUDA/DEMO\d{4}/EX1/\d{4}$"
)

# Real Ahmedabad builders the fictional names must not collide with (D22).
REAL_BUILDERS = (
    "adani", "godrej", "shivalik", "bakeri", "sun builders", "goyal",
    "pacifica", "ganesh housing", "safal", "shilp", "savvy", "binori",
    "dev ", "nila", "arvind", "sheetal", "iscon", "applewoods",
)


def run_seed(tmp_path, *args):
    """Run the CLI the way the demo does, against a throwaway database."""
    env = {**os.environ, "DATABASE_URL": "sqlite:///" + (tmp_path / "allset.db").as_posix()}
    return subprocess.run(
        [sys.executable, "-m", "app.seed", *args],
        capture_output=True, text=True, encoding="utf-8", cwd=os.getcwd(), env=env,
    )


def open_seeded(tmp_path):
    return Session(create_engine("sqlite:///" + (tmp_path / "allset.db").as_posix()))


@pytest.fixture(scope="module")
def seeded(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("seeded")
    result = run_seed(tmp, "--reset")
    assert result.returncode == 0, result.stderr
    return tmp


class TestPropertyDataset:
    def test_exactly_twenty_five_properties(self):
        assert len(seed.PROPERTIES) == 25

    def test_locality_distribution_matches_the_spec(self):
        counts = {}
        for p in seed.PROPERTIES:
            counts[p["locality"]] = counts.get(p["locality"], 0) + 1
        assert counts == LOCALITY_COUNTS

    @pytest.mark.parametrize("prop", seed.PROPERTIES, ids=lambda p: p["title"])
    def test_price_sits_inside_its_band(self, prop):
        key = (prop["locality"], prop["bhk"])
        assert key in BANDS, f"no band defined for {key}"
        low, high = BANDS[key]
        assert low <= prop["price_inr"] <= high, (
            f"{prop['title']}: {prop['price_inr']} outside {low}-{high} for {key}"
        )

    def test_south_bopal_and_satellite_are_three_bhk_only(self):
        for p in seed.PROPERTIES:
            if p["locality"] in ("South Bopal", "Satellite"):
                assert p["bhk"] == 3, p["title"]

    def test_bopal_and_shela_offer_both_sizes(self):
        for locality in ("Bopal", "Shela"):
            sizes = {p["bhk"] for p in seed.PROPERTIES if p["locality"] == locality}
            assert sizes == {2, 3}, locality

    def test_vocabularies_are_respected(self):
        for p in seed.PROPERTIES:
            assert p["locality"] in LOCALITIES
            assert p["status"] in STATUS
            assert p["bhk"] in (2, 3)
            assert set(p["amenities"]) <= set(AMENITIES), p["title"]
            assert p["amenities"], f"{p['title']} has no amenities"

    def test_titles_are_unique(self):
        titles = [p["title"] for p in seed.PROPERTIES]
        assert len(set(titles)) == len(titles)

    def test_both_possession_states_are_represented(self):
        # The possession question needs a real answer either way.
        states = {p["status"] for p in seed.PROPERTIES}
        assert states == set(STATUS)

    def test_every_locality_has_at_least_one_ready_to_move(self):
        for locality in LOCALITIES:
            states = {p["status"] for p in seed.PROPERTIES if p["locality"] == locality}
            assert "ready_to_move" in states, locality

    def test_possession_date_agrees_with_status(self):
        now = datetime.now()
        for p in seed.PROPERTIES:
            if p["status"] == "ready_to_move":
                assert p["possession_date"] <= now, f"{p['title']} is not actually ready"
            else:
                assert p["possession_date"] > now, f"{p['title']} is already delivered"

    def test_carpet_area_grows_with_bhk(self):
        two = [p["carpet_area_sqft"] for p in seed.PROPERTIES if p["bhk"] == 2]
        three = [p["carpet_area_sqft"] for p in seed.PROPERTIES if p["bhk"] == 3]
        assert max(two) < min(three), "a 2BHK must not out-measure every 3BHK"

    def test_highlight_note_is_one_human_line(self):
        for p in seed.PROPERTIES:
            note = p["highlight_note"]
            assert note and "\n" not in note
            assert len(note) <= 120, p["title"]


class TestReraIds:
    def test_every_id_carries_the_demo_segment(self):
        for p in seed.PROPERTIES:
            assert "DEMO" in p["rera_id"], p["title"]

    def test_every_id_uses_the_real_gujarat_format(self):
        for p in seed.PROPERTIES:
            assert RERA_PATTERN.match(p["rera_id"]), p["rera_id"]

    def test_ids_are_unique(self):
        ids = [p["rera_id"] for p in seed.PROPERTIES]
        assert len(set(ids)) == len(ids)


class TestFictionalNames:
    def test_developers_do_not_collide_with_real_ahmedabad_builders(self):
        for p in seed.PROPERTIES:
            lowered = p["developer"].lower()
            for real in REAL_BUILDERS:
                assert real not in lowered, f"{p['developer']} looks like a real builder"

    def test_towers_are_named(self):
        for p in seed.PROPERTIES:
            assert p["tower"]
            assert p["floor"] >= 1


class TestPersonas:
    def test_three_buyers_at_fixed_ids(self):
        assert [b["id"] for b in seed.BUYERS] == [1, 2, 3]

    def test_personas_are_the_three_from_the_spec(self):
        assert [b["name"] for b in seed.BUYERS] == ["Priya", "Rakesh", "Anjali"]

    def test_no_opening_message_is_seeded(self):
        assert not hasattr(seed, "OPENINGS")

    def test_qualification_starts_unknown_so_the_agent_has_work_to_do(self):
        # D12 - the known/unknown block is only interesting if it starts empty.
        for b in seed.BUYERS:
            for field in ("budget_min", "budget_max", "preferred_localities",
                          "bhk_need", "possession_need", "family_size"):
                assert b.get(field) is None, f"{b['name']}.{field} should start unknown"

    def test_anjali_is_genuinely_unsatisfiable(self):
        # Persona 3 must produce zero exact matches so D14 relaxation has to fire.
        matches = [
            p for p in seed.PROPERTIES
            if p["locality"] == "Satellite" and p["bhk"] == 3 and p["price_inr"] < 8_000_000
        ]
        assert matches == []

    def test_priya_has_real_matches_to_find(self):
        matches = [p for p in seed.PROPERTIES if p["locality"] == "Bopal" and p["bhk"] == 3]
        assert len(matches) >= 3

    def test_rakesh_65_lakh_3bhk_has_no_exact_match_but_near_misses_exist(self):
        exact = [p for p in seed.PROPERTIES if p["bhk"] == 3 and p["price_inr"] <= 6_500_000]
        assert exact == [], "a 3BHK at 65 lakh must not exist, or the negotiation never starts"
        near = [p for p in seed.PROPERTIES if p["bhk"] == 2 and p["price_inr"] <= 6_500_000]
        assert near, "there must be 2BHK near-misses to offer him"


class TestSeededDatabase:
    def test_all_eight_tables_exist(self, seeded):
        engine = create_engine("sqlite:///" + (seeded / "allset.db").as_posix())
        assert len(inspect(engine).get_table_names()) == 8

    def test_twenty_five_properties_land_in_the_database(self, seeded):
        with open_seeded(seeded) as s:
            assert len(s.exec(select(Property)).all()) == 25

    def test_buyers_keep_their_fixed_ids(self, seeded):
        with open_seeded(seeded) as s:
            buyers = s.exec(select(Buyer).order_by(Buyer.id)).all()
            assert [b.id for b in buyers] == [1, 2, 3]
            assert [b.name for b in buyers] == ["Priya", "Rakesh", "Anjali"]

    def test_three_active_conversations_at_fixed_ids(self, seeded):
        with open_seeded(seeded) as s:
            convos = s.exec(select(Conversation).order_by(Conversation.id)).all()
            assert [c.id for c in convos] == [1, 2, 3]
            assert [c.buyer_id for c in convos] == [1, 2, 3]
            assert {c.status for c in convos} == {"active"}
            assert {c.clarification_count for c in convos} == {0}

    def test_every_conversation_starts_with_an_empty_transcript(self, seeded):
        with open_seeded(seeded) as s:
            for convo_id in (1, 2, 3):
                msgs = s.exec(
                    select(Message).where(Message.conversation_id == convo_id)
                ).all()
                assert msgs == []

    def test_opening_messages_carry_no_assistant_telemetry(self, seeded):
        with open_seeded(seeded) as s:
            for m in s.exec(select(Message)).all():
                assert m.model_used is None
                assert m.latency_ms is None

    def test_property_amenities_survive_the_json_round_trip(self, seeded):
        with open_seeded(seeded) as s:
            for p in s.exec(select(Property)).all():
                assert isinstance(p.amenities, list)
                assert p.amenities

    def test_possession_dates_survive_as_naive_datetimes(self, seeded):
        with open_seeded(seeded) as s:
            for p in s.exec(select(Property)).all():
                assert isinstance(p.possession_date, datetime)
                assert p.possession_date.tzinfo is None


class TestResetCli:
    def test_reset_prints_a_summary_table(self, tmp_path):
        result = run_seed(tmp_path, "--reset")
        assert result.returncode == 0, result.stderr
        out = result.stdout
        assert "25" in out
        for locality in LOCALITIES:
            assert locality in out
        assert "Priya" in out or "buyers" in out.lower()

    def test_summary_survives_a_non_utf8_console(self, tmp_path):
        # Piping the CLI must not die on the rupee sign; cp1252 cannot encode it.
        result = run_seed(tmp_path, "--reset")
        assert result.returncode == 0, result.stderr
        assert "₹" in result.stdout
        assert "UnicodeEncodeError" not in result.stderr

    def test_reset_runs_cleanly_twice_in_a_row(self, tmp_path):
        # Exit criterion: the file is deleted and rebuilt, not appended to.
        first = run_seed(tmp_path, "--reset")
        assert first.returncode == 0, first.stderr
        second = run_seed(tmp_path, "--reset")
        assert second.returncode == 0, second.stderr
        with open_seeded(tmp_path) as s:
            assert len(s.exec(select(Property)).all()) == 25
            assert len(s.exec(select(Buyer)).all()) == 3
            assert len(s.exec(select(Message)).all()) == 0

    def test_reset_is_deterministic(self, tmp_path):
        def fingerprint():
            with open_seeded(tmp_path) as s:
                return [
                    (p.id, p.title, p.price_inr, p.rera_id)
                    for p in s.exec(select(Property).order_by(Property.id)).all()
                ]

        run_seed(tmp_path, "--reset")
        before = fingerprint()
        run_seed(tmp_path, "--reset")
        assert fingerprint() == before

    def test_reset_deletes_the_file_rather_than_truncating_tables(self, tmp_path):
        run_seed(tmp_path, "--reset")
        db_file = tmp_path / "allset.db"
        stray = "CREATE TABLE leftover_from_an_older_schema (id INTEGER)"
        engine = create_engine("sqlite:///" + db_file.as_posix())
        with engine.connect() as conn:
            conn.exec_driver_sql(stray)
            conn.commit()
        engine.dispose()

        run_seed(tmp_path, "--reset")
        rebuilt = create_engine("sqlite:///" + db_file.as_posix())
        assert "leftover_from_an_older_schema" not in inspect(rebuilt).get_table_names()

    def test_seeding_a_populated_database_without_reset_is_refused(self, tmp_path):
        assert run_seed(tmp_path, "--reset").returncode == 0
        again = run_seed(tmp_path)
        assert again.returncode != 0
        assert "--reset" in (again.stdout + again.stderr)
        with open_seeded(tmp_path) as s:
            assert len(s.exec(select(Property)).all()) == 25, "must not duplicate rows"


class TestOfflineByDesign:
    def test_seed_pulls_in_no_llm_or_http_machinery(self):
        source = (seed.__file__)
        with open(source, encoding="utf-8") as fh:
            text = fh.read()
        for forbidden in ("openai", "httpx", "fastapi", "requests"):
            assert forbidden not in text, f"Phase 1 is offline; {forbidden} does not belong"
