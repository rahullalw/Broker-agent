"""Phase 2 — the five tools as plain Python functions.

Two rules run through every test here:

* buyer-facing fields are display strings, never raw ints or ISO dates (D4) —
  that is what makes the Phase 4 guard a byte-exact substring match;
* bad input steers, it never raises (§1). A tool that throws hands the loop a
  crash where the model needed a sentence telling it what to send instead.
"""
from datetime import datetime, timedelta

import pytest
from sqlmodel import select

from app.db import Buyer, Conversation, Escalation, Property, SiteVisit
from app.enums import LOCALITIES
from app.tools import (
    book_site_visit,
    escalate_to_broker,
    get_property_details,
    search_properties,
    update_buyer_profile,
)

pytestmark = pytest.mark.usefixtures("seeded")

# Raw values that must never reach the buyer through a tool result (D4).
RAW_PRICES = ("8600000", "9200000", "12500000")


def all_cards(result):
    return result["exact_matches"] + result["near_matches"]


def flat_strings(value):
    """Every string anywhere in a nested tool result."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from flat_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from flat_strings(item)


# --------------------------------------------------------------------------
# search_properties (D14)
# --------------------------------------------------------------------------

class TestSearchShape:
    def test_every_call_returns_the_four_documented_keys(self):
        result = search_properties(bhk=3, localities=["Bopal"])
        assert set(result) == {"exact_matches", "near_matches", "relaxed", "note"}

    def test_a_card_carries_exactly_the_documented_fields(self):
        card = search_properties(bhk=3, localities=["Bopal"])["exact_matches"][0]
        assert set(card) == {
            "id", "title", "locality", "bhk",
            "price_display", "possession_display", "fit_reason",
        }

    def test_no_arguments_at_all_is_legal(self):
        # The agent must be able to search on partial qualification.
        result = search_properties()
        assert result["exact_matches"]

    def test_at_most_five_cards_come_back(self):
        result = search_properties()
        assert len(result["exact_matches"]) <= 5

    def test_prices_ship_pre_formatted(self):
        for card in all_cards(search_properties(bhk=3, localities=["Bopal"])):
            assert card["price_display"].startswith("₹")
            assert "lakh" in card["price_display"] or "crore" in card["price_display"]

    def test_no_raw_rupee_integer_leaks_anywhere_in_the_result(self):
        blob = " ".join(flat_strings(search_properties()))
        for raw in RAW_PRICES:
            assert raw not in blob

    def test_no_iso_date_leaks_anywhere_in_the_result(self):
        blob = " ".join(flat_strings(search_properties()))
        assert "T00:00:00" not in blob
        assert "2026-12" not in blob


class TestSearchRanking:
    def test_the_same_query_five_times_is_byte_identical(self):
        query = dict(bhk=3, localities=["Bopal", "Shela"], budget_max=10_000_000)
        first = search_properties(**query)
        for _ in range(4):
            assert search_properties(**query) == first

    def test_in_budget_outranks_out_of_budget(self):
        result = search_properties(bhk=3, localities=["Bopal"], budget_max=9_000_000)
        ids = [c["id"] for c in result["exact_matches"]]
        assert ids[0] in (5, 7), "an in-budget Bopal 3BHK must lead"

    def test_stated_locality_order_breaks_ties(self):
        first = search_properties(bhk=2, localities=["Shela", "Bopal"])["exact_matches"]
        second = search_properties(bhk=2, localities=["Bopal", "Shela"])["exact_matches"]
        assert first[0]["locality"] == "Shela"
        assert second[0]["locality"] == "Bopal"

    def test_id_is_the_final_tie_break(self):
        # Same locality, no budget and no possession filter: id decides.
        ids = [c["id"] for c in search_properties(bhk=2, localities=["Bopal"])["exact_matches"]]
        assert ids == sorted(ids)


class TestSearchFilters:
    def test_bhk_filter_is_exact(self):
        for card in all_cards(search_properties(bhk=2)):
            assert card["bhk"] == 2

    def test_locality_filter_is_respected_when_nothing_is_relaxed(self):
        result = search_properties(bhk=3, localities=["Bopal"])
        assert result["relaxed"] is None
        for card in result["exact_matches"]:
            assert card["locality"] == "Bopal"

    def test_budget_bounds_are_inclusive(self):
        result = search_properties(bhk=3, localities=["Bopal"], budget_max=8_600_000)
        assert 7 in [c["id"] for c in result["exact_matches"]], "₹86 lakh is within ₹86 lakh"

    def test_possession_before_excludes_later_handover(self):
        result = search_properties(bhk=3, localities=["Bopal"], possession_before="2026-06")
        for card in result["exact_matches"]:
            assert card["possession_display"] in ("Ready to move", "February 2026")

    def test_possession_before_month_is_inclusive(self):
        result = search_properties(bhk=3, localities=["Bopal"], possession_before="2026-02")
        assert 5 in [c["id"] for c in result["exact_matches"]]


class TestSearchNeverReturnsEmpty:
    """The Phase 2 exit criterion, spelled out."""

    @pytest.fixture
    def anjali(self):
        return search_properties(bhk=3, localities=["Satellite"], budget_max=8_000_000)

    def test_no_exact_matches(self, anjali):
        assert anjali["exact_matches"] == []

    def test_three_near_matches_come_back_instead(self, anjali):
        assert len(anjali["near_matches"]) == 3

    def test_the_relaxation_is_named(self, anjali):
        assert anjali["relaxed"]
        assert "budget" in anjali["relaxed"]

    def test_the_note_reads_like_a_sentence(self, anjali):
        note = anjali["note"]
        assert note.endswith(".")
        assert "3BHK" in note
        assert "Satellite" in note
        assert "₹80 lakh" in note

    def test_the_note_quotes_the_real_satellite_floor(self, anjali):
        # Cheapest Satellite 3BHK in the seed is ₹1.25 crore.
        assert "₹1.25 crore" in anjali["note"]

    def test_bhk_is_never_relaxed(self, anjali):
        for card in anjali["near_matches"]:
            assert card["bhk"] == 3, "a 3BHK buyer is not shown 2BHKs"

    def test_budget_relaxes_before_locality(self):
        # ₹87 lakh + 25% reaches three Bopal 3BHKs, so rung 2 never runs and the
        # buyer is not offered a locality they did not ask for.
        result = search_properties(bhk=3, localities=["Bopal"], budget_max=8_700_000)
        assert {c["locality"] for c in result["near_matches"]} == {"Bopal"}
        assert "budget" in result["relaxed"]
        assert "localities" not in result["relaxed"]

    def test_locality_widens_only_once_budget_alone_is_not_enough(self):
        # ₹80 lakh + 25% finds only two Bopal 3BHKs, so rung 2 has to run.
        result = search_properties(bhk=3, localities=["Bopal"], budget_max=8_000_000)
        assert result["exact_matches"] == []
        assert len(result["near_matches"]) == 3
        assert "budget" in result["relaxed"]
        assert "localities widened to" in result["relaxed"]
        assert {c["locality"] for c in result["near_matches"]} <= {
            "Bopal", "South Bopal", "Shela"
        }

    def test_relaxation_only_tops_the_list_up_to_three(self):
        result = search_properties(bhk=3, localities=["Bopal"], budget_max=8_700_000)
        assert len(result["exact_matches"]) == 1
        assert len(result["exact_matches"]) + len(result["near_matches"]) == 3

    def test_an_untroubled_search_reports_no_relaxation(self):
        result = search_properties(bhk=3, localities=["Bopal"])
        assert result["relaxed"] is None
        assert result["note"] is None
        assert result["near_matches"] == []


class TestFitReason:
    def test_under_budget_is_templated_with_the_gap(self):
        result = search_properties(bhk=3, localities=["Bopal"], budget_max=9_200_000)
        card = next(c for c in result["exact_matches"] if c["id"] == 7)
        assert "₹6 lakh under your budget" in card["fit_reason"]

    def test_over_budget_is_templated_with_the_gap(self):
        result = search_properties(bhk=3, localities=["Bopal"], budget_max=8_000_000)
        card = next(c for c in result["near_matches"] if c["id"] == 7)
        assert "₹6 lakh over your budget" in card["fit_reason"]

    def test_locality_is_named(self):
        card = search_properties(bhk=3, localities=["Bopal"])["exact_matches"][0]
        assert "in Bopal" in card["fit_reason"]

    def test_parts_are_joined_with_a_middot(self):
        result = search_properties(bhk=3, localities=["Bopal"], budget_max=9_200_000)
        card = next(c for c in result["exact_matches"] if c["id"] == 7)
        assert card["fit_reason"] == "₹6 lakh under your budget · in Bopal"

    def test_possession_appears_only_when_it_was_asked_for(self):
        without = search_properties(bhk=3, localities=["Bopal"])["exact_matches"]
        assert all("possession" not in c["fit_reason"] for c in without)
        with_filter = search_properties(
            bhk=3, localities=["Bopal"], possession_before="2027-12"
        )["exact_matches"]
        assert any(
            "possession" in c["fit_reason"] or "ready to move" in c["fit_reason"]
            for c in with_filter
        )

    def test_ready_to_move_is_its_own_phrase(self):
        result = search_properties(bhk=3, localities=["Bopal"], possession_before="2027-12")
        card = next(c for c in result["exact_matches"] if c["id"] == 5)
        assert "ready to move" in card["fit_reason"]


class TestSearchSteeringErrors:
    def test_an_unknown_locality_names_the_four_valid_ones(self):
        result = search_properties(localities=["Bopal West"])
        assert "error" in result
        for locality in LOCALITIES:
            assert locality in result["error"]

    def test_a_malformed_possession_month_steers(self):
        result = search_properties(possession_before="next December")
        assert "error" in result
        assert "YYYY-MM" in result["error"]

    def test_an_inverted_budget_steers(self):
        result = search_properties(budget_min=9_000_000, budget_max=5_000_000)
        assert "error" in result

    def test_an_unsupported_bhk_steers_rather_than_returning_nothing(self):
        result = search_properties(bhk=7)
        assert "error" in result


# --------------------------------------------------------------------------
# get_property_details
# --------------------------------------------------------------------------

class TestPropertyDetails:
    def test_legal_returns_the_rera_id_verbatim(self, session):
        expected = session.get(Property, 5).rera_id
        assert get_property_details(5, ["legal"])["legal"]["rera_id"] == expected

    def test_legal_says_nothing_about_pricing(self):
        result = get_property_details(5, ["legal"])
        assert "pricing" not in result
        blob = " ".join(flat_strings(result))
        assert "lakh" not in blob and "crore" not in blob and "sq ft" not in blob

    def test_only_requested_sections_come_back(self):
        result = get_property_details(5, ["pricing"])
        assert set(result) & {"pricing", "possession", "legal", "amenities"} == {"pricing"}

    def test_several_sections_at_once(self):
        result = get_property_details(5, ["pricing", "amenities"])
        assert set(result) & {"pricing", "possession", "legal", "amenities"} == {
            "pricing", "amenities"
        }

    def test_pricing_fields_match_the_table(self):
        assert set(get_property_details(5, ["pricing"])["pricing"]) == {
            "price_display", "carpet_area_display", "tower", "floor",
        }

    def test_possession_fields_match_the_table(self):
        assert set(get_property_details(5, ["possession"])["possession"]) == {
            "possession_display", "status_display",
        }

    def test_legal_fields_match_the_table(self):
        assert set(get_property_details(5, ["legal"])["legal"]) == {
            "rera_id", "developer", "approval_note",
        }

    def test_amenities_read_as_prose(self):
        amenities = get_property_details(5, ["amenities"])["amenities"]["amenities"]
        assert isinstance(amenities, str)
        assert "_" not in amenities
        assert ", " in amenities

    def test_pricing_is_pre_formatted(self):
        pricing = get_property_details(5, ["pricing"])["pricing"]
        assert pricing["price_display"] == "₹92 lakh"
        assert pricing["carpet_area_display"] == "1,280 sq ft"

    def test_possession_is_pre_formatted(self):
        possession = get_property_details(6, ["possession"])["possession"]
        assert possession["possession_display"] == "June 2027"
        assert possession["status_display"] == "Under construction"

    def test_ready_to_move_reads_as_ready(self):
        possession = get_property_details(5, ["possession"])["possession"]
        assert possession["possession_display"] == "Ready to move"
        assert possession["status_display"] == "Ready to move"

    def test_approval_note_is_fixed_prose_not_an_inferred_status(self):
        note = get_property_details(5, ["legal"])["legal"]["approval_note"]
        assert note == get_property_details(6, ["legal"])["legal"]["approval_note"]
        assert "RERA" in note

    def test_the_property_is_identified_so_the_agent_can_name_it(self):
        result = get_property_details(5, ["legal"])
        assert result["id"] == 5
        assert result["title"] == "Shrishti Elara"


class TestPropertyDetailsSteeringErrors:
    def test_unknown_property_id_steers_rather_than_404ing(self):
        result = get_property_details(999, ["legal"])
        assert "error" in result
        assert "999" in result["error"]

    def test_an_unknown_section_names_the_four_valid_ones(self):
        result = get_property_details(5, ["schools"])
        assert "error" in result
        for section in ("pricing", "possession", "legal", "amenities"):
            assert section in result["error"]

    def test_no_sections_at_all_steers(self):
        assert "error" in get_property_details(5, [])


# --------------------------------------------------------------------------
# update_buyer_profile (D13)
# --------------------------------------------------------------------------

class TestUpdateBuyerProfile:
    def test_the_documented_three_keys_come_back(self):
        result = update_buyer_profile(2, {"budget_max": 6_500_000})
        assert set(result) == {"profile", "unknown", "next_question_hint"}

    def test_the_exit_criterion_unknowns(self):
        result = update_buyer_profile(2, {"budget_max": 6_500_000})
        assert "possession_need" in result["unknown"]
        assert "family_size" in result["unknown"]

    def test_a_captured_field_leaves_the_unknown_list(self):
        assert "budget" in update_buyer_profile(2, {})["unknown"]
        assert "budget" not in update_buyer_profile(2, {"budget_max": 6_500_000})["unknown"]

    def test_unknown_list_follows_the_qualification_order(self):
        unknown = update_buyer_profile(2, {})["unknown"]
        assert unknown == [
            "budget", "preferred_localities", "bhk_need",
            "possession_need", "family_size",
        ]

    def test_next_question_hint_is_the_first_unknown(self):
        result = update_buyer_profile(2, {"budget_max": 6_500_000})
        assert result["next_question_hint"] == "preferred_localities"

    def test_hint_reaches_possession_once_the_earlier_fields_land(self):
        result = update_buyer_profile(2, {
            "budget_min": 5_500_000, "budget_max": 6_500_000,
            "preferred_localities": ["Bopal", "Shela"], "bhk_need": 3,
        })
        assert result["next_question_hint"] == "possession_need"

    def test_hint_is_none_once_everything_is_known(self):
        result = update_buyer_profile(2, {
            "budget_max": 6_500_000, "preferred_localities": ["Bopal"],
            "bhk_need": 3, "possession_need": "2026-12", "family_size": 4,
        })
        assert result["unknown"] == []
        assert result["next_question_hint"] is None

    def test_budget_comes_back_as_a_display_string(self):
        result = update_buyer_profile(2, {"budget_min": 5_500_000, "budget_max": 6_500_000})
        assert result["profile"]["budget"] == "₹55–65 lakh"

    def test_possession_comes_back_as_a_display_string(self):
        result = update_buyer_profile(2, {"possession_need": "2026-12"})
        assert result["profile"]["possession_need"] == "December 2026"

    def test_no_raw_rupees_or_iso_dates_in_the_profile(self):
        result = update_buyer_profile(2, {
            "budget_min": 5_500_000, "budget_max": 6_500_000, "possession_need": "2026-12",
        })
        blob = " ".join(flat_strings(result))
        assert "6500000" not in blob
        assert "2026-12" not in blob

    def test_unknown_fields_are_absent_from_the_profile(self):
        profile = update_buyer_profile(2, {"bhk_need": 3})["profile"]
        assert "budget" not in profile
        assert "family_size" not in profile

    def test_intent_tier_is_always_present_and_never_unknown(self):
        result = update_buyer_profile(2, {})
        assert result["profile"]["intent_tier"] == "cold"
        assert "intent_tier" not in result["unknown"]

    def test_an_empty_updates_object_acts_as_a_read(self, session):
        update_buyer_profile(2, {"bhk_need": 3})
        assert update_buyer_profile(2, {})["profile"]["bhk_need"] == 3
        session.expire_all()
        assert session.get(Buyer, 2).bhk_need == 3

    def test_updates_persist_across_calls(self):
        update_buyer_profile(2, {"budget_max": 6_500_000})
        assert update_buyer_profile(2, {"bhk_need": 3})["profile"]["budget"] == "up to ₹65 lakh"

    def test_the_row_is_actually_written(self, session):
        update_buyer_profile(2, {"family_size": 4, "intent_tier": "warm"})
        session.expire_all()
        buyer = session.get(Buyer, 2)
        assert buyer.family_size == 4
        assert buyer.intent_tier == "warm"

    def test_possession_need_persists_as_a_naive_datetime(self, session):
        update_buyer_profile(2, {"possession_need": "2026-12"})
        session.expire_all()
        stored = session.get(Buyer, 2).possession_need
        assert stored == datetime(2026, 12, 1)
        assert stored.tzinfo is None


class TestUpdateBuyerProfileSteeringErrors:
    def test_an_unknown_locality_names_the_four_valid_ones(self):
        result = update_buyer_profile(2, {"preferred_localities": ["Bopal West"]})
        assert "error" in result
        for locality in LOCALITIES:
            assert locality in result["error"]

    def test_a_rejected_update_writes_nothing(self, session):
        update_buyer_profile(2, {"bhk_need": 3, "preferred_localities": ["Bopal West"]})
        session.expire_all()
        assert session.get(Buyer, 2).bhk_need is None, "a partial write is worse than none"

    def test_an_unknown_field_name_steers(self):
        result = update_buyer_profile(2, {"favourite_colour": "blue"})
        assert "error" in result
        assert "favourite_colour" in result["error"]

    def test_an_out_of_vocabulary_intent_tier_steers(self):
        result = update_buyer_profile(2, {"intent_tier": "boiling"})
        assert "error" in result
        for tier in ("cold", "warm", "hot"):
            assert tier in result["error"]

    def test_a_malformed_possession_month_steers(self):
        result = update_buyer_profile(2, {"possession_need": "next Diwali"})
        assert "error" in result
        assert "YYYY-MM" in result["error"]

    def test_an_inverted_budget_steers(self):
        result = update_buyer_profile(2, {"budget_min": 9_000_000, "budget_max": 5_000_000})
        assert "error" in result

    def test_a_non_integer_budget_steers(self):
        assert "error" in update_buyer_profile(2, {"budget_max": "65 lakh"})

    def test_unknown_buyer_steers(self):
        result = update_buyer_profile(99, {"bhk_need": 3})
        assert "error" in result
        assert "99" in result["error"]


# --------------------------------------------------------------------------
# book_site_visit (D15, D24)
# --------------------------------------------------------------------------

def in_days(days, hour=17):
    when = datetime.now() + timedelta(days=days)
    return when.replace(hour=hour, minute=0, second=0, microsecond=0)


class TestBookSiteVisit:
    def test_a_five_day_slot_confirms(self):
        result = book_site_visit(1, 5, in_days(5).isoformat(sep=" "), "k1")
        assert result["confirmed"] is True

    def test_confirmation_carries_the_documented_fields(self):
        result = book_site_visit(1, 5, in_days(5).isoformat(sep=" "), "k1")
        assert set(result) == {"confirmed", "slot_display", "property"}

    def test_the_slot_is_pre_formatted(self):
        slot = in_days(5)
        result = book_site_visit(1, 5, slot.isoformat(sep=" "), "k1")
        assert result["slot_display"].endswith("5:00 pm")
        assert str(slot.year) in result["slot_display"]
        assert "T" not in result["slot_display"]

    def test_the_property_is_named_with_its_locality(self):
        result = book_site_visit(1, 5, in_days(5).isoformat(sep=" "), "k1")
        assert result["property"] == "Shrishti Elara, Bopal"

    def test_a_row_lands_in_site_visits(self, session):
        book_site_visit(1, 5, in_days(5).isoformat(sep=" "), "k1")
        rows = session.exec(select(SiteVisit)).all()
        assert len(rows) == 1
        assert rows[0].buyer_id == 1 and rows[0].property_id == 5

    def test_the_tool_does_not_escalate_by_itself(self, session):
        # D15: Phase 4's evaluator picks the booking up post-turn.
        book_site_visit(1, 5, in_days(5).isoformat(sep=" "), "k1")
        assert session.exec(select(Escalation)).all() == []
        assert session.get(Conversation, 1).status == "active"


class TestBookSiteVisitValidation:
    def test_a_natural_language_slot_steers(self):
        result = book_site_visit(1, 5, "kal shaam", "k1")
        assert "error" in result
        assert "kal shaam" in result["error"]
        assert "2026-08-09 17:00" in result["error"]

    def test_a_natural_language_slot_books_nothing(self, session):
        book_site_visit(1, 5, "kal shaam", "k1")
        assert session.exec(select(SiteVisit)).all() == []

    def test_a_past_slot_steers(self):
        result = book_site_visit(1, 5, in_days(-1).isoformat(sep=" "), "k1")
        assert "error" in result
        assert "future" in result["error"]

    def test_twenty_days_out_is_rejected(self):
        result = book_site_visit(1, 5, in_days(20).isoformat(sep=" "), "k1")
        assert "error" in result
        assert "14 days" in result["error"]

    def test_thirteen_days_out_is_accepted(self):
        assert book_site_visit(1, 5, in_days(13).isoformat(sep=" "), "k1")["confirmed"] is True

    def test_an_offset_is_stripped_and_the_wall_clock_kept(self, session):
        # D24: naive IST throughout. The wall clock is kept, not converted.
        slot = in_days(5)
        book_site_visit(1, 5, slot.isoformat(sep=" ") + "-04:00", "k1")
        stored = session.exec(select(SiteVisit)).all()[0].slot
        assert stored.hour == 17
        assert stored.tzinfo is None

    def test_unknown_buyer_steers(self):
        assert "error" in book_site_visit(99, 5, in_days(5).isoformat(sep=" "), "k1")

    def test_unknown_property_steers(self):
        assert "error" in book_site_visit(1, 999, in_days(5).isoformat(sep=" "), "k1")


class TestBookSiteVisitIdempotency:
    def test_the_same_key_twice_returns_one_booking(self, session):
        first = book_site_visit(1, 5, in_days(5).isoformat(sep=" "), "k1")
        second = book_site_visit(1, 5, in_days(5).isoformat(sep=" "), "k1")
        assert first == second
        assert len(session.exec(select(SiteVisit)).all()) == 1

    def test_a_repeat_returns_the_existing_booking_unchanged(self, session):
        original = book_site_visit(1, 5, in_days(5).isoformat(sep=" "), "k1")
        repeat = book_site_visit(1, 7, in_days(6).isoformat(sep=" "), "k1")
        assert repeat == original
        assert len(session.exec(select(SiteVisit)).all()) == 1

    def test_a_different_key_books_again(self, session):
        book_site_visit(1, 5, in_days(5).isoformat(sep=" "), "k1")
        book_site_visit(1, 7, in_days(6).isoformat(sep=" "), "k2")
        assert len(session.exec(select(SiteVisit)).all()) == 2


# --------------------------------------------------------------------------
# escalate_to_broker
# --------------------------------------------------------------------------

class TestEscalateToBroker:
    def test_an_escalation_row_is_written(self, session):
        escalate_to_broker(1, "buyer asked for a discount", "high")
        rows = session.exec(select(Escalation)).all()
        assert len(rows) == 1
        assert rows[0].conversation_id == 1
        assert rows[0].reason == "buyer asked for a discount"
        assert rows[0].urgency == "high"

    def test_the_brief_is_left_empty_for_phase_four(self, session):
        escalate_to_broker(1, "site visit booked", "high")
        assert session.exec(select(Escalation)).all()[0].brief == ""

    def test_the_conversation_becomes_escalated(self, session):
        escalate_to_broker(1, "site visit booked", "high")
        session.expire_all()
        assert session.get(Conversation, 1).status == "escalated"

    def test_the_result_identifies_the_row_phase_four_must_fill(self, session):
        result = escalate_to_broker(1, "site visit booked", "high")
        assert result["escalated"] is True
        assert result["escalation_id"] == session.exec(select(Escalation)).all()[0].id

    def test_an_out_of_vocabulary_urgency_steers(self, session):
        result = escalate_to_broker(1, "whatever", "extreme")
        assert "error" in result
        for urgency in ("low", "medium", "high"):
            assert urgency in result["error"]
        assert session.exec(select(Escalation)).all() == []

    def test_unknown_conversation_steers(self, session):
        assert "error" in escalate_to_broker(99, "whatever", "high")
        assert session.exec(select(Escalation)).all() == []
