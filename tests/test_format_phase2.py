"""The display strings Phase 2 tools add on top of Phase 1's three (D4).

Same contract: these land verbatim in `tool_calls.result`, the model quotes them
verbatim, and the Phase 4 guard matches them byte-for-byte.
"""
from datetime import datetime

import pytest

from app.format import (
    amenities_display,
    budget_display,
    month_display,
    slot_display,
    status_display,
)


class TestBudgetDisplay:
    def test_a_range_inside_one_unit_names_the_unit_once(self):
        assert budget_display(5_500_000, 6_500_000) == "₹55–65 lakh"

    def test_range_uses_an_en_dash_not_a_hyphen(self):
        assert "–" in budget_display(5_500_000, 6_500_000)
        assert "-" not in budget_display(5_500_000, 6_500_000)

    def test_a_range_crossing_units_spells_both_out(self):
        assert budget_display(8_500_000, 12_500_000) == "₹85 lakh–₹1.25 crore"

    def test_a_ceiling_alone_reads_as_up_to(self):
        assert budget_display(None, 6_500_000) == "up to ₹65 lakh"

    def test_a_floor_alone_reads_as_from(self):
        assert budget_display(5_500_000, None) == "from ₹55 lakh"

    def test_equal_bounds_collapse_to_one_figure(self):
        assert budget_display(6_500_000, 6_500_000) == "₹65 lakh"

    def test_nothing_known_is_none(self):
        assert budget_display(None, None) is None


class TestMonthDisplay:
    def test_month_is_spelled_in_full(self):
        assert month_display(datetime(2026, 12, 1)) == "December 2026"

    def test_agrees_with_possession_display(self):
        from app.format import possession_display
        d = datetime(2027, 3, 1)
        assert month_display(d) == possession_display(d, "under_construction")


class TestStatusDisplay:
    @pytest.mark.parametrize("status,expected", [
        ("ready_to_move", "Ready to move"),
        ("under_construction", "Under construction"),
    ])
    def test_reads_as_a_sentence_not_an_enum(self, status, expected):
        assert status_display(status) == expected

    def test_rejects_an_unknown_status(self):
        with pytest.raises(ValueError):
            status_display("nearly_done")


class TestSlotDisplay:
    def test_matches_the_spec_example(self):
        # Phase 2 §5 prints "Saturday, 9 August 2026, 5:00 pm"; 2026-08-09 is a
        # Sunday. The weekday is computed, never transcribed from the doc.
        assert slot_display(datetime(2026, 8, 9, 17, 0)) == "Sunday, 9 August 2026, 5:00 pm"

    def test_weekday_and_month_are_spelled_in_full(self):
        assert slot_display(datetime(2026, 1, 5, 9, 30)) == "Monday, 5 January 2026, 9:30 am"

    def test_noon_and_midnight_do_not_render_as_zero(self):
        assert slot_display(datetime(2026, 8, 9, 12, 0)) == "Sunday, 9 August 2026, 12:00 pm"
        assert slot_display(datetime(2026, 8, 9, 0, 30)) == "Sunday, 9 August 2026, 12:30 am"

    def test_day_of_month_carries_no_leading_zero(self):
        rendered = slot_display(datetime(2026, 8, 9, 17, 0))
        assert "9 August" in rendered
        assert "09 August" not in rendered


class TestAmenitiesDisplay:
    def test_underscores_become_spaces_and_the_list_reads_as_prose(self):
        assert amenities_display(
            ["clubhouse", "gym", "covered_parking"]
        ) == "clubhouse, gym, covered parking"

    def test_order_is_preserved(self):
        assert amenities_display(["gym", "clubhouse"]) == "gym, clubhouse"

    def test_numeric_amenity_survives_intact(self):
        assert amenities_display(["24x7_security"]) == "24x7 security"

    def test_empty_list_is_an_empty_string(self):
        assert amenities_display([]) == ""
