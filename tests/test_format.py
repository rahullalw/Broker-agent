"""D4 — the display strings the guard checks byte-for-byte.

If these do not read exactly as an Ahmedabad broker would write them, the model
will reformat them and the grounding guard will start rejecting good turns.
"""
from datetime import datetime

import pytest

from app.format import area_display, possession_display, price_display


class TestPriceDisplay:
    @pytest.mark.parametrize("rupees,expected", [
        (8_500_000, "₹85 lakh"),        # the exit-criteria case
        (4_800_000, "₹48 lakh"),
        (6_250_000, "₹62.5 lakh"),      # one decimal, no trailing zero
        (4_200_000, "₹42 lakh"),
        (9_900_000, "₹99 lakh"),
        (12_500_000, "₹1.25 crore"),    # the exit-criteria case
        (12_000_000, "₹1.2 crore"),     # 1.20 -> 1.2
        (10_000_000, "₹1 crore"),       # exact boundary, no ".0"
        (22_000_000, "₹2.2 crore"),
        (15_000_000, "₹1.5 crore"),
    ])
    def test_reads_like_a_broker_wrote_it(self, rupees, expected):
        assert price_display(rupees) == expected

    def test_crore_boundary_is_inclusive(self):
        assert price_display(9_999_998) == "₹1 crore"
        assert price_display(10_000_000) == "₹1 crore"
        assert price_display(9_900_000) == "₹99 lakh"

    def test_just_under_a_crore_promotes_rather_than_saying_100_lakh(self):
        # 99.99999 lakh rounds to 100.00 lakh; "₹100 lakh" is not something
        # anyone says, so it must become "₹1 crore".
        assert price_display(9_999_999) == "₹1 crore"

    def test_two_decimals_maximum(self):
        assert price_display(6_255_000) == "₹62.55 lakh"
        assert price_display(6_255_500) == "₹62.56 lakh"   # rounded, not truncated

    def test_never_emits_raw_digits_group_separators_or_rupee_paise(self):
        for rupees in (4_800_000, 12_500_000):
            rendered = price_display(rupees)
            assert str(rupees) not in rendered
            assert "," not in rendered
            assert rendered.startswith("₹")

    def test_rejects_non_positive_amounts(self):
        with pytest.raises(ValueError):
            price_display(0)
        with pytest.raises(ValueError):
            price_display(-100)


class TestPossessionDisplay:
    def test_ready_to_move_ignores_the_date(self):
        assert possession_display(datetime(2024, 3, 1), "ready_to_move") == "Ready to move"

    def test_under_construction_spells_the_month_in_full(self):
        assert possession_display(datetime(2026, 12, 1), "under_construction") == "December 2026"

    @pytest.mark.parametrize("month,expected", [
        (1, "January 2027"), (3, "March 2027"), (6, "June 2027"), (9, "September 2027"),
    ])
    def test_month_names_are_never_abbreviated(self, month, expected):
        assert possession_display(datetime(2027, month, 15), "under_construction") == expected

    def test_never_emits_an_iso_date(self):
        rendered = possession_display(datetime(2026, 12, 1), "under_construction")
        assert "-" not in rendered
        assert "2026-12" not in rendered

    def test_rejects_unknown_status(self):
        with pytest.raises(ValueError):
            possession_display(datetime(2026, 12, 1), "possibly_someday")


class TestAreaDisplay:
    @pytest.mark.parametrize("sqft,expected", [
        (1240, "1,240 sq ft"),
        (980, "980 sq ft"),
        (1_050, "1,050 sq ft"),
        (2_400, "2,400 sq ft"),
    ])
    def test_groups_thousands_and_names_the_unit(self, sqft, expected):
        assert area_display(sqft) == expected

    def test_unit_is_lowercase_sq_ft_not_sqft(self):
        assert area_display(1240).endswith(" sq ft")

    def test_rejects_non_positive_area(self):
        with pytest.raises(ValueError):
            area_display(0)
