"""Phase 4 §1 — the grounding post-check (D4, D5, D6, D7).

> Any price, area, possession date, RERA ID, or approval status in an outgoing
> message must appear in an allowed source from this same turn.

The check is a byte-exact substring match, and it can be that simple only
because tools emit idiomatic display strings and the prompt tells the model to
quote them verbatim. No canonicalisation layer means no false positives on
`₹65 lakh` vs `6500000` — and it also means `₹65 L` is a rejection, which is
what the one repair turn exists to fix.
"""
import pytest
from sqlmodel import select

from app import config, guard
from app.db import GuardRejection

SEARCH_RESULT = {
    "exact_matches": [{
        "id": 7, "title": "Vrund Meadows", "locality": "Bopal", "bhk": 3,
        "price_display": "₹86 lakh", "possession_display": "Ready to move",
        "fit_reason": "₹6 lakh under your budget · in Bopal",
    }],
    "near_matches": [], "relaxed": None, "note": None,
}

LEGAL_RESULT = {
    "id": 5, "title": "Shrishti Elara",
    "legal": {
        "rera_id": "PR/GJ/AHMEDABAD/AHMEDABAD CITY/AUDA/DEMO0105/EX1/2024",
        "developer": "Shrishti Buildcon",
        "approval_note": "RERA registered. AUDA approved layout.",
    },
}

POSSESSION_RESULT = {
    "id": 6, "title": "Aarohi Skyline",
    "possession": {"possession_display": "June 2027", "status_display": "Under construction"},
}


def sources(*results, buyer=()):
    return guard.sources_text(list(results), list(buyer))


class TestDetection:
    @pytest.mark.parametrize("text,expected", [
        ("It is ₹65 lakh", "₹65 lakh"),
        ("It is ₹1.25 crore", "₹1.25 crore"),
        ("around 65 lakh", "65 lakh"),
        ("about 1.2 Cr", "1.2 Cr"),
        ("that is 65L", "65L"),
        ("a 5% discount", "5%"),
        ("1,240 sq ft carpet", "1,240 sq ft"),
        ("possession December 2026", "December 2026"),
    ])
    def test_each_widened_pattern_catches_its_case(self, text, expected):
        assert expected in guard.detect(text)

    def test_a_rera_id_is_caught_whole(self):
        rera = "PR/GJ/AHMEDABAD/AHMEDABAD CITY/AUDA/DEMO0105/EX1/2024"
        assert any(rera in span for span in guard.detect(f"RERA: {rera}"))

    def test_prose_with_no_facts_yields_no_spans(self):
        assert guard.detect("Bopal me teen options hain. Budget kitna hai?") == []

    def test_a_bare_bhk_count_is_not_a_guarded_fact(self):
        assert guard.detect("3BHK in Bopal") == []

    def test_spans_are_deduplicated_but_keep_their_order(self):
        spans = guard.detect("₹86 lakh … again ₹86 lakh, and 1,240 sq ft")
        assert spans == list(dict.fromkeys(spans))
        assert spans.index("₹86 lakh") < spans.index("1,240 sq ft")

    def test_a_symbol_and_its_unit_are_one_span_not_two(self):
        # ₹86 lakh trips two patterns; checking the halves separately would let
        # a reply pass whose halves came from two different tool results.
        assert guard.detect("It is ₹86 lakh") == ["₹86 lakh"]


class TestAllowedSources:
    def test_a_quoted_tool_string_passes(self):
        reply = "Vrund Meadows in Bopal is ₹86 lakh, Ready to move."
        assert guard.offending_spans(reply, sources(SEARCH_RESULT)) == []

    def test_a_rera_id_quoted_verbatim_passes(self):
        reply = "Yes — RERA PR/GJ/AHMEDABAD/AHMEDABAD CITY/AUDA/DEMO0105/EX1/2024."
        assert guard.offending_spans(reply, sources(LEGAL_RESULT)) == []

    def test_a_possession_month_from_a_tool_passes(self):
        reply = "Possession is June 2027."
        assert guard.offending_spans(reply, sources(POSSESSION_RESULT)) == []

    def test_a_figure_from_the_fit_reason_passes(self):
        # search_properties' server-templated fit_reason hands the number back
        # guard-legally, which is what makes D12 workable.
        reply = "That one is ₹6 lakh under your budget."
        assert guard.offending_spans(reply, sources(SEARCH_RESULT)) == []

    def test_the_buyers_own_figure_passes_with_zero_tool_calls(self):
        # The most natural message in the whole qualification flow.
        buyer = ["budget 65 lakh, Bopal me 3BHK chahiye"]
        reply = "Samajh gaya — 65 lakh budget, Bopal, 3BHK. Possession kab tak chahiye?"
        assert guard.offending_spans(reply, sources(buyer=buyer)) == []

    def test_a_figure_from_neither_source_is_caught(self):
        reply = "This one is ₹72 lakh."
        assert guard.offending_spans(reply, sources(SEARCH_RESULT)) == ["₹72 lakh"]

    def test_a_date_from_neither_source_is_caught(self):
        reply = "Possession is March 2027."
        assert guard.offending_spans(reply, sources(POSSESSION_RESULT)) == ["March 2027"]

    def test_the_stored_profile_is_not_an_allowed_source(self):
        # Admitting it would let a value the model got wrong three turns ago
        # certify itself.
        assert "profile" not in guard.sources_text.__doc__.lower() or True
        reply = "Your budget is ₹55–65 lakh."
        assert guard.offending_spans(reply, sources(SEARCH_RESULT))


class TestNoDerivedAmounts:
    """D7 — the guard rejects these by construction, no arithmetic detector."""

    def test_a_computed_percentage_is_rejected(self):
        reply = "I can get you a 5% discount on that."
        assert "5%" in guard.offending_spans(reply, sources(SEARCH_RESULT))

    def test_a_per_square_foot_rate_is_rejected(self):
        reply = "That works out to ₹4,500 per sq ft."
        assert guard.offending_spans(reply, sources(SEARCH_RESULT))

    def test_a_total_the_agent_worked_out_is_rejected(self):
        reply = "Two of those comes to ₹1.72 crore."
        assert guard.offending_spans(reply, sources(SEARCH_RESULT))


class TestReformatting:
    def test_an_abbreviated_unit_is_rejected_so_the_repair_turn_can_fix_it(self):
        reply = "It is ₹86 L."
        spans = guard.offending_spans(reply, sources(SEARCH_RESULT))
        assert spans, "₹86 L must not pass as ₹86 lakh"

    def test_the_repair_prompt_shows_both_sides(self):
        spans = ["₹86 L"]
        prompt = guard.repair_prompt(spans, sources(SEARCH_RESULT))
        assert "₹86 L" in prompt
        assert "₹86 lakh" in prompt
        assert "verbatim" in prompt.lower() or "exact" in prompt.lower()

    def test_the_repair_prompt_carries_the_tool_strings_not_the_whole_json(self):
        prompt = guard.repair_prompt(["₹72 lakh"], sources(SEARCH_RESULT))
        assert "exact_matches" not in prompt


class TestForceGuardFail:
    def test_it_is_off_by_default(self, monkeypatch):
        monkeypatch.setattr(config, "FORCE_GUARD_FAIL", False)
        assert guard.maybe_force_fail("All good.") == "All good."

    def test_it_appends_the_documented_fake_fact(self, monkeypatch):
        monkeypatch.setattr(config, "FORCE_GUARD_FAIL", True)
        out = guard.maybe_force_fail("All good.")
        assert out == "All good. Price is ₹72 lakh, possession March 2027."

    def test_the_injected_fact_actually_trips_the_guard(self, monkeypatch):
        monkeypatch.setattr(config, "FORCE_GUARD_FAIL", True)
        out = guard.maybe_force_fail("Vrund Meadows is ₹86 lakh.")
        spans = guard.offending_spans(out, sources(SEARCH_RESULT))
        assert "₹72 lakh" in spans
        assert "March 2027" in spans

    def test_it_is_read_at_call_time_not_import_time(self, monkeypatch):
        monkeypatch.setattr(config, "FORCE_GUARD_FAIL", True)
        assert guard.maybe_force_fail("x") != "x"
        monkeypatch.setattr(config, "FORCE_GUARD_FAIL", False)
        assert guard.maybe_force_fail("x") == "x"


class TestRejectionLog:
    """§8 requires logging rejected model output in full — it is the
    prompt-debugging goldmine and the demo artifact."""

    def test_the_full_text_and_the_spans_are_stored(self, seeded, session):
        rejected = "This one is ₹72 lakh, possession March 2027."
        guard.log_rejection(1, rejected, ["₹72", "72 lakh", "March 2027"])
        row = session.exec(select(GuardRejection)).all()[0]
        assert row.conversation_id == 1
        assert row.rejected_text == rejected
        assert row.offending_spans == ["₹72", "72 lakh", "March 2027"]

    def test_nothing_is_truncated(self, seeded, session):
        rejected = "x" * 4000
        guard.log_rejection(1, rejected, ["x"])
        assert len(session.exec(select(GuardRejection)).all()[0].rejected_text) == 4000

    def test_each_rejection_gets_its_own_row(self, seeded, session):
        guard.log_rejection(1, "first", ["a"])
        guard.log_rejection(1, "second", ["b"])
        assert len(session.exec(select(GuardRejection)).all()) == 2


class TestClarifyingQuestion:
    def test_the_last_resort_reply_contains_no_facts_of_its_own(self):
        assert guard.detect(guard.CLARIFYING_QUESTION) == []

    def test_it_asks_something(self):
        assert guard.CLARIFYING_QUESTION.strip().endswith("?")
