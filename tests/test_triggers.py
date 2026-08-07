"""Phase 4 §2 — the deterministic tier (D18).

What this tier guarantees is not that it runs before the LLM — the brief-writer
call runs on this path too — but that the escalation *decision* is deterministic.
Same text, same verdict, every time, with no model in the loop.
"""
import pytest

from app import triggers
from tests.fixtures.triggers import MUST_ESCALATE, MUST_NOT_ESCALATE


class TestMustNotEscalate:
    """The expensive half. Run it first and keep it first."""

    @pytest.mark.parametrize("text", MUST_NOT_ESCALATE)
    def test_an_ordinary_message_is_left_alone(self, text):
        assert triggers.match(text) is None, f"false positive on {text!r}"


class TestMustEscalate:
    @pytest.mark.parametrize("text", MUST_ESCALATE)
    def test_a_request_for_action_or_advice_fires(self, text):
        assert triggers.match(text) is not None, f"missed {text!r}"


class TestTheSpecCases:
    def test_the_headline_pair(self):
        assert triggers.match("price kam karo bhai") == "negotiation_request"
        assert triggers.match("loan lunga, budget 65L") is None

    def test_kam_se_kam_is_a_quantity_not_a_haggle(self):
        assert triggers.match("kam se kam 3BHK chahiye") is None

    def test_reporting_someone_elses_discount_is_not_asking_for_one(self):
        assert triggers.match("builder ne discount diya tha pichle project me") is None
        assert triggers.match("thoda discount milega?") == "negotiation_request"


class TestWordBoundaries:
    """`kam` hides in `kaam`, `rate` in `accurate` and `corporate`."""

    @pytest.mark.parametrize("text", [
        "accurate carpet area batao",
        "corporate lease pe hai kya",
        "kaam chalu hai site pe",
        "iska rate accurate batao",
    ])
    def test_a_substring_never_fires(self, text):
        assert triggers.match(text) is None

    def test_every_pattern_is_anchored(self):
        for pattern in triggers.compiled_patterns():
            assert r"\b" in pattern.pattern, pattern.pattern


class TestCategories:
    @pytest.mark.parametrize("text,category", [
        ("price kam karo bhai", "negotiation_request"),
        ("best rate batao", "negotiation_request"),
        ("EMI kitna banega", "loan_advice_request"),
        ("kaunsa bank loan dega?", "loan_advice_request"),
        ("ye to dhokha hai", "legal_dispute"),
        ("aadmi se baat karao", "human_requested"),
    ])
    def test_the_category_names_what_the_broker_inherits(self, text, category):
        assert triggers.match(text) == category

    def test_every_category_is_reachable(self):
        found = {triggers.match(t) for t in MUST_ESCALATE}
        assert found == set(triggers.CATEGORIES)

    def test_the_urgency_is_fixed_not_guessed(self):
        assert triggers.TRIGGER_URGENCY == "high"


class TestDeterminism:
    def test_the_same_text_always_gets_the_same_verdict(self):
        for text in MUST_ESCALATE + MUST_NOT_ESCALATE:
            assert len({triggers.match(text) for _ in range(5)}) == 1

    def test_case_does_not_change_the_verdict(self):
        assert triggers.match("PRICE KAM KARO BHAI") == "negotiation_request"
        assert triggers.match("Price Kam Karo Bhai") == "negotiation_request"

    def test_it_runs_on_the_coalesced_text_not_per_fragment(self):
        # Debounce joins fragments with newlines; a trigger split across two
        # WhatsApp messages must still be seen.
        assert triggers.match("bhai\nprice kam karo") == "negotiation_request"


class TestNoModelInvolved:
    def test_the_module_imports_nothing_that_talks_to_a_model(self):
        source = open(triggers.__file__, encoding="utf-8").read()
        for forbidden in ("openai", "llm", "httpx", "requests"):
            assert forbidden not in source, f"{forbidden} does not belong in triggers.py"
