"""Phase 3 §2 — the system prompt.

Seven static blocks plus one dynamic block rebuilt every turn. The dynamic block
is the load-bearing one: text-only transcript replay (D11) means the agent has no
memory of the profile across turns, so without it the agent re-asks what it
already knows and the qualification flow breaks (D12).
"""
from datetime import datetime

import pytest

from app import guard, prompt, seed


@pytest.fixture
def qualified():
    """The exact shape `update_buyer_profile` returns — one source of truth."""
    return {
        "profile": {
            "budget": "₹55–65 lakh",
            "localities": ["Bopal", "Shela"],
            "bhk_need": 3,
            "intent_tier": "warm",
        },
        "unknown": ["possession_need", "family_size"],
        "next_question_hint": "possession_need",
    }


@pytest.fixture
def blank():
    return {
        "profile": {"intent_tier": "cold"},
        "unknown": [
            "budget", "preferred_localities", "bhk_need",
            "possession_need", "family_size",
        ],
        "next_question_hint": "budget",
    }


class TestRole:
    def test_it_names_the_business_and_the_city(self):
        assert "AllSet" in prompt.SYSTEM_PROMPT
        assert "Ahmedabad" in prompt.SYSTEM_PROMPT

    def test_it_says_first_touch_and_whatsapp_length(self):
        lowered = prompt.SYSTEM_PROMPT.lower()
        assert "first" in lowered
        assert "whatsapp" in lowered


class TestGrounding:
    def test_verbatim_quoting_is_stated_as_an_absolute(self):
        lowered = prompt.SYSTEM_PROMPT.lower()
        assert "verbatim" in lowered
        assert "never reformat" in lowered

    def test_a_fact_without_a_tool_call_this_turn_is_forbidden(self):
        lowered = prompt.SYSTEM_PROMPT.lower()
        assert "did not come from a tool" in lowered or "not come from a tool" in lowered

    def test_the_prompt_carries_no_inventory(self):
        # The agent only *has* those facts via tools (§8, enforcement 2).
        for property_row in seed.PROPERTIES:
            assert property_row["title"] not in prompt.SYSTEM_PROMPT
            assert property_row["rera_id"] not in prompt.SYSTEM_PROMPT
        assert "₹" not in prompt.SYSTEM_PROMPT


class TestQualificationOrder:
    def test_the_five_steps_appear_in_order(self):
        lowered = prompt.SYSTEM_PROMPT.lower()
        positions = [
            lowered.index("budget"), lowered.index("locality"),
            lowered.index("bhk"), lowered.index("possession"),
            lowered.index("family size"),
        ]
        assert positions == sorted(positions)

    def test_one_question_per_message(self):
        assert "One question per message" in prompt.SYSTEM_PROMPT


class TestProfileHandling:
    def test_stored_figures_are_for_tool_arguments_not_for_restating(self):
        lowered = prompt.SYSTEM_PROMPT.lower()
        assert "do not restate" in lowered
        assert "tool arguments" in lowered or "tool argument" in lowered

    def test_a_number_shown_to_the_buyer_must_come_from_a_tool_this_turn(self):
        assert "from a tool this turn" in prompt.SYSTEM_PROMPT.lower()


class TestEscalationTriggers:
    def test_the_triggers_are_listed_plainly(self):
        lowered = prompt.SYSTEM_PROMPT.lower()
        for trigger in ("site visit", "negotiat", "legal", "human"):
            assert trigger in lowered

    def test_the_escalation_tool_is_named(self):
        assert "escalate_to_broker" in prompt.SYSTEM_PROMPT


class TestNeverDo:
    @pytest.mark.parametrize("forbidden", [
        "negotiate", "loan", "legal opinion", "promise",
        "discount", "percentage", "per sq ft", "total",
    ])
    def test_each_forbidden_action_is_named(self, forbidden):
        assert forbidden in prompt.SYSTEM_PROMPT.lower()

    def test_derived_amounts_are_forbidden_explicitly(self):
        # D7 — the guard rejects these by construction, the prompt stops the
        # model reaching for them in the first place.
        lowered = prompt.SYSTEM_PROMPT.lower()
        assert "no discounts" in lowered
        assert "per sq ft" in lowered

    def test_prompt_injection_is_addressed_without_drama(self):
        lowered = prompt.SYSTEM_PROMPT.lower()
        assert "just messages" in lowered
        assert "no tools for these" in lowered


class TestStyle:
    def test_length_and_language_are_pinned(self):
        lowered = prompt.SYSTEM_PROMPT.lower()
        assert "2-3 sentences" in lowered or "2–3 sentences" in lowered
        assert "hinglish" in lowered

    def test_no_bullet_lists_over_chat(self):
        assert "bullet" in prompt.SYSTEM_PROMPT.lower()


class TestTokenBudget:
    def test_the_static_blocks_stay_near_the_650_token_target(self):
        # ~4 characters per token. The budget protects instruction adherence,
        # not cost — a prompt twice this long is one the model skims.
        estimated = len(prompt.SYSTEM_PROMPT) / 4
        assert 300 < estimated < 900, estimated

    def test_the_dynamic_block_is_small(self, qualified):
        assert len(prompt.profile_block(qualified)) / 4 < 80


class TestDynamicBlock:
    def test_it_reads_like_the_spec_example(self, qualified):
        block = prompt.profile_block(qualified)
        assert block == (
            "Known — budget ₹55–65 lakh, localities Bopal/Shela, 3BHK.\n"
            "Unknown — possession timeline, family size.\n"
            "Ask about the first unknown."
        )

    def test_localities_are_slash_joined(self, qualified):
        assert "Bopal/Shela" in prompt.profile_block(qualified)

    def test_unknown_fields_read_as_english_not_column_names(self, qualified):
        block = prompt.profile_block(qualified)
        assert "possession_need" not in block
        assert "family_size" not in block

    def test_an_empty_profile_says_so(self, blank):
        block = prompt.profile_block(blank)
        assert block.startswith("Known — nothing yet.")
        assert "budget" in block

    def test_a_complete_profile_asks_for_nothing(self):
        block = prompt.profile_block({
            "profile": {
                "budget": "₹55–65 lakh", "localities": ["Bopal"], "bhk_need": 3,
                "possession_need": "December 2026", "family_size": 4,
                "intent_tier": "hot",
            },
            "unknown": [],
            "next_question_hint": None,
        })
        assert "Unknown — nothing." in block
        assert "Ask about the first unknown." not in block

    def test_possession_and_family_size_read_naturally(self):
        block = prompt.profile_block({
            "profile": {"possession_need": "December 2026", "family_size": 4,
                        "intent_tier": "warm"},
            "unknown": ["budget"],
            "next_question_hint": "budget",
        })
        assert "possession by December 2026" in block
        assert "family of 4" in block

    def test_intent_tier_stays_out_of_the_block(self, qualified):
        assert "warm" not in prompt.profile_block(qualified)

    def test_a_steering_error_instead_of_a_profile_does_not_crash(self):
        block = prompt.profile_block({"error": "No buyer with id 99."})
        assert isinstance(block, str)


class TestAssembly:
    def test_the_turn_prompt_carries_both_halves(self, qualified):
        assembled = prompt.build_system_prompt(qualified)
        assert prompt.SYSTEM_PROMPT in assembled
        assert prompt.profile_block(qualified) in assembled

    def test_the_dynamic_block_comes_last(self, qualified):
        assembled = prompt.build_system_prompt(qualified)
        assert assembled.endswith(prompt.profile_block(qualified))

    def test_it_is_rebuilt_from_whatever_it_is_given(self, qualified, blank):
        assert prompt.build_system_prompt(qualified) != prompt.build_system_prompt(blank)


class TestIdentity:
    """The tools take a buyer_id and a conversation_id. Nothing else in the
    prompt tells the agent what they are, so a live model guesses — and writes
    another buyer's profile."""

    def test_the_ids_are_stated_when_they_are_known(self, qualified):
        block = prompt.profile_block(qualified, buyer_id=3, conversation_id=7)
        assert "buyer_id 3" in block
        assert "conversation_id 7" in block

    def test_identity_comes_before_the_known_line(self, qualified):
        block = prompt.profile_block(qualified, buyer_id=3, conversation_id=7)
        assert block.index("buyer_id 3") < block.index("Known —")

    def test_the_block_is_unchanged_when_no_ids_are_given(self, qualified):
        assert "buyer_id" not in prompt.profile_block(qualified)

    def test_the_assembled_prompt_carries_them(self, qualified):
        assembled = prompt.build_system_prompt(qualified, buyer_id=3, conversation_id=7)
        assert "buyer_id 3" in assembled
        assert "conversation_id 7" in assembled


class TestTodayBlock:
    """The agent has no clock of its own, and `book_site_visit` rejects anything
    in the past or beyond 14 days (D24). Without today's date, "this Saturday"
    is unanswerable and the scripted demo's booking turn dies on a steering
    error the model cannot resolve."""

    def test_it_states_the_weekday_day_month_and_year(self):
        block = prompt.today_block(datetime(2026, 8, 8, 11, 30))
        assert "Saturday, 8 August 2026" in block

    def test_the_month_is_spelled_out_not_locale_formatted(self):
        # Same reason format.py spells its months: strftime follows the process
        # locale and would quietly emit "agosto" on a differently set-up machine.
        assert "December" in prompt.today_block(datetime(2026, 12, 1))

    def test_it_names_the_format_book_site_visit_accepts(self):
        block = prompt.today_block()
        assert "YYYY-MM-DD HH:MM" in block
        assert "14 days" in block

    def test_it_forbids_saying_the_date_back_to_the_buyer(self):
        # A month-and-year is a guarded span (D6). Today's date came from no
        # tool, so quoting it would be a rejection waiting to happen.
        assert "Do not state today's date" in prompt.today_block()

    def test_it_defaults_to_now(self):
        assert str(datetime.now().year) in prompt.today_block()

    def test_it_is_small(self):
        assert len(prompt.today_block()) / 4 < 100


class TestTodayInTheAssembledPrompt:
    def test_the_assembled_prompt_carries_it(self, qualified):
        assembled = prompt.build_system_prompt(qualified, now=datetime(2026, 8, 8))
        assert "Saturday, 8 August 2026" in assembled

    def test_it_sits_between_the_static_blocks_and_the_profile(self, qualified):
        assembled = prompt.build_system_prompt(qualified, now=datetime(2026, 8, 8))
        assert assembled.index(prompt.SYSTEM_PROMPT) < assembled.index("TODAY")
        assert assembled.index("TODAY") < assembled.index("Known —")

    def test_the_profile_block_is_still_last(self, qualified):
        assembled = prompt.build_system_prompt(qualified)
        assert assembled.endswith(prompt.profile_block(qualified))

    def test_the_static_prompt_itself_carries_no_date(self):
        # It is a constant; a date baked into it would be wrong by tomorrow.
        assert "TODAY" not in prompt.SYSTEM_PROMPT


class TestShownBlock:
    """D11 throws away tool results, so by the next turn the ids
    `search_properties` returned are gone. A model asked about "the second one"
    does not decline — it guesses an id, then reports the wrong building's
    possession date and the wrong building's RERA number, with the guard
    agreeing to every word because every word did come out of a tool."""

    @pytest.fixture
    def cards(self):
        return [
            {"id": 7, "title": "Vrund Meadows", "locality": "Bopal"},
            {"id": 5, "title": "Shrishti Elara", "locality": "Bopal"},
        ]

    def test_it_lists_id_title_and_locality(self, cards):
        block = prompt.shown_block(cards)
        assert "#7 Vrund Meadows (Bopal)" in block
        assert "#5 Shrishti Elara (Bopal)" in block

    def test_it_tells_the_agent_never_to_guess_an_id(self, cards):
        block = prompt.shown_block(cards)
        assert "never guess one" in block
        assert "search_properties" in block

    def test_nothing_shown_means_no_block_at_all(self):
        assert prompt.shown_block([]) == ""

    def test_it_carries_no_guarded_facts(self, cards):
        """A price or a date reachable from the prompt is a figure the agent can
        quote with no tool call behind it (D6)."""
        block = prompt.shown_block(cards)
        assert guard.detect(block) == []

    def test_it_stays_small(self, cards):
        assert len(prompt.shown_block(cards * 4)) / 4 < 150


class TestShownInTheAssembledPrompt:
    def test_it_appears_when_there_is_something_to_show(self, qualified):
        assembled = prompt.build_system_prompt(
            qualified, shown=[{"id": 7, "title": "Vrund Meadows", "locality": "Bopal"}]
        )
        assert "#7 Vrund Meadows" in assembled

    def test_it_is_absent_entirely_when_nothing_has_been_shown(self, qualified):
        assert "SHOWN SO FAR" not in prompt.build_system_prompt(qualified, shown=[])

    def test_it_does_not_leave_a_blank_gap_behind(self, qualified):
        assembled = prompt.build_system_prompt(qualified, shown=[])
        assert "\n\n\n" not in assembled

    def test_the_profile_block_is_still_last(self, qualified):
        assembled = prompt.build_system_prompt(
            qualified, shown=[{"id": 7, "title": "X", "locality": "Bopal"}]
        )
        assert assembled.endswith(prompt.profile_block(qualified))


class TestEscalateBlockIsNarrow:
    """The model's discretionary channel has to agree with the two authorities
    around it: the regex tier (D18) fires on *asks*, not mentions, and the
    evaluator (D16) owns the confirmed-visit and qualification-complete
    handoffs. A prompt looser than either produces escalations the design never
    intended — "Is it RERA registered?" read as a legal matter, and a lead
    handed over instead of answered."""

    def test_a_factual_question_is_named_as_not_an_escalation(self):
        lowered = prompt.SYSTEM_PROMPT.lower()
        assert "factual question is never an escalation" in lowered
        assert "rera" in lowered

    def test_the_tool_that_answers_legal_questions_is_named(self):
        assert "get_property_details" in prompt.SYSTEM_PROMPT

    def test_the_evaluator_s_two_triggers_are_marked_as_automatic(self):
        # D16 computes these post-turn so escalation is guaranteed rather than
        # hoped for. The model doing it too costs a step and swallows the reply
        # the buyer actually needed — "your visit is confirmed".
        lowered = prompt.SYSTEM_PROMPT.lower()
        assert "handed over for you" in lowered
        assert "do not call it for those" in lowered

    @pytest.mark.parametrize("ask", ["negotiate", "loan", "legal dispute", "human"])
    def test_each_real_trigger_survives(self, ask):
        assert ask in prompt.SYSTEM_PROMPT.lower()
