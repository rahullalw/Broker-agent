"""Phase 2 §7 — the OpenAI-format tool schemas.

The schema is the only description of these tools the model ever sees, so it
carries the closed vocabularies (D13). An enum the agent cannot violate removes
an entire class of failure — but only if the enum is actually in the schema and
actually matches the Python-side validation.
"""
import inspect
import json

import pytest

from app import tools
from app.enums import INTENT_TIER, LOCALITIES, URGENCY
from app.schemas import TOOLS

BY_NAME = {t["function"]["name"]: t["function"] for t in TOOLS}


def properties_of(name):
    return BY_NAME[name]["parameters"]["properties"]


class TestOpenAIFormat:
    def test_five_tools_no_more_no_less(self):
        assert len(TOOLS) == 5

    def test_every_entry_is_a_function_tool(self):
        for tool in TOOLS:
            assert tool["type"] == "function"
            assert set(tool) == {"type", "function"}

    def test_names_match_the_dispatcher_mapping_exactly(self):
        assert set(BY_NAME) == set(tools.TOOL_MAPPING)

    def test_every_tool_describes_itself(self):
        for name, function in BY_NAME.items():
            assert function["description"].strip(), name

    def test_parameters_are_json_schema_objects(self):
        for name, function in BY_NAME.items():
            params = function["parameters"]
            assert params["type"] == "object", name
            assert isinstance(params["properties"], dict), name
            assert isinstance(params["required"], list), name

    def test_the_payload_survives_json_serialisation(self):
        assert json.loads(json.dumps(TOOLS)) == TOOLS

    def test_no_unexpected_arguments_are_invited(self):
        for name, function in BY_NAME.items():
            assert function["parameters"]["additionalProperties"] is False, name


class TestSchemaMatchesTheFunction:
    @pytest.mark.parametrize("name", sorted(BY_NAME))
    def test_every_declared_property_is_a_real_parameter(self, name):
        signature = inspect.signature(tools.TOOL_MAPPING[name])
        assert set(properties_of(name)) <= set(signature.parameters), name

    @pytest.mark.parametrize("name", sorted(BY_NAME))
    def test_every_parameter_without_a_default_is_required(self, name):
        signature = inspect.signature(tools.TOOL_MAPPING[name])
        mandatory = {
            key for key, param in signature.parameters.items()
            if param.default is inspect.Parameter.empty
        }
        assert mandatory == set(BY_NAME[name]["parameters"]["required"]), name

    @pytest.mark.parametrize("name", sorted(BY_NAME))
    def test_every_parameter_is_declared(self, name):
        signature = inspect.signature(tools.TOOL_MAPPING[name])
        assert set(signature.parameters) == set(properties_of(name)), name


class TestClosedVocabularies:
    def test_search_declares_the_locality_enum(self):
        assert properties_of("search_properties")["localities"]["items"]["enum"] == list(
            LOCALITIES
        )

    def test_search_arguments_are_all_optional(self):
        # The agent must be able to search on partial qualification.
        assert BY_NAME["search_properties"]["parameters"]["required"] == []

    def test_details_declares_the_section_enum(self):
        enum = properties_of("get_property_details")["sections"]["items"]["enum"]
        assert enum == list(tools.SECTIONS)

    def test_profile_updates_is_one_typed_object(self):
        updates = properties_of("update_buyer_profile")["updates"]
        assert updates["type"] == "object"
        assert set(updates["properties"]) == {
            "budget_min", "budget_max", "preferred_localities", "bhk_need",
            "possession_need", "family_size", "intent_tier",
        }

    def test_every_profile_field_is_optional(self):
        updates = properties_of("update_buyer_profile")["updates"]
        assert updates.get("required", []) == []

    def test_profile_declares_the_locality_and_intent_enums(self):
        fields = properties_of("update_buyer_profile")["updates"]["properties"]
        assert fields["preferred_localities"]["items"]["enum"] == list(LOCALITIES)
        assert fields["intent_tier"]["enum"] == list(INTENT_TIER)

    def test_profile_fields_carry_their_real_types(self):
        fields = properties_of("update_buyer_profile")["updates"]["properties"]
        assert fields["budget_min"]["type"] == "integer"
        assert fields["budget_max"]["type"] == "integer"
        assert fields["bhk_need"]["type"] == "integer"
        assert fields["family_size"]["type"] == "integer"
        assert fields["possession_need"]["type"] == "string"

    def test_escalation_declares_the_urgency_enum(self):
        assert properties_of("escalate_to_broker")["urgency"]["enum"] == list(URGENCY)

    def test_month_shaped_fields_say_so_in_their_description(self):
        assert "YYYY-MM" in properties_of("search_properties")["possession_before"][
            "description"
        ]
        assert "YYYY-MM" in properties_of("update_buyer_profile")["updates"][
            "properties"
        ]["possession_need"]["description"]

    def test_the_slot_description_shows_the_exact_format(self):
        description = properties_of("book_site_visit")["slot"]["description"]
        assert "2026-08-09 17:00" in description


class TestToolsThatDeliberatelyDoNotExist:
    """Blocking by capability beats blocking by instruction (§1)."""

    @pytest.mark.parametrize("forbidden", [
        "negotiate", "discount", "price_negotiation",
        "loan", "emi", "interest",
        "legal_opinion", "legal_advice",
        "promise_possession", "commit_possession",
    ])
    def test_no_tool_offers_it(self, forbidden):
        assert not any(forbidden in name for name in BY_NAME)

    def test_the_schema_text_never_invites_negotiation(self):
        blob = json.dumps(TOOLS).lower()
        for word in ("negotiat", "discount", "emi ", "interest rate"):
            assert word not in blob


class TestStrictMode:
    def test_strict_is_declared_deliberately_on_every_tool(self):
        from app.schemas import STRICT
        for name, function in BY_NAME.items():
            assert function["strict"] is STRICT, name
