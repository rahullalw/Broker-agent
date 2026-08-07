"""§5 — the agent never accepts free text. Every enum here is also declared in a
tool JSON schema (D13), so drift between the two is a real failure mode."""
import pytest

from app import enums


class TestVocabularies:
    def test_localities_are_the_four_seeded_areas(self):
        assert enums.LOCALITIES == ("Bopal", "South Bopal", "Shela", "Satellite")

    def test_status_values(self):
        assert enums.STATUS == ("ready_to_move", "under_construction")

    def test_intent_tier_is_ordered_cold_to_hot(self):
        assert enums.INTENT_TIER == ("cold", "warm", "hot")

    def test_conversation_status_values(self):
        assert enums.CONVO_STATUS == ("active", "escalated", "closed")

    def test_urgency_values(self):
        assert enums.URGENCY == ("low", "medium", "high")

    def test_twelve_amenities_all_snake_case(self):
        assert len(enums.AMENITIES) == 12
        for amenity in enums.AMENITIES:
            assert amenity == amenity.lower()
            assert " " not in amenity

    @pytest.mark.parametrize("name", [
        "LOCALITIES", "STATUS", "INTENT_TIER", "CONVO_STATUS", "URGENCY", "AMENITIES",
    ])
    def test_vocabularies_are_immutable_tuples(self, name):
        # A list here would let a tool handler mutate the schema's own enum.
        assert isinstance(getattr(enums, name), tuple), name

    @pytest.mark.parametrize("name", [
        "LOCALITIES", "STATUS", "INTENT_TIER", "CONVO_STATUS", "URGENCY", "AMENITIES",
    ])
    def test_no_duplicate_members(self, name):
        values = getattr(enums, name)
        assert len(set(values)) == len(values), name


class TestAdjacency:
    """Feeds the D14 near-match relaxation in Phase 2."""

    def test_every_locality_has_neighbours(self):
        assert set(enums.ADJACENT) == set(enums.LOCALITIES)

    def test_neighbours_are_known_localities(self):
        for locality, neighbours in enums.ADJACENT.items():
            for neighbour in neighbours:
                assert neighbour in enums.LOCALITIES, f"{locality} -> {neighbour}"

    def test_no_locality_is_its_own_neighbour(self):
        for locality, neighbours in enums.ADJACENT.items():
            assert locality not in neighbours

    def test_bopal_cluster_is_mutually_adjacent(self):
        # Relaxing Bopal must be able to reach South Bopal and Shela, and back.
        assert set(enums.ADJACENT["Bopal"]) == {"South Bopal", "Shela"}
        assert "Bopal" in enums.ADJACENT["South Bopal"]
        assert "Bopal" in enums.ADJACENT["Shela"]
