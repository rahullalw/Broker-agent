"""Closed vocabularies (§5 — never accept free text).

Each of these is mirrored in a tool's JSON schema (D13). They are tuples rather
than lists so a tool handler cannot mutate the vocabulary the schema advertises.
"""

LOCALITIES = ("Bopal", "South Bopal", "Shela", "Satellite")
STATUS = ("ready_to_move", "under_construction")
INTENT_TIER = ("cold", "warm", "hot")
CONVO_STATUS = ("active", "escalated", "closed")
URGENCY = ("low", "medium", "high")

AMENITIES = (
    "clubhouse", "gym", "swimming_pool", "covered_parking", "kids_play_area",
    "landscaped_garden", "24x7_security", "power_backup", "lift",
    "indoor_games", "jogging_track", "community_hall",
)

# Used by search_properties near-match relaxation (D14, Phase 2). Satellite is
# across town from the Bopal cluster, so it relaxes outward but nothing relaxes
# into it — that asymmetry is deliberate.
ADJACENT = {
    "Bopal":       ("South Bopal", "Shela"),
    "South Bopal": ("Bopal", "Shela"),
    "Shela":       ("Bopal", "South Bopal"),
    "Satellite":   ("South Bopal", "Bopal"),
}
