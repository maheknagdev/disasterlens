# Deterministic proxy mapping: CrisisMMD fine category -> urgency (1-5, 5 = most urgent).
# Rationale: categories implying direct threat to life/safety score highest; categories
# with no specific relief-relevant signal score lowest. Mirrors the same proxy-labeling
# approach used for AIDER severity (vision/clip_finetune/severity_mapping.py).

URGENCY_RUBRIC = {
    "not_humanitarian": 1,
    "other_relevant_information": 2,
    "vehicle_damage": 2,
    "rescue_volunteering_or_donation_effort": 3,
    "infrastructure_and_utility_damage": 3,
    "missing_or_found_people": 4,
    "affected_individuals": 4,
    "injured_or_dead_people": 5,
}


def category_to_urgency(fine_category: str) -> int:
    return URGENCY_RUBRIC[fine_category]
