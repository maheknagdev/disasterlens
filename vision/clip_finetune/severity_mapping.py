# AIDERv2 category -> baseline severity proxy (see AI_Prompts_Log / advisor feedback discussion).
# Vision signal alone is coarse; fusion layer refines this using text severity cues.

CLASS_NAMES = ["Earthquake", "Fire", "Flood", "Normal"]

CATEGORY_TO_SEVERITY = {
    "Normal": "none",
    "Fire": "moderate",
    "Flood": "moderate",
    "Earthquake": "severe",
}

SEVERITY_LEVELS = ["none", "mild", "moderate", "severe"]


def category_to_severity(category: str) -> str:
    return CATEGORY_TO_SEVERITY[category]
