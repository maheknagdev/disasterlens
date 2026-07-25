import os
import sys
from dotenv import load_dotenv
from google import genai
from google.genai import types

from fusion_schema import FusionOutput

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "text", "entity_extraction"))
from schema import EntityExtraction  # noqa: E402

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

SYSTEM_PROMPT = (
    "You are the fusion layer of a disaster relief triage system. You receive a vision-based "
    "severity signal (from a fine-tuned image classifier), and two independent text-analysis "
    "signals for the same report — one from a general-purpose LLM extractor, one from a "
    "fine-tuned RoBERTa classifier — which may disagree with each other or with the vision "
    "signal. Combine all signals into a final severity level and a resource-need vector "
    "(food, water, shelter, medical; each 1-5, 5 = most urgent). "
    "Conflict policy: whenever any signals disagree, ALWAYS resolve to the most severe/urgent "
    "reading among them — do not average or downplay any signal."
)


def fuse(
    vision_severity: str,
    vision_confidence: float,
    gemini_entities: EntityExtraction,
    roberta_output: dict,
) -> FusionOutput:
    prompt = (
        f"Vision signal — severity: {vision_severity} (confidence: {vision_confidence:.2f})\n"
        f"Text signal (LLM extractor) — affected population: {gemini_entities.population_estimate}, "
        f"resources mentioned: {gemini_entities.resource_types_mentioned}, "
        f"locations: {gemini_entities.locations}\n"
        f"Text signal (RoBERTa classifier) — resources mentioned: {roberta_output['resources_mentioned']}, "
        f"location: {roberta_output['location']}, "
        f"population estimate: {roberta_output['population_estimate']}, "
        f"event type: {roberta_output['event_type']}, "
        f"urgency keywords: {roberta_output['urgency_keywords']}"
    )
    response = client.models.generate_content(
        model="gemini-flash-latest",
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=FusionOutput,
        ),
    )
    return response.parsed


if __name__ == "__main__":
    sample_entities = EntityExtraction(
        population_estimate=3000,
        resource_types_mentioned=["shelter", "water", "medical"],
        locations=["Millbrook"],
    )
    sample_roberta = {
        "resources_mentioned": ["food", "water"],
        "location": "Millbrook",
        "population_estimate": 3000,
        "event_type": "flood",
        "urgency_keywords": ["displaced"],
    }
    result = fuse(
        vision_severity="moderate",
        vision_confidence=0.82,
        gemini_entities=sample_entities,
        roberta_output=sample_roberta,
    )
    print(result.model_dump_json(indent=2))
