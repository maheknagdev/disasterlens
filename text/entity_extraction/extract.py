import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

from schema import EntityExtraction

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

SYSTEM_PROMPT = (
    "You extract structured disaster-relief information from a short text snippet "
    "(e.g. a news report or social media post). Only report what is stated or clearly "
    "implied in the text — do not guess or invent numbers or locations."
)


def extract_entities(text: str) -> EntityExtraction:
    response = client.models.generate_content(
        model="gemini-flash-latest",
        contents=text,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=EntityExtraction,
        ),
    )
    return response.parsed


if __name__ == "__main__":
    sample_text = (
        "Flooding has displaced an estimated 3,000 residents in the riverside "
        "district of Millbrook. Local shelters are overwhelmed and volunteers "
        "are calling for clean water and medical supplies."
    )
    result = extract_entities(sample_text)
    print(result.model_dump_json(indent=2))
