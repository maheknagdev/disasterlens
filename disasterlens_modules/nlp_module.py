"""DisasterLens resource classification and disaster-entity extraction."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".cache" / "huggingface"))

import torch
from transformers import RobertaForSequenceClassification, RobertaTokenizerFast


RESOURCES = ["food", "water", "shelter", "medical"]
FINE_CATEGORIES = [
    "affected_individuals",
    "infrastructure_and_utility_damage",
    "injured_or_dead_people",
    "missing_or_found_people",
    "not_humanitarian",
    "other_relevant_information",
    "rescue_volunteering_or_donation_effort",
    "vehicle_damage",
]
FINE_TO_RESOURCES = {
    "affected_individuals": ["medical"],
    "infrastructure_and_utility_damage": ["shelter"],
    "injured_or_dead_people": ["medical"],
    "missing_or_found_people": [],
    "not_humanitarian": [],
    "other_relevant_information": [],
    "rescue_volunteering_or_donation_effort": ["food", "water"],
    "vehicle_damage": [],
}
MODEL_DIR = Path(__file__).resolve().parent / "models" / "roberta_resources"
_RUNTIME: dict[str, Any] | None = None


def _device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _load_runtime() -> dict[str, Any]:
    global _RUNTIME
    if _RUNTIME is not None:
        return _RUNTIME
    if not (MODEL_DIR / "config.json").exists():
        raise FileNotFoundError(f"NLP checkpoint not found: {MODEL_DIR}")
    device = _device()
    _RUNTIME = {
        "device": device,
        "tokenizer": RobertaTokenizerFast.from_pretrained(MODEL_DIR, local_files_only=True),
        "model": RobertaForSequenceClassification.from_pretrained(MODEL_DIR, local_files_only=True).to(device).eval(),
    }
    return _RUNTIME


def classify_resources_fine_scores(text: str) -> dict[str, float]:
    """Return all eight fine-category softmax scores for evaluation/debugging."""
    runtime = _load_runtime()
    inputs = runtime["tokenizer"](
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
    )
    inputs = {key: value.to(runtime["device"]) for key, value in inputs.items()}
    with torch.inference_mode():
        probabilities = torch.softmax(runtime["model"](**inputs).logits, dim=-1)[0]
    return {
        category: float(probability.item())
        for category, probability in zip(FINE_CATEGORIES, probabilities)
    }


def classify_resources_fine(text: str) -> list[str]:
    """Return the single original CrisisMMD category with highest probability."""
    scores = classify_resources_fine_scores(text)
    return [max(FINE_CATEGORIES, key=scores.__getitem__)]


def classify_resources(text: str) -> list[str]:
    """Map fine CrisisMMD predictions into the stable four-resource contract."""
    fine_predictions = classify_resources_fine(text)
    mapped = {
        resource
        for category in fine_predictions
        for resource in FINE_TO_RESOURCES[category]
    }
    return [resource for resource in RESOURCES if resource in mapped]


def _local_entities(text: str) -> dict[str, object]:
    lower = text.lower()
    event_type = next((event for event in ("earthquake", "flood", "fire", "hurricane") if event in lower), None)
    population_match = re.search(r"\b(\d[\d,]*)\s+(?:people|persons|residents|families)\b", text, re.IGNORECASE)
    population = int(population_match.group(1).replace(",", "")) if population_match else None
    urgent_vocabulary = ("urgent", "urgently", "trapped", "critical", "missing", "emergency", "immediate", "displaced")
    urgency = [word for word in urgent_vocabulary if re.search(rf"\b{re.escape(word)}\b", lower)]
    location_match = re.search(
        r"\b(?:in|near|at)\s+([A-Z][A-Za-z.-]*(?:[ ,]+[A-Z][A-Za-z.-]*){0,2})",
        text,
    )
    location = location_match.group(1).strip(" ,.") if location_match else None
    return {
        "location": location,
        "population_estimate": population,
        "event_type": event_type,
        "urgency_keywords": urgency,
    }


def extract_entities(text: str) -> dict[str, object]:
    """Extract disaster entities with deterministic local rules."""
    return _local_entities(text)


def analyze_text(snippet: str) -> dict[str, object]:
    """Return the exact DisasterLens NLP JSON contract for one text snippet."""
    entities = extract_entities(snippet)
    return {
        "resources_mentioned": classify_resources(snippet),
        "location": entities.get("location"),
        "population_estimate": entities.get("population_estimate"),
        "event_type": entities.get("event_type"),
        "urgency_keywords": entities.get("urgency_keywords", []),
    }
