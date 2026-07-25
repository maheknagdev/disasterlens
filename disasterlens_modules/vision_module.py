"""DisasterLens image-severity inference using a fine-tuned CLIP head."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".cache" / "huggingface"))

import torch
from PIL import Image
from torch import nn
from transformers import CLIPModel, CLIPProcessor


LABELS = ["none", "mild", "moderate", "severe"]
MODEL_DIR = Path(__file__).resolve().parent / "models" / "clip_vision"
CHECKPOINT_PATH = MODEL_DIR / "best_head.pt"
_RUNTIME: dict[str, Any] | None = None


class SeverityHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 4),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features)


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
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Vision checkpoint not found: {CHECKPOINT_PATH}")
    device = _device()
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
    model_id = checkpoint.get("model_id", "openai/clip-vit-base-patch32")
    clip = CLIPModel.from_pretrained(model_id, local_files_only=True).to(device).eval()
    head = SeverityHead().to(device).eval()
    head.load_state_dict(checkpoint["state_dict"])
    _RUNTIME = {
        "device": device,
        "clip": clip,
        "head": head,
        "processor": CLIPProcessor.from_pretrained(model_id, local_files_only=True),
    }
    return _RUNTIME


def predict_severity(image_path: str | os.PathLike[str]) -> dict[str, object]:
    """Classify one image and return the exact DisasterLens vision contract."""
    runtime = _load_runtime()
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    inputs = runtime["processor"](images=image, return_tensors="pt")
    pixels = inputs["pixel_values"].to(runtime["device"])
    with torch.inference_mode():
        feature_output = runtime["clip"].get_image_features(pixel_values=pixels)
        features = getattr(feature_output, "pooler_output", feature_output)
        features = nn.functional.normalize(features, dim=-1)
        probabilities = torch.softmax(runtime["head"](features), dim=-1)[0]
    index = int(probabilities.argmax().item())
    return {
        "severity": LABELS[index],
        "confidence": float(probabilities[index].item()),
        "damage_types": [],
        "model_used": "CLIP-finetune",
    }
