#!/usr/bin/env python3
"""Train the DisasterLens frozen-CLIP severity classification head."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".cache" / "huggingface"))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".cache"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from transformers import CLIPModel, CLIPProcessor
from transformers.utils import logging as transformers_logging

transformers_logging.disable_progress_bar()


MODEL_ID = "openai/clip-vit-base-patch32"
LABELS = ["none", "mild", "moderate", "severe"]
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}


class ImageDataset(Dataset):
    def __init__(self, csv_path: Path) -> None:
        with csv_path.open(encoding="utf-8", newline="") as handle:
            self.rows = list(csv.DictReader(handle))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[Image.Image, int]:
        row = self.rows[index]
        with Image.open(row["image_path"]) as source:
            image = source.convert("RGB")
        return image, LABEL_TO_ID[row["severity"]]


class SeverityHead(nn.Module):
    def __init__(self, input_dim: int = 512) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, len(LABELS)),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features)


def select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def embed_split(
    csv_path: Path,
    cache_path: Path,
    model: CLIPModel,
    processor: CLIPProcessor,
    device: torch.device,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if cache_path.exists():
        cached = torch.load(cache_path, map_location="cpu", weights_only=True)
        return cached["features"], cached["labels"]

    dataset = ImageDataset(csv_path)

    def collate(batch: list[tuple[Image.Image, int]]) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        images, labels = zip(*batch)
        inputs = processor(images=list(images), return_tensors="pt")
        return inputs, torch.tensor(labels, dtype=torch.long)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate)
    all_features: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    with torch.inference_mode():
        for inputs, labels in loader:
            pixels = inputs["pixel_values"].to(device)
            feature_output = model.get_image_features(pixel_values=pixels)
            features = getattr(feature_output, "pooler_output", feature_output)
            features = nn.functional.normalize(features, dim=-1)
            all_features.append(features.cpu())
            all_labels.append(labels)
    result = (torch.cat(all_features), torch.cat(all_labels))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"features": result[0], "labels": result[1]}, cache_path)
    return result


def evaluate(head: SeverityHead, loader: DataLoader, device: torch.device) -> tuple[float, list[int], list[int]]:
    head.eval()
    predictions: list[int] = []
    targets: list[int] = []
    with torch.inference_mode():
        for features, labels in loader:
            logits = head(features.to(device))
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
            targets.extend(labels.tolist())
    return accuracy_score(targets, predictions), targets, predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = select_device()
    model_dir = PROJECT_ROOT / "pratyusha" / "models" / "clip_vision"
    model_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = model_dir / "embeddings"

    processor = CLIPProcessor.from_pretrained(MODEL_ID, local_files_only=True)
    clip = CLIPModel.from_pretrained(MODEL_ID, local_files_only=True).to(device)
    clip.eval()
    for parameter in clip.parameters():
        parameter.requires_grad = False

    datasets: dict[str, TensorDataset] = {}
    for split in ("train", "val", "test"):
        features, labels = embed_split(
            PROJECT_ROOT / "pratyusha" / "data" / f"aider_{split}.csv",
            cache_dir / f"{split}.pt",
            clip,
            processor,
            device,
            args.batch_size,
        )
        datasets[split] = TensorDataset(features, labels)

    loaders = {
        split: DataLoader(dataset, batch_size=256, shuffle=(split == "train"))
        for split, dataset in datasets.items()
    }
    head = SeverityHead().to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=args.learning_rate)
    train_labels = datasets["train"].tensors[1]
    counts = torch.bincount(train_labels, minlength=len(LABELS)).float()
    class_weights = (counts.sum() / (len(LABELS) * counts)).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    best_val = -1.0
    checkpoint_path = model_dir / "best_head.pt"

    for epoch in range(1, args.epochs + 1):
        head.train()
        loss_sum = 0.0
        examples = 0
        for features, labels in loaders["train"]:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(head(features), labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * labels.size(0)
            examples += labels.size(0)
        val_accuracy, _, _ = evaluate(head, loaders["val"], device)
        if val_accuracy > best_val:
            best_val = val_accuracy
            torch.save(
                {
                    "state_dict": head.state_dict(),
                    "labels": LABELS,
                    "model_id": MODEL_ID,
                    "input_dim": 512,
                    "validation_accuracy": best_val,
                },
                checkpoint_path,
            )
        print(f"epoch={epoch:02d} train_loss={loss_sum / examples:.4f} val_accuracy={val_accuracy:.4f}", flush=True)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    head.load_state_dict(checkpoint["state_dict"])
    test_accuracy, targets, predictions = evaluate(head, loaders["test"], device)
    matrix = confusion_matrix(targets, predictions, labels=list(range(len(LABELS))))
    confusion_path = model_dir / "confusion_matrix.png"
    figure, axis = plt.subplots(figsize=(6.5, 5.5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=LABELS, yticklabels=LABELS, ax=axis)
    axis.set(title="CLIP severity test confusion matrix", xlabel="Predicted", ylabel="Actual")
    figure.tight_layout()
    figure.savefig(confusion_path, dpi=160)
    plt.close(figure)

    report = classification_report(targets, predictions, target_names=LABELS, output_dict=True, zero_division=0)
    metrics = {
        "test_accuracy": test_accuracy,
        "best_validation_accuracy": best_val,
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
        "labels": LABELS,
        "model_id": MODEL_ID,
        "epochs": args.epochs,
        "seed": args.seed,
    }
    with (model_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
        handle.write("\n")
    print(f"FINAL_TEST_ACCURACY={test_accuracy:.6f}", flush=True)
    print(f"CONFUSION_MATRIX={confusion_path}", flush=True)


if __name__ == "__main__":
    main()
