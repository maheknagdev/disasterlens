#!/usr/bin/env python3
"""Fine-tune RoBERTa for DisasterLens multi-label resource classification."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".cache" / "huggingface"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import numpy as np
import torch
from sklearn.metrics import classification_report, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import RobertaForSequenceClassification, RobertaTokenizerFast
from transformers.utils import logging as transformers_logging

transformers_logging.disable_progress_bar()

MODEL_ID = "FacebookAI/roberta-base"
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


class CrisisDataset(Dataset):
    def __init__(self, csv_path: Path, tokenizer: RobertaTokenizerFast, max_length: int = 128) -> None:
        with csv_path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        texts = [row["tweet_text"] for row in rows]
        self.encodings = tokenizer(texts, truncation=True, padding="max_length", max_length=max_length)
        category_to_id = {category: index for index, category in enumerate(FINE_CATEGORIES)}
        self.labels = torch.tensor(
            [category_to_id[row["fine_category"]] for row in rows],
            dtype=torch.long,
        )

    def __len__(self) -> int:
        return self.labels.shape[0]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = {key: torch.tensor(values[index], dtype=torch.long) for key, values in self.encodings.items()}
        item["labels"] = self.labels[index]
        return item


def choose_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def predict(
    model: RobertaForSequenceClassification,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probabilities: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    with torch.inference_mode():
        for batch in loader:
            labels = batch.pop("labels")
            inputs = {key: value.to(device) for key, value in batch.items()}
            logits = model(**inputs).logits
            probabilities.append(torch.softmax(logits, dim=-1).cpu().numpy())
            targets.append(labels.numpy())
    return np.concatenate(targets), np.concatenate(probabilities)


def to_contract_vectors(fine_vectors: np.ndarray) -> np.ndarray:
    resource_index = {resource: index for index, resource in enumerate(RESOURCES)}
    mapping = np.zeros((len(FINE_CATEGORIES), len(RESOURCES)), dtype=int)
    for fine_index, category in enumerate(FINE_CATEGORIES):
        for resource in FINE_TO_RESOURCES[category]:
            mapping[fine_index, resource_index[resource]] = 1
    return ((fine_vectors @ mapping) > 0).astype(int)


def indices_to_one_hot(indices: np.ndarray) -> np.ndarray:
    result = np.zeros((len(indices), len(FINE_CATEGORIES)), dtype=int)
    result[np.arange(len(indices)), indices.astype(int)] = 1
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "pratyusha" / "data",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "pratyusha" / "models" / "roberta_resources",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = RobertaTokenizerFast.from_pretrained(MODEL_ID, local_files_only=True)
    model = RobertaForSequenceClassification.from_pretrained(
        MODEL_ID,
        num_labels=len(FINE_CATEGORIES),
        problem_type="single_label_classification",
        local_files_only=True,
    ).to(device)
    datasets = {
        split: CrisisDataset(args.data_dir / f"crisismmd_{split}.csv", tokenizer)
        for split in ("train", "val", "test")
    }
    loaders = {
        split: DataLoader(dataset, batch_size=args.batch_size * (2 if split != "train" else 1), shuffle=(split == "train"))
        for split, dataset in datasets.items()
    }
    counts = torch.bincount(datasets["train"].labels, minlength=len(FINE_CATEGORIES)).float()
    class_weights = torch.sqrt(counts.sum() / (len(FINE_CATEGORIES) * counts.clamp_min(1)))
    criterion = nn.CrossEntropyLoss(weight=class_weights.clamp(max=5.0).to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    best_f1 = -1.0
    checkpoint_path = output_dir / "best_state_fine.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        examples = 0
        for batch in loaders["train"]:
            labels = batch.pop("labels").to(device)
            inputs = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            logits = model(**inputs).logits
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * labels.shape[0]
            examples += labels.shape[0]
        val_targets, val_probabilities = predict(model, loaders["val"], device)
        val_predictions = val_probabilities.argmax(axis=1)
        val_f1 = f1_score(val_targets, val_predictions, average="macro", zero_division=0)
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), checkpoint_path)
        print(f"epoch={epoch:02d} train_loss={total_loss / examples:.4f} val_macro_f1={val_f1:.4f}", flush=True)

    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    test_targets, test_probabilities = predict(model, loaders["test"], device)
    test_predictions = test_probabilities.argmax(axis=1)
    test_fine_macro_f1 = f1_score(test_targets, test_predictions, average="macro", zero_division=0)
    fine_report = classification_report(
        test_targets,
        test_predictions,
        target_names=FINE_CATEGORIES,
        output_dict=True,
        zero_division=0,
    )
    contract_targets = to_contract_vectors(indices_to_one_hot(test_targets))
    contract_predictions = to_contract_vectors(indices_to_one_hot(test_predictions))
    test_contract_macro_f1 = f1_score(
        contract_targets,
        contract_predictions,
        average="macro",
        zero_division=0,
    )
    contract_report = classification_report(
        contract_targets,
        contract_predictions,
        target_names=RESOURCES,
        output_dict=True,
        zero_division=0,
    )
    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    metadata = {
        "base_model": MODEL_ID,
        "fine_categories": FINE_CATEGORIES,
        "contract_resources": RESOURCES,
        "fine_to_resources": FINE_TO_RESOURCES,
        "threshold": 0.5,
        "max_length": 128,
        "epochs": args.epochs,
        "seed": args.seed,
        "best_validation_macro_f1": best_f1,
        "test_fine_macro_f1": test_fine_macro_f1,
        "test_contract_macro_f1": test_contract_macro_f1,
        "fine_classification_report": fine_report,
        "contract_classification_report": contract_report,
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")
    print(f"FINAL_TEST_FINE_MACRO_F1={test_fine_macro_f1:.6f}", flush=True)
    print(f"FINAL_TEST_CONTRACT_MACRO_F1={test_contract_macro_f1:.6f}", flush=True)


if __name__ == "__main__":
    main()
