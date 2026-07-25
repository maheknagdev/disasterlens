#!/usr/bin/env python3
"""Generate DisasterLens EDA plots without emitting raw records."""

from __future__ import annotations

import csv
import json
import os
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, UnidentifiedImageError


DATA_DIR = PROJECT_ROOT / "pratyusha" / "data"
OUTPUT_DIR = PROJECT_ROOT / "pratyusha" / "notebooks" / "plots"
COLORS = ["#355070", "#6d597a", "#b56576", "#e56b6f", "#eaac8b"]


def read_rows(name: str) -> list[dict[str, str]]:
    with (DATA_DIR / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def save_figure(figure: plt.Figure, name: str) -> None:
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / name, dpi=160, bbox_inches="tight")
    plt.close(figure)


def plot_class_distribution(
    aider: list[dict[str, str]], crisis: list[dict[str, str]]
) -> dict[str, dict[str, int]]:
    severity = Counter(row["severity"] for row in aider)
    fine_categories = Counter(row["fine_category"] for row in crisis)
    resources: Counter[str] = Counter()
    for row in crisis:
        labels = row["resources"].split("|") if row["resources"] else ["no match"]
        resources.update(labels)

    figure, axes = plt.subplots(1, 3, figsize=(17, 5))
    severity_order = ["none", "mild", "moderate", "severe"]
    axes[0].bar(severity_order, [severity[x] for x in severity_order], color=COLORS[:4])
    axes[0].set(title="AIDER severity classes", ylabel="Samples", xlabel="Severity")
    resource_order = ["food", "water", "shelter", "medical", "no match"]
    fine_order = sorted(fine_categories)
    axes[1].bar(fine_order, [fine_categories[x] for x in fine_order], color=COLORS[1])
    axes[1].set(title="CrisisMMD fine categories", ylabel="Tweets", xlabel="Original category")
    axes[1].tick_params(axis="x", rotation=70, labelsize=8)
    axes[2].bar(resource_order, [resources[x] for x in resource_order], color=COLORS)
    axes[2].set(title="Contract-facing resource labels", ylabel="Samples", xlabel="Resource")
    axes[2].tick_params(axis="x", rotation=25)
    save_figure(figure, "class_distribution.png")
    return {
        "aider_severity": dict(severity),
        "crisismmd_fine_categories": dict(fine_categories),
        "crisismmd_resources": dict(resources),
    }


def plot_sample_counts(
    aider: list[dict[str, str]], crisis: list[dict[str, str]]
) -> dict[str, dict[str, int]]:
    splits = ["train", "val", "test"]
    aider_counts = Counter(row["split"] for row in aider)
    crisis_counts = Counter(row["split"] for row in crisis)
    positions = range(len(splits))
    width = 0.36
    figure, axis = plt.subplots(figsize=(7.5, 4.5))
    axis.bar([x - width / 2 for x in positions], [aider_counts[x] for x in splits], width, label="AIDER", color=COLORS[0])
    axis.bar([x + width / 2 for x in positions], [crisis_counts[x] for x in splits], width, label="CrisisMMD", color=COLORS[3])
    axis.set(title="Dataset sample counts by split", ylabel="Samples", xlabel="Split", xticks=list(positions), xticklabels=splits)
    axis.legend()
    save_figure(figure, "sample_counts.png")
    return {"aider": dict(aider_counts), "crisismmd": dict(crisis_counts)}


def plot_image_sizes(aider: list[dict[str, str]]) -> dict[str, object]:
    widths: list[int] = []
    heights: list[int] = []
    failures = 0
    for row in aider:
        try:
            with Image.open(row["image_path"]) as image:
                width, height = image.size
            widths.append(width)
            heights.append(height)
        except (OSError, UnidentifiedImageError):
            failures += 1

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].hist(widths, bins=40, color=COLORS[0], alpha=0.85)
    axes[0].set(title="AIDER image widths", xlabel="Pixels", ylabel="Images")
    axes[1].hist(heights, bins=40, color=COLORS[3], alpha=0.85)
    axes[1].set(title="AIDER image heights", xlabel="Pixels", ylabel="Images")
    save_figure(figure, "image_size_distribution.png")
    return {
        "readable_images": len(widths),
        "unreadable_images": failures,
        "width_range": [min(widths), max(widths)],
        "height_range": [min(heights), max(heights)],
    }


def plot_tweet_lengths(crisis: list[dict[str, str]]) -> dict[str, float | int]:
    lengths = [len(row["tweet_text"].split()) for row in crisis]
    figure, axis = plt.subplots(figsize=(7.5, 4.5))
    axis.hist(lengths, bins=35, color=COLORS[1], alpha=0.9)
    axis.set(title="CrisisMMD tweet length distribution", xlabel="Whitespace-delimited words", ylabel="Tweets")
    save_figure(figure, "tweet_length_histogram.png")
    ordered = sorted(lengths)
    return {
        "count": len(lengths),
        "minimum_words": min(lengths),
        "median_words": ordered[len(ordered) // 2],
        "mean_words": round(sum(lengths) / len(lengths), 2),
        "maximum_words": max(lengths),
    }


def plot_conflict_distribution() -> dict[str, dict[str, int] | int]:
    conflicts = read_rows("conflict_subset.csv")
    by_type = Counter(row["conflict_type"] for row in conflicts)
    by_source = Counter(row["source"] for row in conflicts)
    types = ["severity_mismatch", "resource_mismatch", "location_mismatch", "none"]
    real = [
        sum(1 for row in conflicts if row["conflict_type"] == conflict_type and row["source"] == "real")
        for conflict_type in types
    ]
    synthetic = [
        sum(1 for row in conflicts if row["conflict_type"] == conflict_type and row["source"] == "synthetic")
        for conflict_type in types
    ]
    positions = range(len(types))
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.bar(positions, real, label="Real", color=COLORS[0])
    axis.bar(positions, synthetic, bottom=real, label="Synthetic", color=COLORS[4])
    axis.set(
        title="Fusion conflict evaluation fixture",
        ylabel="Pairs",
        xlabel="Conflict type",
        xticks=list(positions),
        xticklabels=types,
    )
    axis.tick_params(axis="x", rotation=20)
    axis.legend()
    save_figure(figure, "conflict_distribution.png")
    return {"rows": len(conflicts), "by_type": dict(by_type), "by_source": dict(by_source)}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    aider = read_rows("aider.csv")
    crisis = read_rows("crisismmd.csv")
    previous_summary: dict[str, object] = {}
    summary_path = OUTPUT_DIR / "eda_summary.json"
    if summary_path.exists():
        with summary_path.open(encoding="utf-8") as handle:
            previous_summary = json.load(handle)
    image_sizes = previous_summary.get("image_sizes")
    if not image_sizes or not (OUTPUT_DIR / "image_size_distribution.png").exists():
        image_sizes = plot_image_sizes(aider)
    summary = {
        "class_distribution": plot_class_distribution(aider, crisis),
        "sample_counts": plot_sample_counts(aider, crisis),
        "image_sizes": image_sizes,
        "tweet_lengths": plot_tweet_lengths(crisis),
        "conflict_distribution": plot_conflict_distribution(),
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print("EDA complete: 5 plots generated")


if __name__ == "__main__":
    main()
