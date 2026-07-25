#!/usr/bin/env python3
"""Prepare AIDER and CrisisMMD with deterministic stratified 80/10/10 splits."""

from __future__ import annotations

import argparse
import csv
import hashlib
import random
from collections import Counter, defaultdict
from pathlib import Path


SEED = 42
AIDER_LABELS = {
    "normal": "none",
    "traffic_accident": "mild",
    "traffic_accidents": "mild",
    "traffic_incident": "mild",
    "traffic": "mild",
    "fire": "moderate",
    "flood": "moderate",
    "flooded_areas": "moderate",
    "collapsed_building": "severe",
    "collapsed_buildings": "severe",
    "severe_flood": "severe",
}
CRISIS_LABELS = {
    "infrastructure_and_utility_damage": "shelter",
    "rescue_volunteering_or_donation_effort": "food|water",
    "affected_individuals": "medical",
    "injured_or_dead_people": "medical",
    "missing_or_found_people": "",
    "vehicle_damage": "",
    "other_relevant_information": "",
    "not_humanitarian": "",
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def normalize(value: str) -> str:
    return "_".join(value.strip().lower().replace("/", " ").replace("-", " ").split())


def apportioned_split(indices: list[int], rng: random.Random) -> dict[int, str]:
    shuffled = indices[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    return {
        idx: ("train" if pos < n_train else "val" if pos < n_train + n_val else "test")
        for pos, idx in enumerate(shuffled)
    }


def stratify(rows: list[dict[str, str]], column: str) -> None:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[row[column]].append(index)
    rng = random.Random(SEED)
    assignments: dict[int, str] = {}
    for label in sorted(groups):
        assignments.update(apportioned_split(groups[label], rng))
    for index, row in enumerate(rows):
        row["split"] = assignments[index]


def write_csvs(rows: list[dict[str, str]], output_dir: Path, stem: str) -> None:
    if not rows:
        raise RuntimeError(f"No rows found for {stem}")
    fields = list(rows[0])
    for name, selected in [(f"{stem}.csv", rows)] + [
        (f"{stem}_{split}.csv", [r for r in rows if r["split"] == split])
        for split in ("train", "val", "test")
    ]:
        with (output_dir / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(selected)


def prepare_aider(root: Path, output_dir: Path) -> list[dict[str, str]]:
    rows = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        source = normalize(path.parent.name)
        if source not in AIDER_LABELS:
            continue
        rows.append({
            "sample_id": hashlib.sha1(str(path.relative_to(root)).encode()).hexdigest()[:16],
            "image_path": str(path.resolve()),
            "source_label": source,
            "severity": AIDER_LABELS[source],
        })
    stratify(rows, "source_label")
    write_csvs(rows, output_dir, "aider")
    return rows


def find_column(fields: list[str], candidates: tuple[str, ...]) -> str | None:
    normalized = {normalize(field): field for field in fields}
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]
    return None


def prepare_crisismmd(
    root: Path,
    output_dir: Path,
    holdout_ids: set[str] | None = None,
) -> list[dict[str, str]]:
    holdout_ids = holdout_ids or set()
    by_tweet: dict[str, dict[str, str]] = {}
    for source_file in sorted(root.rglob("*.tsv")):
        if "task_humanitarian" not in source_file.name:
            continue
        with source_file.open(encoding="utf-8-sig", errors="replace", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            fields = reader.fieldnames or []
            label_col = find_column(fields, ("label_text", "label", "text_human"))
            tweet_col = find_column(fields, ("tweet_text", "tweet"))
            id_col = find_column(fields, ("tweet_id", "id"))
            image_col = find_column(fields, ("image_path", "image", "image_url"))
            if not label_col or not tweet_col or not id_col:
                continue
            for raw in reader:
                label = normalize(raw.get(label_col, ""))
                if label not in CRISIS_LABELS:
                    continue
                tweet_id = raw.get(id_col, "").strip()
                if not tweet_id or tweet_id in holdout_ids:
                    continue
                resources = CRISIS_LABELS[label]
                existing = by_tweet.get(tweet_id)
                if existing:
                    if existing["fine_category"] != label:
                        raise ValueError(
                            f"Inconsistent text labels for tweet {tweet_id}: "
                            f"{existing['fine_category']} vs {label}"
                        )
                    continue
                image_value = raw.get(image_col, "").strip() if image_col else ""
                by_tweet[tweet_id] = {
                    "tweet_id": tweet_id,
                    "tweet_text": raw.get(tweet_col, "").replace("\r", " ").replace("\n", " ").strip(),
                    "image_path": image_value,
                    "source_label": label,
                    "fine_category": label,
                    "resources": resources,
                }
    rows = list(by_tweet.values())
    stratify(rows, "fine_category")
    write_csvs(rows, output_dir, "crisismmd")
    return rows


def summarize(name: str, rows: list[dict[str, str]], label: str) -> None:
    splits = Counter(row["split"] for row in rows)
    labels = Counter(row[label] for row in rows)
    print(f"{name}: rows={len(rows)} splits={dict(sorted(splits.items()))}")
    print(f"{name}: labels={dict(sorted(labels.items()))}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aider-root", type=Path, required=True)
    parser.add_argument("--crisismmd-root", type=Path, required=True)
    parser.add_argument("--conflict-holdouts", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("pratyusha/data"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    aider = prepare_aider(args.aider_root, args.output_dir)
    holdout_ids: set[str] = set()
    if args.conflict_holdouts:
        with args.conflict_holdouts.open(encoding="utf-8", newline="") as handle:
            holdout_ids = {row["tweet_id"] for row in csv.DictReader(handle)}
    crisis = prepare_crisismmd(
        args.crisismmd_root,
        args.output_dir,
        holdout_ids=holdout_ids,
    )
    summarize("AIDER", aider, "severity")
    summarize("CrisisMMD", crisis, "fine_category")


if __name__ == "__main__":
    main()
