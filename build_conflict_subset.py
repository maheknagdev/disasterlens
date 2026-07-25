#!/usr/bin/env python3
"""Build a leakage-free CrisisMMD conflict fixture for fusion evaluation."""

from __future__ import annotations

import argparse
import csv
import random
import re
from collections import Counter, defaultdict
from pathlib import Path


SEED = 42
SEVERITY_ORDER = {"none": 0, "mild": 1, "moderate": 2, "severe": 3}
IMAGE_SEVERITY = {
    "little_or_no_damage": "none",
    "mild_damage": "mild",
    "severe_damage": "severe",
}


def load_task(root: Path, task: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(root.glob(f"task_{task}_text_img_*.tsv")):
        with path.open(encoding="utf-8-sig", errors="replace", newline="") as handle:
            rows.extend(csv.DictReader(handle, delimiter="\t"))
    return rows


def text_severity(text: str) -> tuple[str, str]:
    """Apply a deterministic, reportable text-severity rubric."""
    normalized = " ".join(text.lower().split())
    explicit_none = (
        r"\bno (?:major )?damage\b",
        r"\bno (?:one|people|casualties) (?:was |were )?(?:hurt|injured|killed)\b",
        r"\blittle or no damage\b",
        r"\bminor damage\b",
    )
    severe_terms = (
        "dead", "death toll", "fatalities", "killed", "injured", "trapped",
        "missing", "collapsed", "collapse", "destroyed", "devastated",
        "catastrophic", "critical", "mass casualty",
    )
    moderate_terms = (
        "displaced", "evacuated", "evacuation", "damaged", "damage",
        "flooded", "flooding", "power outage", "without power", "affected",
        "rescue", "emergency shelter",
    )
    mild_terms = ("minor", "limited", "contained", "small fire", "road closed")

    if any(re.search(pattern, normalized) for pattern in explicit_none):
        return "none", "explicit_no_or_minor_damage"

    population_match = re.search(
        r"\b(\d[\d,]*)\s+(?:people|persons|residents|families)\b",
        normalized,
    )
    population = int(population_match.group(1).replace(",", "")) if population_match else 0
    severe_hits = [term for term in severe_terms if term in normalized]
    moderate_hits = [term for term in moderate_terms if term in normalized]
    mild_hits = [term for term in mild_terms if term in normalized]

    if severe_hits or population >= 1000:
        reason = f"severe_terms={','.join(severe_hits) or 'none'};population={population}"
        return "severe", reason
    if moderate_hits or population >= 100:
        reason = f"moderate_terms={','.join(moderate_hits) or 'none'};population={population}"
        return "moderate", reason
    if mild_hits or population > 0:
        reason = f"mild_terms={','.join(mild_hits) or 'none'};population={population}"
        return "mild", reason
    return "none", "no_damage_or_affected_population_signal"


def direction(image_signal: str, text_signal: str) -> str:
    if SEVERITY_ORDER[image_signal] > SEVERITY_ORDER[text_signal]:
        return "image>text"
    if SEVERITY_ORDER[text_signal] > SEVERITY_ORDER[image_signal]:
        return "text>image"
    return ""


def sample_rows(rows: list[dict[str, str]], count: int, rng: random.Random) -> list[dict[str, str]]:
    ordered = sorted(rows, key=lambda row: row["image_id"])
    rng.shuffle(ordered)
    return ordered[: min(count, len(ordered))]


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path("pratyusha/data/raw/crisismmd_annotations/crisismmd_datasplit_all"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("pratyusha/data"))
    parser.add_argument(
        "--media-root",
        type=Path,
        default=Path("pratyusha/data/raw/crisismmd/CrisisMMD_v2.0"),
    )
    parser.add_argument("--resource-count", type=int, default=150)
    parser.add_argument("--severity-count", type=int, default=150)
    parser.add_argument("--location-count", type=int, default=100)
    parser.add_argument("--control-count", type=int, default=100)
    args = parser.parse_args()

    rng = random.Random(SEED)
    humanitarian = {row["image_id"]: row for row in load_task(args.annotations, "humanitarian")}
    damage = {row["image_id"]: row for row in load_task(args.annotations, "damage")}
    joined: list[dict[str, str]] = []
    for image_id in sorted(humanitarian.keys() & damage.keys()):
        human = humanitarian[image_id]
        damage_row = damage[image_id]
        text_signal, rubric_reason = text_severity(human["tweet_text"])
        image_signal = IMAGE_SEVERITY[damage_row["label"]]
        joined.append(
            {
                **human,
                "image_path": str((args.media_root / human["image"]).resolve()),
                "image_severity_signal": image_signal,
                "text_severity_signal": text_signal,
                "rubric_reason": rubric_reason,
                "severity_direction": direction(image_signal, text_signal),
            }
        )

    resource_candidates = [
        row for row in joined
        if row["label_text"] != row["label_image"] and row["severity_direction"]
    ]
    severity_candidates = [
        row for row in joined
        if row["label_text"] == row["label_image"] and row["severity_direction"]
    ]
    control_candidates = [
        row for row in joined
        if row["label_text"] == row["label_image"] and not row["severity_direction"]
    ]

    selected: list[dict[str, str]] = []
    for conflict_type, candidates, count in (
        ("resource_mismatch", resource_candidates, args.resource_count),
        ("severity_mismatch", severity_candidates, args.severity_count),
        ("none", control_candidates, args.control_count),
    ):
        for row in sample_rows(candidates, count, rng):
            selected.append(
                {
                    **row,
                    "pair_id": f"real:{row['image_id']}",
                    "conflict_type": conflict_type,
                    "conflict_direction": row["severity_direction"],
                    "source": "real",
                    "image_event": row["event_name"],
                    "text_event": row["event_name"],
                    "image_tweet_id": row["tweet_id"],
                    "text_tweet_id": row["tweet_id"],
                }
            )

    location_pool = [
        row for row in joined
        if row["label_text"] == row["label_image"] and not row["severity_direction"]
    ]
    by_label: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in location_pool:
        by_label[row["label_text"]].append(row)
    synthetic: list[dict[str, str]] = []
    used_pairs: set[tuple[str, str]] = set()
    shuffled_pool = sorted(location_pool, key=lambda row: row["image_id"])
    rng.shuffle(shuffled_pool)
    for image_row in shuffled_pool:
        candidates = [
            text_row for text_row in by_label[image_row["label_image"]]
            if text_row["event_name"] != image_row["event_name"]
            and text_row["tweet_id"] != image_row["tweet_id"]
        ]
        rng.shuffle(candidates)
        for text_row in candidates:
            key = (image_row["image_id"], text_row["tweet_id"])
            if key in used_pairs:
                continue
            used_pairs.add(key)
            text_signal, reason = text_severity(text_row["tweet_text"])
            conflict_direction = direction(image_row["image_severity_signal"], text_signal)
            if not conflict_direction:
                # The required schema defines direction for every conflict row.
                continue
            synthetic.append(
                {
                    **image_row,
                    "tweet_text": text_row["tweet_text"],
                    "pair_id": f"synthetic:{image_row['image_id']}:{text_row['tweet_id']}",
                    "conflict_type": "location_mismatch",
                    "text_severity_signal": text_signal,
                    "rubric_reason": reason,
                    "conflict_direction": conflict_direction,
                    "source": "synthetic",
                    "image_event": image_row["event_name"],
                    "text_event": text_row["event_name"],
                    "image_tweet_id": image_row["tweet_id"],
                    "text_tweet_id": text_row["tweet_id"],
                }
            )
            break
        if len(synthetic) >= args.location_count:
            break
    selected.extend(synthetic)

    selected.sort(key=lambda row: (row["conflict_type"], row["pair_id"]))
    schema_fields = [
        "pair_id",
        "conflict_type",
        "image_severity_signal",
        "text_severity_signal",
        "conflict_direction",
        "source",
    ]
    write_csv(args.output_dir / "conflict_subset.csv", selected, schema_fields)

    input_fields = [
        "pair_id", "image_tweet_id", "text_tweet_id", "image_id", "image", "image_path",
        "tweet_text", "image_event", "text_event", "label_text", "label_image",
        "rubric_reason",
    ]
    write_csv(args.output_dir / "conflict_subset_inputs.csv", selected, input_fields)

    holdout_rows = []
    for row in selected:
        holdout_rows.append({"pair_id": row["pair_id"], "tweet_id": row["image_tweet_id"], "role": "image"})
        holdout_rows.append({"pair_id": row["pair_id"], "tweet_id": row["text_tweet_id"], "role": "text"})
    unique_holdouts = {
        (row["pair_id"], row["tweet_id"], row["role"]): row for row in holdout_rows
    }
    write_csv(
        args.output_dir / "conflict_holdout_ids.csv",
        list(unique_holdouts.values()),
        ["pair_id", "tweet_id", "role"],
    )

    review_candidates = [row for row in selected if row["source"] == "real"]
    review_rows: list[dict[str, str]] = []
    for conflict_type in ("severity_mismatch", "resource_mismatch", "none"):
        group = [row for row in review_candidates if row["conflict_type"] == conflict_type]
        review_rows.extend(sample_rows(group, 20, rng))
    write_csv(
        args.output_dir / "conflict_text_severity_review.csv",
        review_rows,
        [
            "pair_id", "tweet_text", "image_severity_signal", "text_severity_signal",
            "rubric_reason", "conflict_type",
        ],
    )

    counts = Counter((row["source"], row["conflict_type"]) for row in selected)
    print(f"conflict_subset_rows={len(selected)}")
    print(f"holdout_tweet_ids={len({row['tweet_id'] for row in unique_holdouts.values()})}")
    print(f"review_rows={len(review_rows)}")
    for key, value in sorted(counts.items()):
        print(f"source={key[0]} conflict_type={key[1]} rows={value}")


if __name__ == "__main__":
    main()
