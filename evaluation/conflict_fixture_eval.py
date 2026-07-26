import csv
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_CSV = BASE_DIR / "disasterlens_modules" / "data" / "conflict_text_severity_review.csv"
OUTPUT_CSV = BASE_DIR / "evaluation" / "conflict_fixture_results.csv"

load_dotenv(BASE_DIR / ".env")
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

SEVERITY_ORDER = {"none": 0, "mild": 1, "moderate": 2, "severe": 3}

# Same conflict-resolution policy sentence used in fusion/fusion.py's production
# SYSTEM_PROMPT, applied here directly to the two severity signals the review sheet
# records (image-derived, text-derived), since the sheet does not carry per-row
# resource/population entity detail for the full 3-signal fuse() call.
SYSTEM_PROMPT = (
    "You are the conflict-resolution step of a disaster relief triage system's fusion layer. "
    "You are given a severity signal derived from an image and a severity signal derived from "
    "an accompanying text report for the same disaster event. Severity levels, from least to "
    "most severe, are: none, mild, moderate, severe. "
    "Conflict policy: whenever the two signals disagree, ALWAYS resolve to whichever signal is "
    "more severe. Do not average, do not split the difference, and do not default to one signal "
    "by ignoring the other."
)


class ConflictResolution(BaseModel):
    resolved_severity: str = Field(description="One of: none, mild, moderate, severe")


def resolve(image_severity: str, text_severity: str) -> str:
    prompt = (
        f"Image-derived severity signal: {image_severity}\n"
        f"Text-derived severity signal: {text_severity}\n"
        "Apply the conflict policy and return the resolved final severity."
    )
    response = client.models.generate_content(
        model="gemini-flash-latest",
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=ConflictResolution,
        ),
    )
    return response.parsed.resolved_severity


def main():
    with open(INPUT_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    results = []
    done_ids = set()
    if OUTPUT_CSV.exists():
        with open(OUTPUT_CSV, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                r["is_conflict"] = r["is_conflict"] == "True"
                r["correct"] = r["correct"] == "True"
                results.append(r)
                done_ids.add(r["pair_id"])
        print(f"resuming: {len(done_ids)} rows already done")

    for i, row in enumerate(rows):
        if row["pair_id"] in done_ids:
            continue
        img_sev = row["image_severity_signal"]
        txt_sev = row["text_severity_signal"]
        expected = img_sev if SEVERITY_ORDER[img_sev] >= SEVERITY_ORDER[txt_sev] else txt_sev
        is_conflict = img_sev != txt_sev

        for attempt in range(6):
            try:
                actual = resolve(img_sev, txt_sev)
                break
            except Exception as e:  # noqa: BLE001
                if attempt == 5:
                    raise
                print(f"retry {row['pair_id']}: {e}", file=sys.stderr)
                time.sleep(15 * (attempt + 1))

        correct = actual == expected
        results.append({
            "pair_id": row["pair_id"],
            "conflict_type": row["conflict_type"],
            "image_severity_signal": img_sev,
            "text_severity_signal": txt_sev,
            "is_conflict": is_conflict,
            "expected_resolution": expected,
            "actual_resolution": actual,
            "correct": correct,
        })
        print(f"[{i+1}/{len(rows)}] {row['pair_id']}: img={img_sev} text={txt_sev} "
              f"-> expected={expected} actual={actual} {'OK' if correct else 'MISS'}")

        with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)

        time.sleep(13)

    # Aggregate summary
    by_type = {}
    for r in results:
        by_type.setdefault(r["conflict_type"], []).append(r)

    print("\n=== Aggregate results ===")
    for ctype, rs in sorted(by_type.items()):
        n = len(rs)
        n_correct = sum(1 for r in rs if r["correct"])
        n_conflict = sum(1 for r in rs if r["is_conflict"])
        print(f"{ctype}: n={n}, correct={n_correct}/{n} ({100*n_correct/n:.1f}%), "
              f"actual disagreement rows={n_conflict}/{n}")

    overall_n = len(results)
    overall_correct = sum(1 for r in results if r["correct"])
    conflict_rows = [r for r in results if r["is_conflict"]]
    agreement_rows = [r for r in results if not r["is_conflict"]]
    conflict_correct = sum(1 for r in conflict_rows if r["correct"])
    agreement_correct = sum(1 for r in agreement_rows if r["correct"])
    print(f"\nOverall: {overall_correct}/{overall_n} ({100*overall_correct/overall_n:.1f}%) resolved as expected")
    if conflict_rows:
        print(f"Rows with an actual signal disagreement: {conflict_correct}/{len(conflict_rows)} "
              f"({100*conflict_correct/len(conflict_rows):.1f}%) resolved to the more severe reading")
    if agreement_rows:
        print(f"Rows where signals already agreed: {agreement_correct}/{len(agreement_rows)} "
              f"({100*agreement_correct/len(agreement_rows):.1f}%) left unchanged (no false-positive escalation)")

    print(f"\nwrote {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
