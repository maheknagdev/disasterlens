import sys
import csv
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
from urgency_rubric import category_to_urgency

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_CSV = BASE_DIR / "disasterlens_modules" / "data" / "crisismmd_test.csv"
OUTPUT_CSV = BASE_DIR / "evaluation" / "urgency_sanity_subset.csv"
PER_CATEGORY = 5

with open(INPUT_CSV, newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

# Stratified sample: up to PER_CATEGORY rows per fine_category, so rare categories
# aren't drowned out by the dominant not_humanitarian / other_relevant_information classes.
by_category = {}
for row in rows:
    by_category.setdefault(row["fine_category"], []).append(row)

sample = []
for category, category_rows in sorted(by_category.items()):
    sample.extend(category_rows[:PER_CATEGORY])

with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["tweet_id", "tweet_text", "fine_category", "proxy_urgency", "mahek_urgency", "pratyusha_urgency"])
    for row in sample:
        writer.writerow([
            row["tweet_id"],
            row["tweet_text"],
            row["fine_category"],
            category_to_urgency(row["fine_category"]),
            "",
            "",
        ])

print(f"wrote {len(sample)} rows across {len(by_category)} categories to {OUTPUT_CSV}")
