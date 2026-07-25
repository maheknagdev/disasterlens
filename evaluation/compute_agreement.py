import csv
from pathlib import Path
from sklearn.metrics import cohen_kappa_score

SUBSET_CSV = Path(__file__).resolve().parent / "urgency_sanity_subset.csv"

with open(SUBSET_CSV, newline="", encoding="utf-8") as f:
    rows = [row for row in csv.DictReader(f) if row["mahek_urgency"] and row["pratyusha_urgency"]]

if not rows:
    raise SystemExit("No fully-labeled rows found — fill in mahek_urgency and pratyusha_urgency first.")

mahek = [int(row["mahek_urgency"]) for row in rows]
pratyusha = [int(row["pratyusha_urgency"]) for row in rows]
proxy = [int(row["proxy_urgency"]) for row in rows]

exact_agreement = sum(m == p for m, p in zip(mahek, pratyusha)) / len(rows)
kappa = cohen_kappa_score(mahek, pratyusha, weights="linear")

proxy_vs_mahek = sum(pr == m for pr, m in zip(proxy, mahek)) / len(rows)
proxy_vs_pratyusha = sum(pr == p for pr, p in zip(proxy, pratyusha)) / len(rows)

print(f"labeled rows: {len(rows)}")
print(f"human-human exact agreement: {exact_agreement:.2%}")
print(f"human-human linear-weighted Cohen's kappa: {kappa:.3f}")
print(f"proxy-rubric vs Mahek exact agreement: {proxy_vs_mahek:.2%}")
print(f"proxy-rubric vs Pratyusha exact agreement: {proxy_vs_pratyusha:.2%}")
