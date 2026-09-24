#!/usr/bin/env python3

import json
from pathlib import Path

WMT_FILE = (
    Path(__file__).resolve().parents[2]
    / "wmt25-general-mt"
    / "data"
    / "wmt25-genmt-humeval.jsonl"
)

print("=" * 70)
print("WMT25 HUMAN-EVALUATION DATA INSPECTION")
print("=" * 70)

print(f"\nData file: {WMT_FILE}")

if not WMT_FILE.exists():
    raise FileNotFoundError(
        f"WMT25 data file not found:\n{WMT_FILE}"
    )

with open(WMT_FILE, "r", encoding="utf-8") as f:
    row = json.loads(next(f))

print("\nTOP-LEVEL KEYS")
for key in row:
    print(f"  - {key}")

print("\nDOCUMENT INFORMATION")
print("doc_id:", row.get("doc_id"))
print("src_lang:", row.get("src_lang"))
print("tgt_lang:", row.get("tgt_lang"))

print("\nSOURCE TEXT")
print(row.get("src_text"))

translations = row.get("tgt_text", {})
scores = row.get("scores", {})

print("\nCOUNTS")
print("Systems with translations:", len(translations))
print("Systems with human scores:", len(scores))

print("\nFIRST 10 SYSTEMS WITH HUMAN SCORES")

for system, annotations in list(scores.items())[:10]:
    values = [
        annotation.get("score")
        for annotation in annotations
        if annotation.get("score") is not None
    ]

    print(
        f"{system:30s} "
        f"annotations={len(values):2d} "
        f"scores={values}"
    )
