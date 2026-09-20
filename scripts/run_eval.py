#!/usr/bin/env python3
"""Eval harness (issue #7): score one extraction arm's output against the frozen labels.

Matching: extracted records are pooled across the corpus and matched 1:1 (greedy)
to labels on exact (date, amount). A null extracted date never matches a dated
label — an unresolved date is a miss, per schema ("never a guess"). Vendor is
never scored (passthrough). Extra records (unmatched extractions — phantoms,
engine collapse dups, cross-screenshot re-showings of the same transaction, and
transactions outside label coverage) all count as extras; missing labels count
as misses and are attributed to a failure cohort:

- ocr_miss       — the label's amount appears in no OCR text at all (OCR's fault)
- field_attach   — the amount IS in some OCR text but no record carried it with
                   the right date (the model's attachment failure)
- category_confusion — matched pair where the emitted category is wrong
                   (scored only for arms that emit a category)

Per-field accuracy note: matching on (date, amount) makes date/amount accuracy
vacuous on matched pairs, so the scope's date/amount bars are checked against
record recovery (fraction of labels recovered with exact date+amount) and the
category bar against category accuracy over matched pairs. Written to
data/out/eval/<arm>.json and printed.

Usage:
    uv run python scripts/run_eval.py --arm regex
    uv run python scripts/run_eval.py --arm needle3
    uv run python scripts/run_eval.py --arm hybrid
"""

import argparse
import csv
import json
import sys
from datetime import date, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_OCR = ROOT / "data" / "ocr"
DATA_EVAL = ROOT / "data" / "out" / "eval"
LABELS_PATH = DATA_RAW / "ground_truth.csv"

ARMS = {"regex": ROOT / "data" / "out" / "regex",
        "needle3": ROOT / "data" / "out" / "needle3",
        "hybrid": ROOT / "data" / "out" / "hybrid",
        "sklearn": ROOT / "data" / "out" / "sklearn"}

BARS = {"record_recovery": 90.0, "category_accuracy": 85.0}


def parse_label_date(raw: str) -> str | None:
    """Labels use d/m/yyyy."""
    try:
        return datetime.strptime(raw.strip(), "%d/%m/%Y").date().isoformat()
    except ValueError:
        return None


def norm_amount(value) -> float | None:
    try:
        return float(str(value).replace(",", "").replace("$", "").replace(" ", ""))
    except (TypeError, ValueError):
        return None


def load_labels() -> list[dict]:
    if not LABELS_PATH.exists():
        raise SystemExit(f"no labels at {LABELS_PATH} — close #3 first")
    labels = []
    with open(LABELS_PATH, encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            amount = norm_amount(row.get("total"))
            labels.append({
                "date": parse_label_date(row.get("date") or ""),
                "amount": amount,
                "category": (row.get("category") or "").strip(),
            })
    return [l for l in labels if l["amount"] is not None]


def load_arm(arm: str) -> list[dict]:
    arm_dir = ARMS[arm]
    if not arm_dir.is_dir():
        raise SystemExit(f"no output directory {arm_dir} for arm '{arm}'")
    records = []
    for path in sorted(arm_dir.glob("*.json")):
        if path.name.startswith("_"):
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        for rec in payload.get("records") or []:
            records.append({
                "file": payload.get("file"),
                "date": rec.get("date"),
                "amount": norm_amount(rec.get("amount")),
                "category": rec.get("category"),
                "vendor": rec.get("vendor"),
            })
    return records


def parse_amount_token(token: str) -> set[float]:
    """OCR amounts arrive as '$4,500.00' (comma-thousands) or '$3,30' (comma-decimal noise);
    try both readings plus the raw token."""
    parses = set()
    base = token.strip("$-+").strip()
    for candidate in (base, base.replace(",", ""), base.replace(",", ".")):
        try:
            parses.add(round(float(candidate), 2))
        except ValueError:
            continue
    return parses


def ocr_amounts() -> set[float]:
    """Every amount that appears anywhere in the corpus OCR text."""
    found = set()
    for path in DATA_OCR.glob("*.json"):
        if path.name.startswith("_"):
            continue
        text = json.loads(path.read_text(encoding="utf-8")).get("full_text") or ""
        for token in text.split():
            found |= parse_amount_token(token)
    return found


def greedy_match(labels: list[dict], records: list[dict]):
    """1:1 match on exact (date, amount); null date only matches a null-dated label."""
    used = [False] * len(records)
    matches, missing = [], []
    for label in labels:
        hit = None
        for i, rec in enumerate(records):
            if not used[i] and rec["amount"] == label["amount"] and rec["date"] == label["date"]:
                hit = i
                break
        if hit is None:
            missing.append(label)
        else:
            used[hit] = True
            matches.append((label, records[hit]))
    extras = [r for i, r in enumerate(records) if not used[i]]
    return matches, missing, extras


def score(arm: str) -> dict:
    labels = load_labels()
    records = load_arm(arm)
    matches, missing, extras = greedy_match(labels, records)
    ocr = ocr_amounts()

    ocr_miss = [l for l in missing if round(abs(l["amount"]), 2) not in ocr]
    field_attach = [l for l in missing if round(abs(l["amount"]), 2) in ocr]

    scored_categories = [(l, r) for l, r in matches if l["category"] and r.get("category") is not None]
    cat_correct = [(l, r) for l, r in scored_categories if r["category"] == l["category"]]
    cat_confusions = [(l, r) for l, r in scored_categories if r["category"] != l["category"]]

    n_labels, n_records = len(labels), len(records)
    recovery = 100.0 * len(matches) / n_labels if n_labels else 0.0
    precision = 100.0 * len(matches) / n_records if n_records else 0.0
    cat_acc = 100.0 * len(cat_correct) / len(scored_categories) if scored_categories else None

    return {
        "arm": arm,
        "n_labels": n_labels,
        "n_records": n_records,
        "matched": len(matches),
        "missing": len(missing),
        "extra_records": len(extras),
        "record_recovery_pct": round(recovery, 1),
        "record_precision_pct": round(precision, 1),
        "category_scored": len(scored_categories),
        "category_correct": len(cat_correct),
        "category_accuracy_pct": round(cat_acc, 1) if cat_acc is not None else None,
        "cohorts": {
            "ocr_miss": [{"date": l["date"], "amount": l["amount"]} for l in ocr_miss],
            "field_attach": [{"date": l["date"], "amount": l["amount"]} for l in field_attach],
            "category_confusion": [{"label": l["category"], "predicted": r["category"],
                                    "amount": l["amount"]} for l, r in cat_confusions],
        },
        "bars": BARS,
        "matches": [{"date": l["date"], "amount": l["amount"], "label_category": l["category"],
                     "predicted_category": r.get("category")} for l, r in matches],
        "extras": [{"file": r.get("file"), "date": r["date"], "amount": r["amount"],
                    "category": r.get("category")} for r in extras],
    }


def report(res: dict) -> None:
    print(f"arm: {res['arm']}  |  labels: {res['n_labels']}  extracted: {res['n_records']}  "
          f"matched: {res['matched']}  missing: {res['missing']}  extra: {res['extra_records']}")
    print(f"{'metric':<34}{'value':>8}{'bar':>9}  pass")
    row = lambda m, v, bar: print(f"{m:<34}{v:>7.1f}%{bar:>8.0f}%  {'PASS' if v >= bar else 'FAIL'}")
    row("record recovery (date+amount)", res["record_recovery_pct"], BARS["record_recovery"])
    row("record precision", res["record_precision_pct"], 0.0)  # measured, no bar in scope
    if res["category_accuracy_pct"] is None:
        print(f"{'category accuracy':<34}{'n/a':>8}{BARS['category_accuracy']:>8.0f}%  (arm emits no category)")
    else:
        row("category accuracy", res["category_accuracy_pct"], BARS["category_accuracy"])
    c = res["cohorts"]
    print(f"cohorts: ocr_miss={len(c['ocr_miss'])}  field_attach={len(c['field_attach'])}  "
          f"category_confusion={len(c['category_confusion'])}  extra={res['extra_records']}"
          f"  (extras include cross-screenshot re-showings and label-coverage gap)")
    for kind in ("ocr_miss", "field_attach"):
        for item in c[kind]:
            print(f"  {kind:<18} {item['date']}  {item['amount']}")
    for item in c["category_confusion"]:
        print(f"  category_confusion  {item['amount']:>10}  label={item['label']!r}  got={item['predicted']!r}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    args = ap.parse_args()

    res = score(args.arm)
    DATA_EVAL.mkdir(parents=True, exist_ok=True)
    out = DATA_EVAL / f"{args.arm}.json"
    out.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")

    report(res)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
