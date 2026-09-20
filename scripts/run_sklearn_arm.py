#!/usr/bin/env python3
"""sklearn mapper arm (issue #12): TF-IDF char-n-grams over vendor+printed label ->
LogisticRegression, trained on the labeled history. The no-foundation-model control:
if this clears the 85% category bar, the mapping job is memorization, not semantics.

Reuses the #11 hybrid's deterministic extraction verbatim (same OCR -> records code),
so the only difference from the hybrid arm is the mapper: needle3 -> sklearn.

Training pairs are derived, not relabeled: extracted records join ground_truth.csv on
(date, amount). Scoring protocol is GROUP leave-one-out — when predicting the rows of
one label, the mapper is fit only on rows of the other labels (plain row-LOO would
leak through cross-screenshot duplicates of the same transaction, whose text is
near-identical). Records matching no label (unscored extras) are predicted with the
full fit and flagged loo=false.

Writes data/out/sklearn/<stem>.json in the shared record schema, resume-safe,
--force redoes all, --show <substring> prints one screenshot's records.

Usage:
    uv run python scripts/run_sklearn_arm.py
    uv run python scripts/run_sklearn_arm.py --force
    uv run python scripts/run_sklearn_arm.py --show IMG_6767
"""

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_hybrid_arm import clean_printed, extract_records, y_sorted  # noqa: E402
from sticky_dates import parse_capture_time  # noqa: E402

from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA_OCR = ROOT / "data" / "ocr"
DATA_RAW = ROOT / "data" / "raw"
DATA_OUT = ROOT / "data" / "out" / "sklearn"
ERRORS_PATH = DATA_OUT / "_errors.json"
LABELS_PATH = DATA_RAW / "ground_truth.csv"

MAPPER = "tfidf-char-wb-2-4 + LogisticRegression on vendor+printed label (group LOO)"


def load_labels() -> dict[tuple[str, float], str]:
    labels = {}
    with open(LABELS_PATH, encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                date = datetime.strptime(row["date"].strip(), "%d/%m/%Y").date().isoformat()
                amount = float(row["total"].replace(",", ""))
            except (ValueError, KeyError):
                continue
            labels[(date, round(amount, 2))] = row["category"].strip()
    return labels


def fit_predict(train: list[tuple[str, str]], text: str) -> str | None:
    """Fit on (text, category) rows, predict one text; None when training is degenerate."""
    train = [(t, c) for t, c in train if t]
    if len(train) < 2 or len({c for _, c in train}) < 2:
        return None
    clf = make_pipeline(
        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1),
        LogisticRegression(max_iter=2000),
    )
    clf.fit([t for t, _ in train], [c for _, c in train])
    return clf.predict([text])[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", help="print the stored output for one screenshot")
    ap.add_argument("--force", action="store_true", help="re-extract even if output exists")
    args = ap.parse_args()

    if args.show:
        matches = [m for m in sorted(DATA_OUT.glob("*" + args.show + "*.json")) if not m.name.startswith("_")]
        if not matches:
            print(f"no sklearn-arm output matching '{args.show}' in {DATA_OUT}", file=sys.stderr)
            return 1
        print(matches[0].read_text(encoding="utf-8"))
        return 0

    if not DATA_OCR.is_dir():
        print(f"error: {DATA_OCR} does not exist — run scripts/run_ocr.py first", file=sys.stderr)
        return 1
    DATA_OUT.mkdir(parents=True, exist_ok=True)

    labels = load_labels()

    # deterministic extraction shared with the hybrid arm
    all_records: dict[str, list[dict]] = {}
    for ocr_path in sorted(p for p in DATA_OCR.glob("*.json") if not p.name.startswith("_")):
        ocr = json.loads(ocr_path.read_text(encoding="utf-8"))
        records = extract_records(y_sorted(ocr), parse_capture_time(ocr.get("capture_time")))
        for r in records:
            r["category_printed"] = clean_printed(r["category_printed"])
        all_records[ocr["file"]] = records

    # join to labels -> training rows keyed by label group
    train_rows: dict[tuple[str, float], list[tuple[str, str]]] = {}
    labeled_of: dict[str, dict[int, tuple[str, float]]] = {}
    for fname, records in all_records.items():
        labeled_of[fname] = {}
        for i, r in enumerate(records):
            key = (r["date"], r["amount"])
            if key in labels and (r["vendor"] or r["category_printed"]):
                labeled_of[fname][i] = key
                text = f"{r['vendor'] or ''} {r['category_printed'] or ''}".strip()
                train_rows.setdefault(key, []).append((text, labels[key]))

    written = skipped = 0
    errors = []
    counts = []
    for fname, records in all_records.items():
        out = DATA_OUT / (fname.rsplit(".", 1)[0] + ".json")
        if out.exists() and not args.force:
            skipped += 1
            try:
                counts.append((out.name, json.loads(out.read_text(encoding="utf-8"))["n_records"]))
            except (json.JSONDecodeError, KeyError):
                counts.append((out.name, "?"))
            continue
        try:
            for i, r in enumerate(records):
                key = labeled_of.get(fname, {}).get(i)
                if key:  # scored row: group LOO — train on OTHER label groups only
                    others = [(t, c) for k, rows in train_rows.items() if k != key for t, c in rows]
                    r["category"] = fit_predict(others, f"{r['vendor'] or ''} {r['category_printed'] or ''}".strip())
                    r["loo"] = True
                else:    # unscored extra: full fit
                    everything = [t for rows in train_rows.values() for t in rows]
                    r["category"] = fit_predict(everything, f"{r['vendor'] or ''} {r['category_printed'] or ''}".strip())
                    r["loo"] = False
        except Exception as exc:  # noqa: BLE001 — log per screenshot, keep the batch going
            errors.append({"file": fname, "error": str(exc)})
            print(f"  ERROR  {fname}: {exc}", file=sys.stderr)
            counts.append((out.name, "ERROR"))
            continue
        payload = {
            "file": fname,
            "engine": f"hybrid extraction + sklearn mapper: {MAPPER}",
            "n_records": len(records),
            "records": records,
        }
        out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        counts.append((out.name, payload["n_records"]))
        summary = "  ".join(f"{r['category_printed']}->{r['category']}" for r in records)
        print(f"  wrote {out.name}: {payload['n_records']} records  {summary}")
        written += 1

    ERRORS_PATH.write_text(json.dumps(errors, indent=2) + "\n", encoding="utf-8")
    print(f"{len(all_records)} screenshots: {written} extracted, {skipped} skipped (already exist), {len(errors)} errors")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
