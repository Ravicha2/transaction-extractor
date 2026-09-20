#!/usr/bin/env python3
"""TabFM mapper arm (issue #12): google-research/tabfm tabular foundation model,
scikit-learn interface, in-context learning over the labeled history.

STATUS (2026-09-20): BUILT BUT DECLARED OPERATIONALLY INFEASIBLE on personal
hardware — never run to completion by protocol. Measured on this machine
(M-series, torch CPU float32): predict = one forward pass of the 24-block ICL
transformer over the full context table PER ensemble member, ~25-40 s per call
fixed cost (1 vs 8 query rows barely differs; fit is just encoding, ~0.7 s).
Group LOO needs one call per label group (33 groups + 1 extras batch = 34
calls) -> 20-30 min per rerun, for a job the sklearn arm does in ~50 ms and
needle3 decodes in seconds. KV-cache-style amortization does not apply: within
a call the context is already shared; across calls every LOO group has a
different 42-row context and the attention is bidirectional, so cached context
representations would be wrong (hit rate zero by construction). MPS benchmarked
slower than CPU at this sequence length. A reduced n_estimators=1 run (~3 s per
call) was scoped as the fallback but the arm was superseded by the Jev arm
(issue #8) before a scored run. The cost asymmetry IS the finding: in-context
tabular has no training to amortize, which is exactly the wrong cost shape for
a 65-record personal corpus.

Protocol kept as written for the record — same as scripts/run_sklearn_arm.py:
shared with the #11 hybrid, training pairs derived by joining extracted records
to ground_truth.csv on (date, amount), GROUP leave-one-out scoring (a scored
row is predicted with its whole label group held out of the fit/context;
unscored extras use the full fit, flagged loo=false). Only the mapper differs:
TabFMClassifier over a 3-column table per transaction —
  vendor (bank descriptor, passthrough), printed label, day-of-week
(derived from the resolved date; raw dates excluded — they would memorize the
calendar and leak the label join).

Engine config (disclosed deviations from TabFM defaults, provisional-arm
justified at 43 labeled rows): n_estimators=8 (default 32 — 4x predict cost for
noise-level fidelity at this data size), torch CPU float32 with 8 threads (MPS
benchmarked slower at this sequence length), deterministic seed 42, ensemble
calibration off (the default), use_amp off. Predictions are batched per label
group: all records sharing a (date, amount) key are one predict call, verified
identical to per-row calls.

Domain facts recorded at build time, deliberately NOT hand-coded into this arm
(arms stay pure and comparable; a hand rule here would flatter TabFM relative
to the already-scored sklearn arm):
- `withdraw` is deterministic lexically: vendor starting 'Withdrawal CBA …'
  (CBA ATM cash-out descriptor) -> withdraw. Any arm failing this row under
  LOO is failing a 1-line regex job, which is itself a finding.
- `tuition fee` is a dying class (last term): its single key will never gain
  siblings, so its guaranteed-wrong status under group LOO is permanent and
  says nothing about the mapper.

Writes data/out/tabfm/<stem>.json in the shared record schema, resume-safe,
--force redoes all, --show <substring> prints one screenshot's records.

Usage:
    uv run python scripts/run_tabfm_arm.py
    uv run python scripts/run_tabfm_arm.py --force
    uv run python scripts/run_tabfm_arm.py --show IMG_6776
"""

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_hybrid_arm import clean_printed, extract_records, y_sorted  # noqa: E402
from sticky_dates import parse_capture_time  # noqa: E402

from tabfm import TabFMClassifier  # noqa: E402
from tabfm.src.pytorch.tabfm_v1_0_0 import load as load_tabfm_model  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA_OCR = ROOT / "data" / "ocr"
DATA_RAW = ROOT / "data" / "raw"
DATA_OUT = ROOT / "data" / "out" / "tabfm"
ERRORS_PATH = DATA_OUT / "_errors.json"
LABELS_PATH = DATA_RAW / "ground_truth.csv"

MAPPER = ("TabFM 1.0.1, google/tabfm-1.0.0-pytorch weights (non-commercial license), "
          "torch CPU float32, n_estimators=8 (default 32, disclosed) tabular ICL on "
          "[vendor, printed label, day-of-week] (group LOO)")
COLUMNS = ["vendor", "printed", "dow"]
N_ESTIMATORS = 8


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


def feats(vendor: str | None, printed: str | None, date: str | None) -> dict:
    dow = ""
    if date:
        try:
            dow = datetime.fromisoformat(date).strftime("%a")
        except ValueError:
            pass
    return {"vendor": vendor or "", "printed": printed or "", "dow": dow}


def fit_predict_batch(model, train: list[dict], y: list[str],
                      queries: list[dict]) -> list[str | None]:
    """Fit the tabular ICL model on training rows, predict a batch of query rows.
    All-None when training is degenerate (same guard as the sklearn arm)."""
    rows = [(r, c) for r, c in zip(train, y) if r["vendor"] or r["printed"]]
    train = [r for r, _ in rows]
    y = [c for _, c in rows]
    if len(train) < 2 or len(set(y)) < 2 or not queries:
        return [None] * len(queries)
    clf = TabFMClassifier(model=model, n_estimators=N_ESTIMATORS,
                          random_state=42, verbose=False, use_amp=False)
    clf.fit(pd.DataFrame(train, columns=COLUMNS), y)
    return [str(p) for p in clf.predict(pd.DataFrame(queries, columns=COLUMNS))]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", help="print the stored output for one screenshot")
    ap.add_argument("--force", action="store_true", help="re-extract even if output exists")
    args = ap.parse_args()

    if args.show:
        matches = [m for m in sorted(DATA_OUT.glob("*" + args.show + "*.json")) if not m.name.startswith("_")]
        if not matches:
            print(f"no tabfm-arm output matching '{args.show}' in {DATA_OUT}", file=sys.stderr)
            return 1
        print(matches[0].read_text(encoding="utf-8"))
        return 0

    if not DATA_OCR.is_dir():
        print(f"error: {DATA_OCR} does not exist — run scripts/run_ocr.py first", file=sys.stderr)
        return 1
    DATA_OUT.mkdir(parents=True, exist_ok=True)

    labels = load_labels()

    # deterministic extraction shared with the hybrid/sklearn arms
    all_records: dict[str, list[dict]] = {}
    for ocr_path in sorted(p for p in DATA_OCR.glob("*.json") if not p.name.startswith("_")):
        ocr = json.loads(ocr_path.read_text(encoding="utf-8"))
        records = extract_records(y_sorted(ocr), parse_capture_time(ocr.get("capture_time")))
        for r in records:
            r["category_printed"] = clean_printed(r["category_printed"])
        all_records[ocr["file"]] = records

    # join to labels -> training rows keyed by label group
    train_rows: dict[tuple[str, float], list[dict]] = {}
    train_ys: dict[tuple[str, float], list[str]] = {}
    by_key: dict[tuple[str, float], list[tuple[str, int, dict]]] = {}
    extras: list[tuple[str, int, dict]] = []
    for fname, records in all_records.items():
        for i, r in enumerate(records):
            x = feats(r["vendor"], r["category_printed"], r["date"])
            key = (r["date"], r["amount"])
            if key in labels and (r["vendor"] or r["category_printed"]):
                train_rows.setdefault(key, []).append(x)
                train_ys.setdefault(key, []).append(labels[key])
                by_key.setdefault(key, []).append((fname, i, x))
            else:
                extras.append((fname, i, x))

    # mapping is corpus-global (group LOO pools rows across screenshots), so
    # resume is all-or-nothing: skip entirely unless --force or an output is missing
    outputs = {fname: DATA_OUT / (fname.rsplit(".", 1)[0] + ".json") for fname in all_records}
    if not args.force and all(p.exists() for p in outputs.values()):
        print(f"{len(all_records)} screenshots: 0 extracted, {len(all_records)} skipped (already exist)")
        return 0

    # pretrained weights: downloaded once from HF (google/tabfm-1.0.0-pytorch,
    # subfolder classification/), cached by huggingface_hub, CPU float32
    try:
        torch.set_num_threads(max(1, torch.get_num_threads() * 2))
        model = load_tabfm_model(model_type="classification")
    except Exception as exc:  # noqa: BLE001 — no model, no arm
        print(f"error: loading TabFM weights failed: {exc}", file=sys.stderr)
        return 1

    written = skipped = 0
    errors = []
    counts = []

    try:
        # scored records: one fit+batched predict per label group (group LOO)
        for n, (key, items) in enumerate(sorted(by_key.items()), 1):
            train = [r for k, rows in train_rows.items() if k != key for r in rows]
            ys = [c for k, cys in train_ys.items() if k != key for c in cys]
            preds = fit_predict_batch(model, train, ys, [x for _, _, x in items])
            for (fname, i, _), p in zip(items, preds):
                all_records[fname][i]["category"] = p
                all_records[fname][i]["loo"] = True
            print(f"  [{n}/{len(by_key)}] group {key}: {len(items)} rows <- "
                  f"{len({c for c in ys})} classes in fit")

        # unscored extras: full fit, one batch
        if extras:
            preds = fit_predict_batch(
                model,
                [r for rows in train_rows.values() for r in rows],
                [c for cys in train_ys.values() for c in cys],
                [x for _, _, x in extras])
            for (fname, i, _), p in zip(extras, preds):
                all_records[fname][i]["category"] = p
                all_records[fname][i]["loo"] = False
            print(f"  extras: {len(extras)} rows (full fit)")
    except Exception as exc:  # noqa: BLE001 — abort batch, per-file write below still runs
        errors.append({"file": "_batch", "error": str(exc)})
        print(f"  ERROR  during mapping: {exc}", file=sys.stderr)

    for fname, records in all_records.items():
        out = outputs[fname]
        payload = {
            "file": fname,
            "engine": f"hybrid extraction + TabFM mapper: {MAPPER}",
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
