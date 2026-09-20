#!/usr/bin/env python3
"""Jev mapper arm (issues #8 stretch + #12 shootout): TypeSafe Jev (System One)
choice-over-enum, the official API at api.typesafe.ai/v1/systemone.

The task's exact primitive: one Choice question per transaction — state =
vendor + printed bank label (same text the #11 hybrid mapper sees), criteria =
the frozen enum strings verbatim with the same meaning definitions the hybrid
mapper gets, so both mappers see identical information. Zero-shot and
data-free: ground-truth labels never enter this arm, so there is no LOO
protocol — every record is simply predicted (unlike the sklearn/TabFM arms,
which must hold groups out).

OOD FLAG (issue #8): Jev is a general System-1 decision model, out-of-
distribution for bank-transaction text; every output of this arm is marked
ood=true and its number must stay out of the pass/fail headline in REPORT.md.

Auth: JEV_API_KEY from the environment or .env (gitignored). Writes
data/out/jev/<stem>.json in the shared record schema, resume-safe, --force
redoes all, --show <substring> prints one screenshot's records.

Usage:
    uv run python scripts/run_jev_arm.py
    uv run python scripts/run_jev_arm.py --force
    uv run python scripts/run_jev_arm.py --show IMG_6767
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_hybrid_arm import clean_printed, extract_records, y_sorted  # noqa: E402
from sticky_dates import parse_capture_time  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA_OCR = ROOT / "data" / "ocr"
DATA_OUT = ROOT / "data" / "out" / "jev"
ERRORS_PATH = DATA_OUT / "_errors.json"

API_URL = "https://api.typesafe.ai/v1/systemone"
MODEL = "jev-latest"
CATEGORIES = json.loads((ROOT / "categories.json").read_text(encoding="utf-8"))["categories"]

# same meanings the hybrid mapper's enum definition carries, + withdraw (2026-09-20 amendment)
CRITERIA = {
    "grocery&eat out": "buying food to cook or ready meals",
    "health & fitness": "pharmacy, medical, gym, sport",
    "tuition fee": "school or university payments",
    "deposit": "cash deposits, incoming money, income",
    "withdraw": "cash withdrawals, ATM cash-outs",
    "transport": "public transport, fuel, rides",
    "Misc and souvenir": "everything else, gifts, utilities, donations",
}


def load_api_key() -> str:
    key = os.environ.get("JEV_API_KEY")
    if key:
        return key
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("JEV_API_KEY="):
                return line.split("=", 1)[1].strip().strip("'\"")
    print("error: JEV_API_KEY not set (env or .env)", file=sys.stderr)
    raise SystemExit(1)


def jev_choice(key: str, state: str) -> tuple[str | None, float | None, str | None]:
    """One Choice question over the frozen enum; returns (choice, confidence, model_ver)."""
    body = json.dumps({
        "state": state,
        "model": MODEL,
        "questions": {
            "category": {
                "type": "choice",
                "instructions": "Which enum category best fits this bank transaction?",
                "criteria": CRITERIA,
            },
        },
    }).encode("utf-8")
    req = urllib.request.Request(API_URL, data=body, method="POST", headers={
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    })
    last_exc = None
    for attempt in range(3):  # retry transient failures, back off politely
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            answer = payload["answers"]["category"]
            choice = answer.get("choice")
            return (choice if choice in CATEGORIES else None,
                    answer.get("confidence"), payload.get("model"))
        except (urllib.error.URLError, urllib.error.HTTPError, KeyError, json.JSONDecodeError) as exc:
            last_exc = exc
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"jev api failed after 3 attempts: {last_exc}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", help="print the stored output for one screenshot")
    ap.add_argument("--force", action="store_true", help="re-extract even if output exists")
    args = ap.parse_args()

    if args.show:
        matches = [m for m in sorted(DATA_OUT.glob("*" + args.show + "*.json")) if not m.name.startswith("_")]
        if not matches:
            print(f"no jev-arm output matching '{args.show}' in {DATA_OUT}", file=sys.stderr)
            return 1
        print(matches[0].read_text(encoding="utf-8"))
        return 0

    if not DATA_OCR.is_dir():
        print(f"error: {DATA_OCR} does not exist — run scripts/run_ocr.py first", file=sys.stderr)
        return 1
    DATA_OUT.mkdir(parents=True, exist_ok=True)
    key = load_api_key()

    ocr_files = sorted(p for p in DATA_OCR.glob("*.json") if not p.name.startswith("_"))
    todo = []
    for ocr_path in ocr_files:
        out = DATA_OUT / ocr_path.name
        if out.exists() and not args.force:
            continue
        todo.append(ocr_path)
    if not todo:
        print(f"{len(ocr_files)} screenshots: 0 extracted, {len(ocr_files)} skipped (already exist)")
        return 0

    written = skipped = len(ocr_files) - len(todo)
    errors = []
    counts = []
    for ocr_path in todo:
        try:
            ocr = json.loads(ocr_path.read_text(encoding="utf-8"))
            capture = parse_capture_time(ocr.get("capture_time"))
            records = extract_records(y_sorted(ocr), capture)
            model_ver = None
            for r in records:
                r["category_printed"] = clean_printed(r["category_printed"])
                if r["vendor"] or r["category_printed"]:
                    state = f"Vendor: {r['vendor'] or 'unknown'}\nPrinted bank label: {r['category_printed'] or 'none'}"
                    r["category"], r["category_confidence"], model_ver = jev_choice(key, state)
                else:
                    r["category"], r["category_confidence"] = None, None
                r["ood"] = True
        except Exception as exc:  # noqa: BLE001 — log per screenshot, keep the batch going
            errors.append({"file": ocr_path.name, "error": str(exc)})
            print(f"  ERROR  {ocr_path.name}: {exc}", file=sys.stderr)
            counts.append((ocr_path.name, "ERROR"))
            continue
        payload = {
            "file": ocr["file"],
            "engine": f"hybrid extraction + TypeSafe Jev choice mapper ({model_ver or MODEL}) — OOD arm",
            "ood": True,
            "n_records": len(records),
            "records": records,
        }
        out = DATA_OUT / ocr_path.name
        out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        counts.append((out.name, payload["n_records"]))
        summary = "  ".join(f"{r['category_printed']}->{r['category']}" for r in records)
        print(f"  wrote {out.name}: {payload['n_records']} records  {summary}")
        written += 1

    ERRORS_PATH.write_text(json.dumps(errors, indent=2) + "\n", encoding="utf-8")
    print(f"{len(ocr_files)} screenshots: {written} extracted, {skipped} skipped (already exist), {len(errors)} errors")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
