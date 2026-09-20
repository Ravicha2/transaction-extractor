#!/usr/bin/env python3
"""needle3 extraction (issues #5 tracer, #6 corpus batch): OCR text -> grammar-constrained records.

Fully local. Env managed by uv (pyproject.toml / uv.lock). Input shape (frozen
for the corpus run, chosen after testing 8 zero-shot variants): OCR lines sorted by
y-coordinate (visual reading order) as the query text; the capture timestamp
pinned as the needle date fact in the system prompt; a pydantic array schema as
the grammar (enum-constrained category, nullable date, exact amounts).

Known zero-shot failure mode (the honest v1 finding): the 121M model grounds
numbers and respects the enum grammar but cannot reliably attach vendor/amount/
date to each other on fragmented OCR text — vendors collapse, amounts repeat,
dates fall back to the capture date. Envelope confidence is scattered and does
not track output quality. All of this is recorded, nothing gated (v1, measurement
only). The eval (#7) decides what the numbers say.

Batch mode follows the run_ocr.py / run_regex_arm.py conventions: skips outputs
that already exist (resume-safe), --force redoes all, per-screenshot failures are
logged to data/out/needle3/_errors.json instead of crashing the batch, and
per-screenshot record counts are printed at the end for a sanity eyeball.

Usage:
    uv run python scripts/run_needle3.py                    # batch over the corpus
    uv run python scripts/run_needle3.py --force            # re-decode everything
    uv run python scripts/run_needle3.py --file IMG_6767    # single-screenshot trace
    uv run python scripts/run_needle3.py --show IMG_6767    # print stored output
    uv run python scripts/run_needle3.py --max-tokens 4096
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sticky_dates import parse_capture_time  # noqa: E402

from needle import Needle  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA_OCR = ROOT / "data" / "ocr"
DATA_OUT = ROOT / "data" / "out" / "needle3"
ERRORS_PATH = DATA_OUT / "_errors.json"
CATEGORIES_PATH = ROOT / "categories.json"

CATEGORIES = json.loads(CATEGORIES_PATH.read_text(encoding="utf-8"))["categories"]
Category = Literal[*CATEGORIES]


class Transaction(BaseModel):
    vendor: str
    date: str | None = None
    amount: float
    category: Category


class Transactions(BaseModel):
    """Every transaction found in the OCR text."""
    transactions: list[Transaction]


SYSTEM_TEMPLATE = (
    "date: {capture_date}; You are a bank statement transcriber. "
    "Sticky date headers (e.g. 'Today', 'Thu 10 Sep') apply to the transactions "
    "listed beneath them until the next header."
)
QUERY_TEMPLATE = "Transcribe every transaction in the OCR text below.\n\nOCR TEXT:\n{text}"


def y_sorted_text(ocr: dict) -> str:
    return "\n".join(l["text"] for l in sorted(ocr["lines"], key=lambda l: l["bbox"][1]))


def decode_one(ocr: dict, max_tokens: int) -> dict:
    """Run the frozen decode over one screenshot's OCR; returns the output payload."""
    capture = parse_capture_time(ocr.get("capture_time"))
    system = SYSTEM_TEMPLATE.format(capture_date=capture.isoformat())
    query = QUERY_TEMPLATE.format(text=y_sorted_text(ocr))

    with Needle(tools=[Transactions], system=system) as agent:
        envelope = agent.complete(query, max_new_tokens=max_tokens)

    calls = envelope.get("function_calls") or envelope.get("suppressed_calls") or []
    suppressed = not envelope.get("function_calls")
    records = []
    for call in calls:
        for t in (call.get("arguments") or {}).get("transactions") or []:
            records.append({
                "vendor": t.get("vendor"),
                "date": t.get("date"),
                "amount": t.get("amount"),
                "category": t.get("category"),
                "category_in_enum": t.get("category") in CATEGORIES,
                "suppressed": suppressed,
            })

    return {
        "file": ocr["file"],
        "capture_time": ocr.get("capture_time"),
        "engine": "needle3",
        "package_version": __import__("needle").__version__,
        "input_shape": "y-sorted OCR lines, capture date as system date fact, pydantic array grammar",
        "envelope_confidence": envelope.get("confidence"),
        "validation": envelope.get("validation"),
        "n_records": len(records),
        "records": records,
        "envelope": envelope,
    }


def print_records(payload: dict) -> None:
    print(f"screenshot: {payload['file']}")
    print(f"envelope confidence: {payload['envelope_confidence']}  "
          f"suppressed call: {'yes' if payload['records'] and payload['records'][0]['suppressed'] else 'no'}")
    print(f"records: {payload['n_records']}")
    for r in payload["records"]:
        print(f"  {str(r['date']):>10}  {str(r['amount']):>12}  {r['category']}  | {r['vendor']}")


def pick(substring: str, directory: Path) -> Path:
    candidates = sorted(p for p in directory.glob("*.json") if not p.name.startswith("_"))
    if substring:
        candidates = [c for c in candidates if substring in c.name]
    if not candidates:
        raise SystemExit(f"no output matching '{substring}' in {directory}")
    return candidates[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", help="single-screenshot trace mode: substring of the OCR file to decode")
    ap.add_argument("--show", help="print the stored output for one screenshot")
    ap.add_argument("--force", action="store_true", help="re-decode even if output already exists")
    ap.add_argument("--max-tokens", type=int, default=4096)
    args = ap.parse_args()

    if args.show:
        print(pick(args.show, DATA_OUT).read_text(encoding="utf-8"))
        return 0

    if not DATA_OCR.is_dir():
        print(f"error: {DATA_OCR} does not exist — run scripts/run_ocr.py first", file=sys.stderr)
        return 1

    if args.file:
        ocr_path = pick(args.file, DATA_OCR)
        ocr = json.loads(ocr_path.read_text(encoding="utf-8"))
        payload = decode_one(ocr, args.max_tokens)
        DATA_OUT.mkdir(parents=True, exist_ok=True)
        out_path = DATA_OUT / ocr_path.name
        out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print_records(payload)
        print(f"wrote {out_path}")
        return 0

    ocr_files = sorted(p for p in DATA_OCR.glob("*.json") if not p.name.startswith("_"))
    if not ocr_files:
        print(f"no OCR output found in {DATA_OCR}")
        return 0
    DATA_OUT.mkdir(parents=True, exist_ok=True)

    written = skipped = 0
    errors = []
    counts = []
    for ocr_path in ocr_files:
        out = DATA_OUT / ocr_path.name
        if out.exists() and not args.force:
            skipped += 1
            try:
                counts.append((ocr_path.name, json.loads(out.read_text(encoding="utf-8"))["n_records"]))
            except (json.JSONDecodeError, KeyError):
                counts.append((ocr_path.name, "?"))
            continue
        try:
            ocr = json.loads(ocr_path.read_text(encoding="utf-8"))
            payload = decode_one(ocr, args.max_tokens)
        except Exception as exc:  # noqa: BLE001 — log per screenshot, keep the batch going
            errors.append({"file": ocr_path.name, "error": str(exc)})
            print(f"  ERROR  {ocr_path.name}: {exc}", file=sys.stderr)
            counts.append((ocr_path.name, "ERROR"))
            continue
        out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        counts.append((ocr_path.name, payload["n_records"]))
        print(f"  wrote {out.name}: {payload['n_records']} records, conf {payload['envelope_confidence']}")
        written += 1

    ERRORS_PATH.write_text(json.dumps(errors, indent=2) + "\n", encoding="utf-8")
    print(f"{len(ocr_files)} screenshots: {written} decoded, {skipped} skipped (already exist), {len(errors)} errors")
    print("records per screenshot:")
    for name, n in counts:
        print(f"  {name}: {n}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
