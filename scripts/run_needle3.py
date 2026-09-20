#!/usr/bin/env python3
"""needle3 extraction tracer (issue #5): one screenshot -> grammar-constrained records.

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

Usage:
    uv run python scripts/run_needle3.py                    # first screenshot
    uv run python scripts/run_needle3.py --file IMG_6767    # pick by substring
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


def pick_ocr(substring: str | None) -> Path:
    candidates = sorted(p for p in DATA_OCR.glob("*.json") if not p.name.startswith("_"))
    if substring:
        candidates = [c for c in candidates if substring in c.name]
    if not candidates:
        raise SystemExit(f"no OCR output in {DATA_OCR} — run scripts/run_ocr.py first")
    return candidates[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", help="substring of the OCR file to trace (default: first)")
    ap.add_argument("--max-tokens", type=int, default=4096)
    args = ap.parse_args()

    ocr_path = pick_ocr(args.file)
    ocr = json.loads(ocr_path.read_text(encoding="utf-8"))
    capture = parse_capture_time(ocr.get("capture_time"))

    system = SYSTEM_TEMPLATE.format(capture_date=capture.isoformat())
    query = QUERY_TEMPLATE.format(text=y_sorted_text(ocr))

    with Needle(tools=[Transactions], system=system) as agent:
        envelope = agent.complete(query, max_new_tokens=args.max_tokens)

    DATA_OUT.mkdir(parents=True, exist_ok=True)
    ungrounded = (envelope.get("validation") or {}).get("ungrounded") or []

    calls = envelope.get("function_calls") or envelope.get("suppressed_calls") or []
    records = []
    for call in calls:
        for t in (call.get("arguments") or {}).get("transactions") or []:
            records.append({
                "vendor": t.get("vendor"),
                "date": t.get("date"),
                "amount": t.get("amount"),
                "category": t.get("category"),
                "category_in_enum": t.get("category") in CATEGORIES,
                "suppressed": not envelope.get("function_calls"),
            })

    out = {
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
    out_path = DATA_OUT / ocr_path.name
    out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")

    print(f"screenshot: {ocr['file']}")
    print(f"envelope confidence: {out['envelope_confidence']}  suppressed call: {'yes' if not envelope.get('function_calls') else 'no'}")
    print(f"records: {out['n_records']}")
    for r in records:
        print(f"  {str(r['date']):>10}  {str(r['amount']):>12}  {r['category']}  | {r['vendor']}")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
