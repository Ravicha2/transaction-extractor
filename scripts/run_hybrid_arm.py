#!/usr/bin/env python3
"""Hybrid arm (issue #11): deterministic day-grouping + regex field extraction +
needle3 used ONLY to map the bank's printed category labels to the frozen enum.

Suggested by the #5/#6/#7 zero-shot findings: needle3 cannot attach fields on
fragmented OCR (0/10 recovered, all field-attach), but is perfect on clean prose
— so regex does everything deterministic and the model does the one judgment
call left: 9->6 enum semantics over noisy printed label strings.

Per screenshot (one command over the corpus, run_ocr.py conventions):
1. lines sorted by y; sticky headers resolved via shared sticky_dates helper
2. per amount line: date = nearest header at/above, vendor = the single line
   immediately above (best-effort passthrough, never scored), printed label =
   the line immediately below
3. needle3 enum mapping ONLY — one grammar-constrained call per transaction,
   vendor + printed label -> {enum: Category} with enum definitions in the field
   description (shape frozen after testing 4 variants, see comment below);
   unmappable labels -> null category and recorded confidence — never a guess.

Writes data/out/hybrid/<stem>.json with records
{vendor, date, amount, category_printed, category, category_confidence}.

Usage:
    uv run python scripts/run_hybrid_arm.py
    uv run python scripts/run_hybrid_arm.py --force
    uv run python scripts/run_hybrid_arm.py --show IMG_6767
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sticky_dates import parse_capture_time, resolve_header  # noqa: E402

from needle import Needle  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA_OCR = ROOT / "data" / "ocr"
DATA_OUT = ROOT / "data" / "out" / "hybrid"
ERRORS_PATH = DATA_OUT / "_errors.json"
CATEGORIES = json.loads((ROOT / "categories.json").read_text(encoding="utf-8"))["categories"]

# amounts: '$4,500.00' comma-thousands or '$3,30' comma-decimal OCR noise
AMOUNT_RE = re.compile(r"[-+]?\$\s*\d{1,3}(?:,\d{3})+[.,]\d{2}\b|[-+]?\$\s*\d+[.,]\d{2}\b")


# Mapping call shape, frozen after testing 4 variants (issue #11): array-of-mappings,
# plain single choice, label-only with enum definitions in the field description (3/8),
# few-shot exemplars (worse — the exemplars became new attractors; consistent with the
# known few-shot-degrades-trivial-tasks result), vendor+label per record (4/9, frozen
# per owner call; vendor flips individual pairs both ways — noise, not signal).
# Honest pattern either way: the 121M model maps only over a lexical bridge
# (Vehicle & transport -> transport); semantically-mediated maps (Income -> deposit,
# Education -> tuition fee) fail zero-shot and land on the current repetition attractor.
ENUM_DEFINITIONS = (
    "Choose the enum category for the transaction. Meanings: "
    "'grocery&eat out' = buying food to cook or ready meals; "
    "'health & fitness' = pharmacy, medical, gym, sport; "
    "'tuition fee' = school or university payments; "
    "'deposit' = cash deposits, incoming money, income; "
    "'transport' = public transport, fuel, rides; "
    "'Misc and souvenir' = everything else, gifts, utilities, donations."
)


class Pick(BaseModel):
    """One transaction mapped to its enum category, judged from vendor + printed bank label."""
    enum: Literal[*CATEGORIES] = Field(description=ENUM_DEFINITIONS)


MAP_SYSTEM = "You are a precise bank-transaction categorizer."


def parse_amount(token: str) -> float | None:
    s = token.replace("$", "").replace(" ", "")
    if "," in s and "." in s:            # 4,500.00 — comma is thousands
        s = s.replace(",", "")
    elif "," in s:                        # 3,30 — OCR comma-decimal noise
        s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None


def y_sorted(ocr: dict) -> list[dict]:
    return sorted(ocr["lines"], key=lambda l: l["bbox"][1])


def extract_records(lines: list[dict], capture) -> list[dict]:
    headers = [(ln["bbox"][1], resolve_header(ln["text"], capture)) for ln in lines]
    records = []
    for i, ln in enumerate(lines):
        m = AMOUNT_RE.search(ln["text"])
        if not m:
            continue
        amount = parse_amount(m.group())
        if amount is None:
            continue
        y = ln["bbox"][1]
        date = next((d for hy, d in reversed(headers) if d and hy <= y), None)
        vendor = lines[i - 1]["text"] if i > 0 else None
        printed = lines[i + 1]["text"].strip() if i + 1 < len(lines) else None
        if printed and (AMOUNT_RE.search(printed) or resolve_header(printed, capture)):
            printed = None
        records.append({
            "vendor": vendor,
            "date": date.isoformat() if date else None,
            "amount": round(amount, 2),
            "category_printed": printed,
        })
    return records


def clean_printed(text: str | None) -> str | None:
    """Strip OCR punctuation junk; drop digit-heavy lines (account numbers etc.)."""
    if not text:
        return None
    t = text.strip().lstrip("-").strip().rstrip(",;").strip()
    if not t or sum(c.isdigit() for c in t) / len(t) > 0.3:
        return None
    return t


def map_record(vendor: str | None, printed: str) -> tuple[str | None, float | None]:
    """One needle3 call per transaction: vendor + printed label -> enum, with the
    engine's envelope confidence recorded as the category-field confidence."""
    query = (f"Vendor: {vendor or 'unknown'}\n"
             f"Printed bank label: {printed}\nBest-fitting enum category:")
    with Needle(tools=[Pick], system=MAP_SYSTEM) as agent:
        envelope = agent.complete(query, max_new_tokens=256)
    calls = envelope.get("function_calls") or envelope.get("suppressed_calls") or []
    for call in calls:
        enum = (call.get("arguments") or {}).get("enum")
        if enum in CATEGORIES:
            return enum, envelope.get("confidence")
    return None, envelope.get("confidence")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", help="print the stored output for one screenshot")
    ap.add_argument("--force", action="store_true", help="re-extract even if output exists")
    args = ap.parse_args()

    if args.show:
        matches = [m for m in sorted(DATA_OUT.glob("*" + args.show + "*.json")) if not m.name.startswith("_")]
        if not matches:
            print(f"no hybrid output matching '{args.show}' in {DATA_OUT}", file=sys.stderr)
            return 1
        print(matches[0].read_text(encoding="utf-8"))
        return 0

    if not DATA_OCR.is_dir():
        print(f"error: {DATA_OCR} does not exist — run scripts/run_ocr.py first", file=sys.stderr)
        return 1
    DATA_OUT.mkdir(parents=True, exist_ok=True)

    ocr_files = sorted(p for p in DATA_OCR.glob("*.json") if not p.name.startswith("_"))
    if not ocr_files:
        print(f"no OCR output found in {DATA_OCR}")
        return 0

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
            capture = parse_capture_time(ocr.get("capture_time"))
            records = extract_records(y_sorted(ocr), capture)
            for r in records:
                r["category_printed"] = clean_printed(r["category_printed"])
                if r["category_printed"]:
                    r["category"], r["category_confidence"] = map_record(
                        r["vendor"], r["category_printed"])
                else:
                    r["category"], r["category_confidence"] = None, None
        except Exception as exc:  # noqa: BLE001 — log per screenshot, keep the batch going
            errors.append({"file": ocr_path.name, "error": str(exc)})
            print(f"  ERROR  {ocr_path.name}: {exc}", file=sys.stderr)
            counts.append((ocr_path.name, "ERROR"))
            continue
        payload = {
            "file": ocr["file"],
            "capture_time": ocr.get("capture_time"),
            "engine": "hybrid: regex fields + needle3 enum mapping (vendor+label per record)",
            "n_records": len(records),
            "records": records,
        }
        out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        counts.append((ocr_path.name, payload["n_records"]))
        summary = "  ".join(f"{r['category_printed']}->{r['category']}" for r in records)
        print(f"  wrote {out.name}: {payload['n_records']} records  {summary}")
        written += 1

    ERRORS_PATH.write_text(json.dumps(errors, indent=2) + "\n", encoding="utf-8")
    print(f"{len(ocr_files)} screenshots: {written} extracted, {skipped} skipped (already exist), {len(errors)} errors")
    print("records per screenshot:")
    for name, n in counts:
        print(f"  {name}: {n}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
