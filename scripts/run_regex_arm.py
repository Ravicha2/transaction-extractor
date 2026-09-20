#!/usr/bin/env python3
"""Regex control arm: the deliberately dumb baseline for date + amount extraction.

~5 lines of matching logic (marked below) over the same OCR JSON needle3 sees:
- sticky date headers resolved via the shared sticky_dates helper + timestamp sidecars
- one record per amount match, attached to the nearest header line above it (by y)
- no category (regex can't do it — that asymmetry is the headline), vendor passthrough
  left to needle3; records carry only {date, amount} in the shared schema.

Writes data/out/regex/<stem>.json, one command over the full corpus. Resume-safe;
--force redoes all; --show <substring> prints one screenshot's records for the
side-by-side demo against needle3 output.

Usage:
    python3 scripts/run_regex_arm.py
    python3 scripts/run_regex_arm.py --force
    python3 scripts/run_regex_arm.py --show <substring>
"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sticky_dates import parse_capture_time, resolve_header  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_RAW = DATA_DIR / "raw"
DATA_OCR = DATA_DIR / "ocr"
DATA_OUT = DATA_DIR / "out" / "regex"
ERRORS_PATH = DATA_OUT / "_errors.json"

# --- the ~5 lines of matching logic ------------------------------------------
AMOUNT_RE = re.compile(r"[-+]?\$\s*\d{1,3}(?:,\d{3})*\.\d{2}\b")


def extract(lines, capture):
    headers = [(ln["bbox"][1], d) for ln in lines if (d := resolve_header(ln["text"], capture))]
    amounts = [(ln["bbox"][1], float(AMOUNT_RE.search(ln["text"]).group()
               .replace(",", "").replace(" ", "").replace("$", "")))
               for ln in lines if AMOUNT_RE.search(ln["text"])]
    headers.sort()
    return [{"date": next((d.isoformat() for y, d in reversed(headers) if y <= ay), None),
             "amount": round(a, 2)} for ay, a in sorted(amounts)]
# ------------------------------------------------------------------------------


def main() -> int:
    args = sys.argv[1:]
    force = "--force" in args
    show = args[args.index("--show") + 1] if "--show" in args else None

    if show:
        matches = [m for m in sorted(DATA_OUT.glob("*" + show + "*.json")) if not m.name.startswith("_")]
        if not matches:
            print(f"no regex-arm output matching '{show}' in {DATA_OUT}", file=sys.stderr)
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
    for ocr_path in ocr_files:
        out = DATA_OUT / ocr_path.name
        if out.exists() and not force:
            skipped += 1
            continue
        try:
            ocr = json.loads(ocr_path.read_text(encoding="utf-8"))
            capture = parse_capture_time(ocr.get("capture_time"))
            records = extract(ocr["lines"], capture)
        except Exception as exc:  # noqa: BLE001 — log per screenshot, keep the batch going
            errors.append({"file": ocr_path.name, "error": str(exc)})
            print(f"  ERROR  {ocr_path.name}: {exc}", file=sys.stderr)
            continue
        payload = {"file": ocr["file"], "capture_time": ocr.get("capture_time"), "records": records}
        out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"  wrote {out.name}: {len(records)} records")
        written += 1

    ERRORS_PATH.write_text(json.dumps(errors, indent=2) + "\n", encoding="utf-8")
    print(f"{len(ocr_files)} screenshots: {written} extracted, {skipped} skipped (already exist), {len(errors)} errors")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
