#!/usr/bin/env python3
"""Run Tesseract over every screenshot in data/raw/, writing one JSON per screenshot
to data/ocr/ with full text, per-line and per-word text with bounding boxes and
Tesseract confidences — the structure downstream record-attachment and word-error
analysis consume.

- Skips screenshots that already have OCR output (resume-safe); --force redoes all.
- Reads the capture-time sidecar (<stem>.meta.json) when present.
- Logs per-screenshot failures to data/ocr/_errors.json instead of crashing the batch.

Usage:
    python3 scripts/run_ocr.py                     # batch over data/raw/
    python3 scripts/run_ocr.py --force             # re-OCR everything
    python3 scripts/run_ocr.py --show <substring>  # print stored OCR text of one screenshot
    python3 scripts/run_ocr.py --lang eng          # tesseract language (default eng)
"""

import csv
import io
import json
import subprocess
import sys
from pathlib import Path
from statistics import fmean

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_RAW = DATA_DIR / "raw"
DATA_OCR = DATA_DIR / "ocr"
ERRORS_PATH = DATA_OCR / "_errors.json"
LOW_CONF = 60.0


def union_bbox(words):
    return [
        min(w["bbox"][0] for w in words),
        min(w["bbox"][1] for w in words),
        max(w["bbox"][0] + w["bbox"][2] for w in words) - min(w["bbox"][0] for w in words),
        max(w["bbox"][1] + w["bbox"][3] for w in words) - min(w["bbox"][1] for w in words),
    ]


def parse_tsv(tsv: str):
    rows = csv.DictReader(io.StringIO(tsv), delimiter="\t", quoting=csv.QUOTE_NONE)
    words = []
    for r in rows:
        try:
            conf = float(r["conf"])
            text = (r.get("text") or "").strip()
            if conf < 0 or not text:
                continue
            words.append({
                "text": text,
                "conf": conf,
                "bbox": [int(r["left"]), int(r["top"]), int(r["width"]), int(r["height"])],
                "line_key": (int(r["page_num"]), int(r["block_num"]), int(r["par_num"]), int(r["line_num"])),
            })
        except (ValueError, KeyError, TypeError):
            continue

    lines = {}
    for w in words:
        lines.setdefault(w.pop("line_key"), []).append(w)
    line_list = []
    for key in sorted(lines):
        ws = lines[key]
        line_list.append({
            "text": " ".join(w["text"] for w in ws),
            "conf": round(fmean(w["conf"] for w in ws), 1),
            "bbox": union_bbox(ws),
            "words": ws,
        })
    return line_list


def ocr_image(img: Path, lang: str):
    proc = subprocess.run(
        ["tesseract", str(img), "stdout", "-l", lang, "tsv"],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"tesseract exit {proc.returncode}: {proc.stderr.strip()[:300]}")
    lines = parse_tsv(proc.stdout)
    all_confs = [w["conf"] for line in lines for w in line["words"]]
    mean_conf = round(fmean(all_confs), 1) if all_confs else 0.0
    flags = []
    if not lines:
        flags.append("blank")
    elif mean_conf < LOW_CONF:
        flags.append("low_confidence")
    return {
        "full_text": "\n".join(line["text"] for line in lines),
        "lines": lines,
        "stats": {"word_count": len(all_confs), "line_count": len(lines), "mean_word_conf": mean_conf},
        "flags": flags,
    }


def load_capture_time(img: Path):
    sidecar = img.parent / (img.stem + ".meta.json")
    if sidecar.exists():
        try:
            return json.loads(sidecar.read_text(encoding="utf-8")).get("capture_time")
        except (json.JSONDecodeError, OSError):
            return None
    return None


def main() -> int:
    args = sys.argv[1:]
    force = "--force" in args
    lang = "eng"
    if "--lang" in args:
        lang = args[args.index("--lang") + 1]
    show = None
    if "--show" in args:
        show = args[args.index("--show") + 1]

    if show:
        matches = sorted(DATA_OCR.glob("*" + show + "*.json"))
        matches = [m for m in matches if not m.name.startswith("_")]
        if not matches:
            print(f"no OCR output matching '{show}' in {DATA_OCR}", file=sys.stderr)
            return 1
        print(matches[0].read_text(encoding="utf-8"))
        return 0

    if not DATA_RAW.is_dir():
        print(f"error: {DATA_RAW} does not exist", file=sys.stderr)
        return 1
    DATA_OCR.mkdir(parents=True, exist_ok=True)

    images = sorted(p for p in DATA_RAW.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not images:
        print(f"no screenshots found in {DATA_RAW}")
        return 0

    written = skipped = 0
    errors = []
    for img in images:
        out = DATA_OCR / (img.stem + ".json")
        if out.exists() and not force:
            skipped += 1
            continue
        try:
            result = ocr_image(img, lang)
        except Exception as exc:  # noqa: BLE001 — log per screenshot, keep the batch going
            errors.append({"file": img.name, "error": str(exc)})
            print(f"  ERROR  {img.name}: {exc}", file=sys.stderr)
            continue
        result["file"] = img.name
        result["lang"] = lang
        result["capture_time"] = load_capture_time(img)
        out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        note = f" flags={','.join(result['flags'])}" if result["flags"] else ""
        print(f"  wrote {out.name}: {result['stats']['word_count']} words, conf {result['stats']['mean_word_conf']}{note}")
        written += 1

    ERRORS_PATH.write_text(json.dumps(errors, indent=2) + "\n", encoding="utf-8")
    print(f"{len(images)} screenshots: {written} OCR'd, {skipped} skipped (already exist), {len(errors)} errors")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
