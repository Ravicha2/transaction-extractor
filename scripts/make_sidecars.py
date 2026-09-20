#!/usr/bin/env python3
"""Write a capture-time sidecar (<name>.meta.json) next to every screenshot in data/raw/.

The capture timestamp comes from the file's birth time (macOS/Windows creation time),
falling back to mtime where birth time is unavailable. Sidecars are written once and
never overwritten on re-runs, because the capture time is ground truth used later to
resolve sticky date headers ("Today", "Yesterday", "Tue 15 Sep").

Usage:
    python3 scripts/make_sidecars.py            # scan data/raw/, skip existing sidecars
    python3 scripts/make_sidecars.py --force    # rewrite sidecars from current file times
"""

import json
import sys
from datetime import datetime
from pathlib import Path

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".heic", ".webp"}
DATA_RAW = Path(__file__).resolve().parent.parent / "data" / "raw"


def capture_time(path: Path) -> tuple[str, str]:
    st = path.stat()
    birth = getattr(st, "st_birthtime", None)
    if birth is not None:
        source = "birthtime"
        ts = birth
    else:
        source = "mtime (birthtime unavailable)"
        ts = st.st_mtime
    iso = datetime.fromtimestamp(ts).astimezone().isoformat(timespec="seconds")
    return iso, source


def main() -> int:
    force = "--force" in sys.argv[1:]
    if not DATA_RAW.is_dir():
        print(f"error: {DATA_RAW} does not exist", file=sys.stderr)
        return 1

    images = sorted(p for p in DATA_RAW.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not images:
        print(f"no screenshots found in {DATA_RAW}")
        return 0

    written = skipped = 0
    for img in images:
        sidecar = img.parent / (img.stem + ".meta.json")
        if sidecar.exists() and not force:
            skipped += 1
            continue
        iso, source = capture_time(img)
        payload = {
            "file": img.name,
            "capture_time": iso,
            "capture_time_source": source,
        }
        sidecar.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        written += 1
        print(f"  wrote {sidecar.name}: {iso} [{source}]")

    print(f"{len(images)} screenshots: {written} sidecars written, {skipped} skipped (already exist)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
