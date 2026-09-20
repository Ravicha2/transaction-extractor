#!/usr/bin/env python3
"""Import pipeline: sheet tracer (issue #13) + plan phase (issue #15).

--tracer-sheet proves the sheet layer of the import pipeline with zero writes
to real ledger data:

1. gws auth preflight — `gws-auth-refresh --check` must exit 0, or we fail fast
   before touching anything (expired token ⇒ "run gws-auth-refresh", non-zero
   exit, no partial writes).
2. Resolve the workbook: SHEET_ID from the environment or .env, overridable
   with `--sheet`.
3. Tab existence check against the live tab list (default: current month, e.g.
   "Sep 26"); a misspelled/missing tab errors with a listing of available tabs.
4. Header validation of the target tab against
   `date | vendor | category | amount | Total` (exact, case-sensitive — a tab
   with legacy headers fails loudly).
5. Round-trip on a self-created `_tracer_<date>` scratch tab: create tab,
   write the expected headers, `values.append` one row (USER_ENTERED,
   INSERT_ROWS), read it back, verify equality, delete the tab. Any failure
   mid-roundtrip best-effort deletes the scratch tab so nothing is left behind.

--plan is the extraction half of the import pipeline, end-to-end on real
screenshots, ending in a reviewable plan with nothing written anywhere:

`--plan --tab 'Sep 26'` scans `data/new/` and per screenshot: birth-time
capture sidecar (make_sidecars.capture_time conventions), Tesseract OCR
(run_ocr.ocr_image), phantom pre-filter (summary/balance lines — 'spend this
month', 'spending', 'available', 'balance' — dropped before extraction so they
can never become records), hybrid field extraction
(run_hybrid_arm.extract_records), one Jev choice call per record (run_jev_arm
patterns). Then ONE values.get collision pass flags (date, amount) pairs
already in the target tab (`already-in-sheet`) and duplicates across this
run's screenshots (`re-shown`); null-date records are pre-flagged (`no-date`)
— review must edit or reject those, they are never silently plan'able as-is.
Output: printed review table `date | vendor | amount | category (conf) |
flags` (+ `--json` for a machine-readable document on stdout, table to stderr)
and a staging JSON per screenshot in `data/out/import/<stem>.json`. The sheet
is only read (values.get, no append); screenshots stay in `data/new/` (zero
file moves); HEIC files are rejected with a `sips -s format png` hint.

The sheet-I/O helpers here (get/append/validation) are the reusable import
block for the later commit slice. No new Python dependencies: everything
shells out to `gws`/`tesseract` or reuses the existing arms.

Usage:
    uv run python scripts/import_screenshots.py --tracer-sheet
    uv run python scripts/import_screenshots.py --tracer-sheet --sheet <ID> --tab "Sep 26"
    uv run python scripts/import_screenshots.py --plan --tab 'Sep 26'
    uv run python scripts/import_screenshots.py --plan --tab 'Sep 26' --json
"""

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_sidecars import capture_time  # noqa: E402
from run_ocr import IMAGE_EXTS, load_capture_time, ocr_image  # noqa: E402
from sticky_dates import parse_capture_time  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA_NEW = ROOT / "data" / "new"
DATA_IMPORT = ROOT / "data" / "out" / "import"
ERRORS_PATH = DATA_IMPORT / "_errors.json"

EXPECTED_HEADERS = ["date", "vendor", "category", "amount", "Total"]

# USER_ENTERED-safe probe row: no cell parses to a date, numbers read back
# verbatim under General format on a fresh tab.
TRACER_ROW_PREFIX = "sheet-tracer"

PLAN_ENGINE = ("plan: run_ocr + phantom pre-filter + hybrid extraction "
               "+ Jev choice mapper (import pipeline, issue #15)")

# Summary/balance furniture the bank app paints above the transaction list;
# their amounts ('spend this month' = -15,670.03 class) must never reach staging.
PHANTOM_RE = re.compile(
    r"\bspend\s+this\s+month\b|\bspending\b|\bavailable\b|\bbalance\b",
    re.IGNORECASE,
)

HEIC_EXTS = {".heic", ".heif"}


class TracerError(SystemExit):
    """Fatal, user-facing failure. No partial writes precede it."""

    def __init__(self, message: str):
        print(f"ERROR: {message}", file=sys.stderr)
        super().__init__(2)


def run_gws_sheets(method: str, params: dict, body: dict | None = None) -> dict:
    """One gws sheets call; clean JSON on stdout, noise on stderr."""
    cmd = ["gws", "sheets", "spreadsheets", *method.split("--"),
           "--params", json.dumps(params)]
    if body is not None:
        cmd += ["--json", json.dumps(body)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except FileNotFoundError:
        raise TracerError("gws CLI not found on PATH — install it first")
    if proc.returncode != 0:
        raise TracerError(
            f"gws sheets {' '.join(method.split('--'))} failed "
            f"(exit {proc.returncode}): {proc.stderr.strip() or proc.stdout.strip()}")
    return json.loads(proc.stdout)


def gws_values_call(method: str, params: dict, body: dict | None = None) -> dict:
    cmd = ["gws", "sheets", "spreadsheets", "values", method,
           "--params", json.dumps(params)]
    if body is not None:
        cmd += ["--json", json.dumps(body)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except FileNotFoundError:
        raise TracerError("gws CLI not found on PATH — install it first")
    if proc.returncode != 0:
        raise TracerError(
            f"gws values {method} failed (exit {proc.returncode}): "
            f"{proc.stderr.strip() or proc.stdout.strip()}")
    return json.loads(proc.stdout)


def preflight_auth() -> None:
    """Fail fast on a dead gws token before any sheet I/O."""
    try:
        proc = subprocess.run(["gws-auth-refresh", "--check"],
                              capture_output=True, text=True, timeout=120)
    except FileNotFoundError:
        raise TracerError("gws-auth-refresh not found on PATH — cannot preflight auth")
    if proc.returncode != 0:
        raise TracerError(
            "gws token expired or missing — run gws-auth-refresh, then retry. "
            "No sheet I/O attempted.")


def load_env(path: Path = ROOT / ".env") -> dict[str, str]:
    env: dict[str, str] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            env[key.strip()] = value.strip()
    return env


def resolve_sheet_id(args: argparse.Namespace) -> str:
    if args.sheet:
        return args.sheet
    sheet_id = os.environ.get("SHEET_ID") or load_env().get("SHEET_ID")
    if not sheet_id:
        raise TracerError(
            "SHEET_ID not set — add SHEET_ID=<spreadsheet id> to .env "
            "or pass --sheet <ID>")
    return sheet_id


def list_tabs(sheet_id: str) -> list[str]:
    doc = run_gws_sheets("get", {"spreadsheetId": sheet_id,
                                 "fields": "sheets.properties.title"})
    return [s["properties"]["title"] for s in doc.get("sheets", [])]


def require_tab(sheet_id: str, tab: str) -> None:
    tabs = list_tabs(sheet_id)
    if tab not in tabs:
        raise TracerError(
            f"tab {tab!r} not found in spreadsheet {sheet_id}. "
            f"Available tabs: {', '.join(repr(t) for t in tabs)}")


def get_values(sheet_id: str, rng: str) -> list[list[str]]:
    resp = gws_values_call("get", {"spreadsheetId": sheet_id, "range": rng})
    return resp.get("values", [])


def validate_headers(sheet_id: str, tab: str) -> None:
    header_row = get_values(sheet_id, f"'{tab}'!A1:E1")
    got = [cell.strip() for cell in header_row[0]] if header_row else []
    if got != EXPECTED_HEADERS:
        raise TracerError(
            f"tab {tab!r} header validation failed.\n"
            f"  expected: {EXPECTED_HEADERS}\n"
            f"  got:      {got}\n"
            "Fix the tab headers before importing.")


def add_tab(sheet_id: str, title: str) -> int:
    resp = run_gws_sheets("batchUpdate", {"spreadsheetId": sheet_id},
                          {"requests": [{"addSheet": {"properties": {"title": title}}}]})
    return resp["replies"][0]["addSheet"]["properties"]["sheetId"]


def delete_tab(sheet_id: str, sheet_tab_id: int) -> None:
    run_gws_sheets("batchUpdate", {"spreadsheetId": sheet_id},
                   {"requests": [{"deleteSheet": {"sheetId": sheet_tab_id}}]})


def append_rows(sheet_id: str, tab: str, rows: list[list[str]]) -> None:
    gws_values_call("append",
                    {"spreadsheetId": sheet_id, "range": f"'{tab}'!A1",
                     "valueInputOption": "USER_ENTERED",
                     "insertDataOption": "INSERT_ROWS"},
                    {"values": rows})


def roundtrip_scratch_tab(sheet_id: str) -> None:
    """Create → write headers → append one row → read back → verify → delete."""
    stamp = datetime.now().strftime("%Y%m%d")
    tab = f"_tracer_{stamp}"
    row = [f"{TRACER_ROW_PREFIX}-{stamp}", "roundtrip", "Misc", "12.34", "12.34"]

    # A leftover tab from a crashed run would break append verification.
    if tab in list_tabs(sheet_id):
        print(f"  removing stale {tab} from a previous run")
        delete_tab(sheet_id, _tab_id_by_title(sheet_id, tab))

    tab_id = add_tab(sheet_id, tab)
    print(f"  created scratch tab {tab} (sheetId {tab_id})")
    try:
        gws_values_call("update",
                        {"spreadsheetId": sheet_id, "range": f"'{tab}'!A1",
                         "valueInputOption": "USER_ENTERED"},
                        {"values": [EXPECTED_HEADERS]})
        append_rows(sheet_id, tab, [row])
        read_back = get_values(sheet_id, f"'{tab}'!A1:E2")
        if len(read_back) != 2 or [c.strip() for c in read_back[1]] != row:
            raise TracerError(
                f"round-trip mismatch on {tab}.\n"
                f"  appended:  {row}\n"
                f"  read back: {read_back[1:] or 'nothing'}")
        print("  append → read-back verified (USER_ENTERED, INSERT_ROWS)")
    finally:
        delete_tab(sheet_id, tab_id)
        print(f"  deleted scratch tab {tab}")


def _tab_id_by_title(sheet_id: str, title: str) -> int:
    doc = run_gws_sheets("get", {"spreadsheetId": sheet_id,
                                 "fields": "sheets.properties(sheetId,title)"})
    for s in doc.get("sheets", []):
        props = s["properties"]
        if props["title"] == title:
            return props["sheetId"]
    raise TracerError(f"tab {title!r} vanished while resolving its sheetId")


def tracer_sheet(args: argparse.Namespace) -> None:
    preflight_auth()
    print("auth: gws token valid")

    sheet_id = resolve_sheet_id(args)
    tab = args.tab or datetime.now().strftime("%b %y")

    require_tab(sheet_id, tab)
    print(f"workbook {sheet_id}: tab {tab!r} found")

    validate_headers(sheet_id, tab)
    print(f"headers of {tab!r}: {' | '.join(EXPECTED_HEADERS)} ✓")

    roundtrip_scratch_tab(sheet_id)
    print("sheet tracer: round-trip complete, real ledger untouched")


# ---------------------------------------------------------------------------
# Plan phase (issue #15)
# ---------------------------------------------------------------------------

def capture_sidecar(img: Path, stream=sys.stdout) -> str | None:
    """Birth-time capture sidecar next to the screenshot (make_sidecars
    conventions: write once, never overwrite); returns the ISO timestamp."""
    sidecar = img.parent / (img.stem + ".meta.json")
    if not sidecar.exists():
        iso, source = capture_time(img)
        sidecar.write_text(json.dumps({
            "file": img.name,
            "capture_time": iso,
            "capture_time_source": source,
        }, indent=2) + "\n", encoding="utf-8")
        print(f"  sidecar {sidecar.name}: {iso} [{source}]", file=stream)
        return iso
    return load_capture_time(img)


def drop_phantom_lines(ocr: dict) -> tuple[dict, int]:
    lines = [ln for ln in ocr["lines"] if not PHANTOM_RE.search(ln["text"])]
    dropped = len(ocr["lines"]) - len(lines)
    return {**ocr, "lines": lines}, dropped


def extract_screenshot(img: Path, stream=sys.stdout) -> dict:
    """OCR → phantom pre-filter → hybrid fields → one Jev call per record."""
    from run_hybrid_arm import clean_printed, extract_records, y_sorted
    from run_jev_arm import jev_choice, load_api_key

    capture_iso = capture_sidecar(img, stream=stream)
    if not capture_iso:
        raise RuntimeError(
            "no capture time (missing/unreadable sidecar) — sticky dates "
            "unresolvable; re-run make_sidecars conventions on this file")
    capture = parse_capture_time(capture_iso)

    ocr = ocr_image(img, "eng")
    ocr, dropped = drop_phantom_lines(ocr)

    records = extract_records(y_sorted(ocr), capture)
    key = load_api_key()
    for r in records:
        r["category_printed"] = clean_printed(r["category_printed"])
        if r["vendor"] or r["category_printed"]:
            state = (f"Vendor: {r['vendor'] or 'unknown'}\n"
                     f"Printed bank label: {r['category_printed'] or 'none'}")
            r["category"], r["category_confidence"], _ = jev_choice(key, state)
        else:
            r["category"], r["category_confidence"] = None, None
        r["flags"] = []
    return {
        "file": img.name,
        "capture_time": capture_iso,
        "engine": PLAN_ENGINE,
        "phantom_lines_dropped": dropped,
        "records": records,
    }


def parse_sheet_date(raw) -> str | None:
    """Sheet date cell → ISO. Day-first (AUS workbook); d/m without a year is
    read as the current year — false already-in-sheet flags only cost review."""
    s = str(raw or "").strip().strip("'")
    if not s:
        return None
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d/%m/%y",
                "%d %b %Y", "%d %b %y", "%d %B %Y", "%d %B %y"):
        try:
            return datetime.strptime(s, fmt).date().isoformat()
        except ValueError:
            continue
    m = re.fullmatch(r"(\d{1,2})/(\d{1,2})", s)
    if m:
        try:
            return datetime.now().date().replace(
                month=int(m.group(2)), day=int(m.group(1))).isoformat()
        except ValueError:
            return None
    return None


def parse_sheet_amount(raw) -> float | None:
    s = str(raw or "").replace("$", "").replace(",", "").replace(" ", "")
    if not s:
        return None
    try:
        return round(float(s), 2)
    except ValueError:
        return None


def flag_collisions(screens: list[dict], sheet_rows: list[list[str]]) -> dict:
    """One collision pass over the already-fetched tab: (date, amount) already
    in the sheet, re-showing across this run's screenshots, null dates."""
    existing = set()
    for row in sheet_rows[1:]:
        d = parse_sheet_date(row[0] if row else None)
        a = parse_sheet_amount(row[3] if len(row) > 3 else None)
        if d and a is not None:
            existing.add((d, a))

    seen: dict[tuple[str, float], set[int]] = {}
    for si, scr in enumerate(screens):
        for r in scr["records"]:
            if r["date"] and r["amount"] is not None:
                seen.setdefault((r["date"], round(r["amount"], 2)), set()).add(si)
    for (d, a), file_ids in seen.items():
        for si, scr in enumerate(screens):
            for r in scr["records"]:
                if r["date"] == d and round(r["amount"], 2) == a:
                    if (d, a) in existing:
                        r["flags"].append("already-in-sheet")
                    if len(file_ids) > 1:
                        r["flags"].append("re-shown")
    for scr in screens:
        for r in scr["records"]:
            if not r["date"]:
                r["flags"].insert(0, "no-date")

    flags = [f for scr in screens for r in scr["records"] for f in r["flags"]]
    return {
        "already-in-sheet": flags.count("already-in-sheet"),
        "re-shown": flags.count("re-shown"),
        "no-date": flags.count("no-date"),
    }


def write_staging(scr: dict, tab: str) -> Path:
    payload = {
        "file": scr["file"],
        "capture_time": scr["capture_time"],
        "engine": scr["engine"],
        "target_tab": tab,
        "planned_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "phantom_lines_dropped": scr["phantom_lines_dropped"],
        "n_records": len(scr["records"]),
        "needs_review": any(r["flags"] for r in scr["records"]),
        "records": scr["records"],
    }
    stem = Path(scr["file"]).stem
    out = DATA_IMPORT / f"{stem}.json"
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return out


def review_table(scr: dict) -> str:
    header = (f"{'date':<10} | {'vendor':<38} | {'amount':>9} | "
              f"{'category (conf)':<32} | flags")
    lines = [header, "-" * len(header)]
    for r in scr["records"]:
        conf = r["category_confidence"]
        conf_s = f"{conf:.2f}" if isinstance(conf, (int, float)) else "?"
        cat = f"{r['category'] or '—'} ({conf_s})"
        vendor = (r["vendor"] or "—")[:38]
        flags = ",".join(r["flags"]) or "-"
        lines.append(
            f"{r['date'] or '????-??-??':<10} | {vendor:<38} | "
            f"{r['amount']:>9.2f} | {cat:<32} | {flags}")
    return "\n".join(lines)


def plan(args: argparse.Namespace) -> int:
    # --json: stdout carries only the machine document; all human progress
    # (auth, per-screenshot lines, review table) moves to stderr.
    stream = sys.stderr if args.json else sys.stdout
    preflight_auth()
    print("auth: gws token valid", file=stream)

    sheet_id = resolve_sheet_id(args)
    tab = args.tab or datetime.now().strftime("%b %y")

    require_tab(sheet_id, tab)
    print(f"workbook {sheet_id}: tab {tab!r} found", file=stream)

    validate_headers(sheet_id, tab)
    print(f"headers of {tab!r}: {' | '.join(EXPECTED_HEADERS)} ✓", file=stream)

    if not DATA_NEW.is_dir():
        raise TracerError(
            f"{DATA_NEW} does not exist — create it and drop the month's "
            "screenshots there (they stay put; the plan never moves files)")
    candidates = sorted(p for p in DATA_NEW.iterdir()
                        if p.suffix.lower() in IMAGE_EXTS | HEIC_EXTS)
    if not candidates:
        print(f"no screenshots found in {DATA_NEW}", file=stream)
        return 0

    errors = []
    todo = []
    for img in candidates:
        if img.suffix.lower() in HEIC_EXTS:
            hint = (f"HEIC not OCR-able — convert first: "
                    f"sips -s format png '{img}' --out '{DATA_NEW / (img.stem + '.png')}'")
            errors.append({"file": img.name, "error": hint})
            print(f"  ERROR  {img.name}: {hint}", file=stream)
        else:
            todo.append(img)

    DATA_IMPORT.mkdir(parents=True, exist_ok=True)
    screens = []
    for img in todo:
        try:
            scr = extract_screenshot(img, stream=stream)
        except Exception as exc:  # noqa: BLE001 — log per screenshot, keep the batch going
            errors.append({"file": img.name, "error": str(exc)})
            print(f"  ERROR  {img.name}: {exc}", file=stream)
            continue
        dropped = f", {scr['phantom_lines_dropped']} phantom line(s) dropped" \
            if scr["phantom_lines_dropped"] else ""
        print(f"  extracted {img.name}: {len(scr['records'])} record(s){dropped}",
              file=stream)
        screens.append(scr)

    # One values.get collision pass — the only sheet I/O in the plan phase.
    sheet_rows = get_values(sheet_id, f"'{tab}'!A1:E")
    counts = flag_collisions(screens, sheet_rows)

    staged = [write_staging(scr, tab) for scr in screens]
    ERRORS_PATH.write_text(json.dumps(errors, indent=2) + "\n", encoding="utf-8")

    n_records = sum(len(s["records"]) for s in screens)
    for scr, path in zip(screens, staged):
        print(f"\n== {scr['file']}  (capture {scr['capture_time']}) ==", file=stream)
        print(review_table(scr), file=stream)
        flagged = sum(1 for r in scr["records"] if r["flags"])
        note = " — NEEDS REVIEW" if flagged else ""
        print(f"staging: {path}  ({len(scr['records'])} record(s), "
              f"{flagged} flagged{note})", file=stream)

    print(f"\nplan vs {tab!r}: {len(screens)} screenshot(s), {n_records} record(s); "
          f"flags: {counts['already-in-sheet']} already-in-sheet, "
          f"{counts['re-shown']} re-shown, {counts['no-date']} no-date; "
          f"{len(errors)} error(s)", file=stream)
    print("sheet untouched (values.get only); screenshots untouched in data/new/",
          file=stream)

    if args.json:
        doc = {
            "mode": "plan",
            "sheet": sheet_id,
            "tab": tab,
            "planned_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "screenshots": [
                json.loads(p.read_text(encoding="utf-8")) for p in staged],
            "errors": errors,
            "summary": {"n_screenshots": len(screens), "n_records": n_records,
                        **counts},
        }
        print(json.dumps(doc, indent=2))

    return 1 if errors else 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import pipeline (issue #13: sheet tracer, #15: plan phase)")
    parser.add_argument("--tracer-sheet", action="store_true",
                        help="run the sheet-layer round-trip tracer")
    parser.add_argument("--plan", action="store_true",
                        help="plan phase: scan data/new/, extract + flag, write "
                             "staging JSONs — reads the sheet, never writes it")
    parser.add_argument("--json", action="store_true",
                        help="with --plan: machine-readable JSON on stdout "
                             "(review table moves to stderr)")
    parser.add_argument("--sheet", default=None,
                        help="spreadsheet ID override (default: SHEET_ID in .env)")
    parser.add_argument("--tab", default=None,
                        help="target tab (default: current month, e.g. 'Sep 26')")
    args = parser.parse_args()

    if args.tracer_sheet and args.plan:
        raise TracerError("pick one mode: --tracer-sheet or --plan")
    if args.json and not args.plan:
        raise TracerError("--json only applies to --plan")

    if args.tracer_sheet:
        tracer_sheet(args)
    elif args.plan:
        raise SystemExit(plan(args))
    else:
        parser.print_help()
        raise TracerError("pick a mode, e.g. --plan or --tracer-sheet")


if __name__ == "__main__":
    main()
