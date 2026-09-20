#!/usr/bin/env python3
"""Import pipeline: sheet tracer (issue #13) + plan (issue #15) + commit (issue #16).

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
(run_hybrid_arm.extract_records), one Jev call per record carrying two
mixed-type questions: the frozen enum category choice plus a noul
vendor-sanity check — vendors that read as mangled OCR are flagged
`vendor-suspect` for review to edit or reject, never discarded. Then ONE
values.get collision pass flags (date, amount) pairs
already in the target tab (`already-in-sheet`) and duplicates across this
run's screenshots (`re-shown`); null-date records are pre-flagged (`no-date`)
— review must edit or reject those, they are never silently plan'able as-is.
Output: printed review table `date | vendor | amount | category (conf) |
flags` (+ `--json` for a machine-readable document on stdout, table to stderr)
and a staging JSON per screenshot in `data/out/import/<stem>.json`. The sheet
is only read (values.get, no append); screenshots stay in `data/new/` (zero
file moves); HEIC files are rejected with a `sips -s format png` hint.

--commit is the write half: it consumes plan-phase staging JSONs and nothing
else (never re-runs OCR/jev — a retry never re-pays model calls):

`--commit --staging data/out/import/<stem>.json --tab 'Sep 26'` re-validates
the target tab's headers (catching wrong-tab mistakes like legacy-format
'Aug 26'), refuses stale staging (the staging JSON records the tab's row count
at plan time; if the tab's row count changed since, abort — a crashed
mid-append commit also shows up here), refuses records review left flagged
with a null date (edit or reject them in staging first), ensures the `source`
header exists in F1 (added once on a tab's first import), then appends ALL
staging rows in ONE `values.append` (USER_ENTERED, INSERT_ROWS; columns
`date | vendor | category | amount | source` with Total left blank in E for
manual drag-fill). The appended rows are read back and verified before
anything moves. Only after a verified append are the image + sidecar moved
from `data/new/` to `data/processed/` (the staging JSON rides along, so a
committed screenshot can never be committed twice). Invariant: a file in
`processed/` ⟺ its rows are in the sheet. Any append failure leaves files in
`data/new/` and staging untouched.

The sheet-I/O helpers here (get/append/validation) are the reusable import
block for the later commit slice. No new Python dependencies: everything
shells out to `gws`/`tesseract` or reuses the existing arms.

With NO mode flag — the direct human run — the pipeline first opens
data/new/ in the file manager and waits for the reviewer to drop this
month's screenshots in (Enter scans, q aborts, re-prompts while the
folder is empty). It then plans, and reviews day by day, newest day
first: each day's records are listed together and the reviewer keeps
[a]ll, [n]one, exact rows by number (1,3,4,5 — the rest of the day is
dropped), or [e]<num>-edits one (vendor / date / amount / category)
before deciding; [q] aborts at any prompt. Vendor sanity is judged by
the same Jev call as the category (a noul yes/no on the OCR line) and
mangled vendors are flagged `vendor-suspect` — review edits or rejects
them, never discards. Edits apply in place and the grouping re-sorts,
so an edited date moves the record to its own day. The accepted subset
is re-flagged against the already-fetched sheet rows (no extra API
call), staging is written for what was accepted only, and it commits
through the exact --commit path above (one values.append, read-back
verification, then the moves). `q` aborts before any staging write:
nothing appended, nothing moved, screenshots stay in data/new/.
--plan and --commit remain the agent-facing contract underneath; the
keystrokes work single-key on a TTY and line-by-line when stdin is piped.

Usage:
    record     # same as below with no mode flag; wrapper symlinked onto PATH
    uv run python scripts/import_screenshots.py     # plan → review → commit
    uv run python scripts/import_screenshots.py --tracer-sheet
    uv run python scripts/import_screenshots.py --tracer-sheet --sheet <ID> --tab "Sep 26"
    uv run python scripts/import_screenshots.py --plan --tab 'Sep 26'
    uv run python scripts/import_screenshots.py --plan --tab 'Sep 26' --json
    uv run python scripts/import_screenshots.py --commit --tab 'Sep 26' \
        --staging data/out/import/<stem>.json [more.json ...]
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_sidecars import capture_time  # noqa: E402
from run_ocr import IMAGE_EXTS, load_capture_time, ocr_image  # noqa: E402
from sticky_dates import parse_capture_time  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA_NEW = ROOT / "data" / "new"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_IMPORT = ROOT / "data" / "out" / "import"
ERRORS_PATH = DATA_IMPORT / "_errors.json"

EXPECTED_HEADERS = ["date", "vendor", "category", "amount", "Total"]

# USER_ENTERED-safe probe row: no cell parses to a date, numbers read back
# verbatim under General format on a fresh tab.
TRACER_ROW_PREFIX = "sheet-tracer"

PLAN_ENGINE = ("plan: run_ocr + phantom pre-filter + hybrid extraction "
               "+ Jev (category choice + noul vendor sanity) "
               "(import pipeline, issue #15)")

# Summary/balance furniture the bank app paints above the transaction list;
# their amounts ('spend this month' = -15,670.03 class) must never reach staging.
PHANTOM_RE = re.compile(
    r"\bspend\s+this\s+month\b|\bspending\b|\bavailable\b|\bbalance\b",
    re.IGNORECASE,
)

HEIC_EXTS = {".heic", ".heif"}


class TracerError(SystemExit):
    """Fatal, user-facing failure. Commit refuses strictly before any write;
    the one sanctioned post-append failure (read-back mismatch) says so
    explicitly and moves nothing."""

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


def append_rows(sheet_id: str, tab: str, rows: list[list[str]]) -> dict:
    return gws_values_call("append",
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
        r["flags"] = []
        if r["vendor"] or r["category_printed"]:
            state = (f"Vendor: {r['vendor'] or 'unknown'}\n"
                     f"Printed bank label: {r['category_printed'] or 'none'}")
            if r["vendor"]:
                r["category"], r["category_confidence"], _, sane = \
                    jev_record(key, state)
                if sane is not None and sane < VENDOR_SANE_THRESHOLD:
                    r["flags"].append("vendor-suspect")
            else:
                r["category"], r["category_confidence"], _ = \
                    jev_choice(key, state)
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


def display_date(iso: str | None) -> str:
    """ISO → dd/mm/yyyy for human-facing surfaces (AUS workbook); staging
    JSONs and sheet rows stay ISO."""
    if not iso:
        return "??/??/????"
    try:
        return datetime.strptime(iso, "%Y-%m-%d").strftime("%d/%m/%Y")
    except ValueError:
        return iso


# Vendor sanity rides the same Jev call as the category choice: a noul
# (yes/no) question judges whether the OCR vendor line is a clean merchant
# name. Below this threshold the record is flagged vendor-suspect — review
# edits or rejects it; nothing is ever discarded outright. Tunable in the
# #18 shakedown (observed: clean ≈ 0.5, mangled ≈ 0.15).
VENDOR_SANE_THRESHOLD = 0.35


def jev_record(key: str, state: str) -> tuple[str | None, float | None,
                                              str | None, float | None]:
    """One Jev call, two mixed-type questions: the frozen enum category
    choice (payload identical to run_jev_arm.jev_choice) plus a noul
    vendor-sanity check. Returns (category, confidence, model, vendor_sane)
    where vendor_sane is the 0..1 probability the OCR vendor line is a
    clean merchant name (None if the API withheld it)."""
    from run_jev_arm import API_URL, CATEGORIES, CRITERIA, MODEL
    body = json.dumps({
        "state": state,
        "model": MODEL,
        "questions": {
            "category": {
                "type": "choice",
                "instructions": "Which enum category best fits this bank transaction?",
                "criteria": CRITERIA,
            },
            "vendor_sane": {
                "type": "noul",
                "instructions": ("The vendor string above was read off a bank screenshot "
                                 "by OCR. It is a clean, correctly-read merchant name "
                                 "with no OCR artifacts."),
                "criteria": {"true": "a proper merchant name as-is",
                             "false": "contains OCR artifacts or is not a merchant name"},
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
                payload = json.loads(resp.read())
            answer = payload["answers"]["category"]
            sane = payload["answers"].get("vendor_sane", {}).get("noul")
            return (answer.get("choice") if answer.get("choice") in CATEGORIES else None,
                    answer.get("confidence"), payload.get("model"),
                    float(sane) if isinstance(sane, (int, float)) else None)
        except (urllib.error.URLError, urllib.error.HTTPError, KeyError,
                json.JSONDecodeError, TypeError, ValueError) as exc:
            last_exc = exc
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"jev api failed after 3 attempts: {last_exc}")


def flag_collisions(screens: list[dict], sheet_rows: list[list[str]]) -> dict:
    """One collision pass over the already-fetched tab: (date, amount) already
    in the sheet, re-showing across this run's screenshots, null dates.
    vendor-suspect flags are attached earlier, at extraction (Jev noul), and
    only counted here."""
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
        "vendor-suspect": flags.count("vendor-suspect"),
    }


def write_staging(scr: dict, tab: str, tab_rows: int) -> Path:
    payload = {
        "file": scr["file"],
        "capture_time": scr["capture_time"],
        "engine": scr["engine"],
        "target_tab": tab,
        "tab_rows_at_plan": tab_rows,
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
            f"{display_date(r['date']):<10} | {vendor:<38} | "
            f"{r['amount']:>9.2f} | {cat:<32} | {flags}")
    return "\n".join(lines)


def plan_extract(args: argparse.Namespace,
                 stream=sys.stdout) -> dict | None:
    """Plan-phase prologue shared by --plan and the default interactive run
    (issues #15/#17): auth → tab validation → scan data/new/ → extract → ONE
    values.get collision pass. Writes nothing (no staging, no sheet writes,
    no moves). Returns the working set, or None when data/new/ holds no
    screenshots at all (message already printed)."""
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
        return None

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
    return {"sheet_id": sheet_id, "tab": tab, "screens": screens,
            "errors": errors, "sheet_rows": sheet_rows, "counts": counts}


def plan(args: argparse.Namespace) -> int:
    # --json: stdout carries only the machine document; all human progress
    # (auth, per-screenshot lines, review table) moves to stderr.
    stream = sys.stderr if args.json else sys.stdout
    res = plan_extract(args, stream)
    if res is None:
        return 0
    screens, errors, counts = res["screens"], res["errors"], res["counts"]

    staged = [write_staging(scr, res["tab"], len(res["sheet_rows"]))
              for scr in screens]
    ERRORS_PATH.write_text(json.dumps(errors, indent=2) + "\n", encoding="utf-8")

    n_records = sum(len(s["records"]) for s in screens)
    for scr, path in zip(screens, staged):
        print(f"\n== {scr['file']}  (capture {scr['capture_time']}) ==", file=stream)
        print(review_table(scr), file=stream)
        flagged = sum(1 for r in scr["records"] if r["flags"])
        note = " — NEEDS REVIEW" if flagged else ""
        print(f"staging: {path}  ({len(scr['records'])} record(s), "
              f"{flagged} flagged{note})", file=stream)

    print(f"\nplan vs {res['tab']!r}: {len(screens)} screenshot(s), {n_records} record(s); "
          f"flags: {counts['already-in-sheet']} already-in-sheet, "
          f"{counts['re-shown']} re-shown, {counts['no-date']} no-date, "
          f"{counts['vendor-suspect']} vendor-suspect; "
          f"{len(errors)} error(s)", file=stream)
    print("sheet untouched (values.get only); screenshots untouched in data/new/",
          file=stream)

    if args.json:
        doc = {
            "mode": "plan",
            "sheet": res["sheet_id"],
            "tab": res["tab"],
            "planned_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "screenshots": [
                json.loads(p.read_text(encoding="utf-8")) for p in staged],
            "errors": errors,
            "summary": {"n_screenshots": len(screens), "n_records": n_records,
                        **counts},
        }
        print(json.dumps(doc, indent=2))

    return 1 if errors else 0


# ---------------------------------------------------------------------------
# Commit phase (issue #16)
# ---------------------------------------------------------------------------

def ensure_source_header(sheet_id: str, tab: str) -> None:
    """F1 must read `source`; written once on a tab's first import."""
    row = get_values(sheet_id, f"'{tab}'!F1:F1")
    got = str(row[0][0]).strip() if row and row[0] else ""
    if got == "source":
        print(f"  {tab!r}!F1 = 'source' (already present)")
        return
    if got:
        raise TracerError(
            f"tab {tab!r}: F1 is {got!r}, expected 'source' or empty — "
            "fix the tab layout before importing")
    gws_values_call("update",
                    {"spreadsheetId": sheet_id, "range": f"'{tab}'!F1",
                     "valueInputOption": "USER_ENTERED"},
                    {"values": [["source"]]})
    print(f"  added 'source' header at {tab!r}!F1")


def build_rows(doc: dict, staging_name: str) -> list[list[str]]:
    """Staging records → sheet rows: date | vendor | category | amount | ''
    | source. Total (E) stays blank for manual drag-fill. Refuses records
    review left unresolved (null/unparseable date, unusable amount)."""
    rows = []
    for i, r in enumerate(doc.get("records", [])):
        where = f"{staging_name} record {i} ({r.get('vendor') or 'unknown vendor'})"
        date = parse_sheet_date(r.get("date"))
        if not date:
            raise TracerError(
                f"{where} still has no usable date — review must edit (fill "
                "the date) or reject (delete the record) it in the staging "
                "JSON first; nothing was appended")
        amount = r.get("amount")
        if isinstance(amount, bool) or not isinstance(amount, (int, float)):
            raise TracerError(
                f"{where} has no usable amount ({amount!r}) — edit the "
                "staging JSON first; nothing was appended")
        rows.append([
            date,
            str(r.get("vendor") or ""),
            str(r.get("category") or ""),
            f"{float(amount):.2f}",
            "",  # Total: manual drag-fill
            str(doc["file"]),
        ])
    return rows


def verify_appended(sheet_id: str, tab: str, updated_range: str,
                    rows: list[list[str]]) -> None:
    """Read the appended range back and verify it (tracer pattern) BEFORE any
    file moves. Dates/amounts compared normalized (USER_ENTERED reformats)."""
    m = re.search(r"!([A-Z]+)(\d+):([A-Z]+)(\d+)$", updated_range)
    if not m:
        raise TracerError(
            f"could not parse append range {updated_range!r} — check {tab!r} "
            "manually; files NOT moved, staging retained")
    start_row = int(m.group(2))
    readback = get_values(sheet_id,
                          f"'{tab}'!A{start_row}:F{start_row + len(rows) - 1}")
    problems = []
    if len(readback) != len(rows):
        problems.append(f"expected {len(rows)} row(s) at {updated_range}, "
                        f"read back {len(readback)}")
    for i, (want, got) in enumerate(zip(rows, readback)):
        got = [str(c).strip() for c in got] + [""] * (6 - len(got))
        date, vendor, category, amount, _total, source = want
        if parse_sheet_date(got[0]) != date:
            problems.append(f"row {i}: date {got[0]!r} != {date!r}")
        if got[1] != vendor:
            problems.append(f"row {i}: vendor {got[1]!r} != {vendor!r}")
        if got[2] != category:
            problems.append(f"row {i}: category {got[2]!r} != {category!r}")
        if parse_sheet_amount(got[3]) != parse_sheet_amount(amount):
            problems.append(f"row {i}: amount {got[3]!r} != {amount!r}")
        if got[4]:
            problems.append(f"row {i}: Total should be blank, got {got[4]!r}")
        if got[5] != source:
            problems.append(f"row {i}: source {got[5]!r} != {source!r}")
    if problems:
        raise TracerError(
            "appended rows failed read-back verification — the rows ARE in "
            f"{tab!r}; inspect the tab manually and fix before re-running "
            "anything. Files NOT moved, staging retained:\n  "
            + "\n  ".join(problems))
    print("read-back verified: dates, vendors, categories, amounts, blank "
          "Total, source per row")


def move_to_processed(docs: list[tuple[Path, dict]]) -> None:
    """Post-verified-append only: image + sidecar out of data/new/, staging
    JSON rides along (from data/out/import) so a committed screenshot can
    never be committed twice."""
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    for p, doc in docs:
        stem = Path(doc["file"]).stem
        img = DATA_NEW / doc["file"]
        img.rename(DATA_PROCESSED / doc["file"])
        sidecar = DATA_NEW / f"{stem}.meta.json"
        if sidecar.is_file():
            sidecar.rename(DATA_PROCESSED / sidecar.name)
            print(f"  moved {doc['file']} + {sidecar.name} → {DATA_PROCESSED.relative_to(ROOT)}")
        else:
            print(f"  moved {doc['file']} → {DATA_PROCESSED.relative_to(ROOT)} "
                  "(no sidecar found)")
        if p.resolve().parent == DATA_IMPORT.resolve():
            p.rename(DATA_PROCESSED / p.name)
            print(f"  moved staging {p.name} → {DATA_PROCESSED.relative_to(ROOT)}")
        else:
            print(f"  staging {p} left in place (outside {DATA_IMPORT.relative_to(ROOT)})")


def load_staging(raw_paths: list[str]) -> list[tuple[Path, dict]]:
    docs = []
    for raw in raw_paths:
        p = Path(raw)
        if not p.is_file():
            raise TracerError(f"staging file not found: {p}")
        try:
            doc = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise TracerError(f"{p}: not valid JSON ({exc})")
        if not isinstance(doc, dict) or not doc.get("file") \
                or not isinstance(doc.get("records"), list):
            raise TracerError(
                f"{p}: not a plan-phase staging JSON (need 'file' + 'records')")
        docs.append((p, doc))
    return docs


def commit(args: argparse.Namespace) -> int:
    preflight_auth()
    print("auth: gws token valid")

    sheet_id = resolve_sheet_id(args)
    docs = load_staging(args.staging)

    tab = args.tab or datetime.now().strftime("%b %y")
    for p, doc in docs:
        if doc.get("target_tab") != tab:
            raise TracerError(
                f"{p.name} was planned for tab {doc.get('target_tab')!r}, "
                f"not {tab!r} — pass the tab it was planned for, or re-plan")

    require_tab(sheet_id, tab)
    print(f"workbook {sheet_id}: tab {tab!r} found")
    validate_headers(sheet_id, tab)
    print(f"headers of {tab!r}: {' | '.join(EXPECTED_HEADERS)} ✓")

    # Freshness: the tab must look exactly as it did when the plan was made.
    now_rows = len(get_values(sheet_id, f"'{tab}'!A1:E"))
    for p, doc in docs:
        at_plan = doc.get("tab_rows_at_plan")
        if at_plan is None:
            raise TracerError(
                f"{p.name} has no tab_rows_at_plan — staging from an older "
                "plan format; re-run --plan and review before committing")
        if at_plan != now_rows:
            raise TracerError(
                f"stale staging: {p.name} was planned against {at_plan} "
                f"row(s) in {tab!r}, but the tab now has {now_rows} row(s) — "
                "the sheet changed since the plan. Re-run --plan and review "
                "again; if a previous commit crashed mid-append, check the "
                "tab for already-appended rows first. Nothing was appended.")

    # All refusal checks happen BEFORE any write; one bad file aborts all.
    rows = []
    for p, doc in docs:
        img = DATA_NEW / doc["file"]
        if (DATA_PROCESSED / doc["file"]).exists():
            raise TracerError(
                f"{doc['file']} is already in {DATA_PROCESSED.relative_to(ROOT)} "
                "— its rows were likely committed before; refusing to append "
                "twice")
        if not img.is_file():
            raise TracerError(
                f"screenshot {img} not found in data/new/ — commit only moves "
                "what plan staged; restore it or re-plan")
        rows += build_rows(doc, p.name)

    flagged = sum(1 for _, doc in docs for r in doc["records"] if r.get("flags"))
    if flagged:
        names = ", ".join(p.name for p, doc in docs
                          if any(r.get("flags") for r in doc["records"]))
        print(f"note: {flagged} flagged record(s) still in staging ({names}) — "
              "committing as instructed")

    if not rows:
        print("staging has no records — nothing to commit")
        return 0

    ensure_source_header(sheet_id, tab)
    resp = append_rows(sheet_id, tab, rows)
    updated = resp.get("updates", {})
    rng = updated.get("updatedRange", "")
    print(f"appended {len(rows)} row(s) in one values.append → {rng or '?'}")

    verify_appended(sheet_id, tab, rng, rows)
    move_to_processed(docs)
    print(f"commit complete: {tab!r} now holds every committed screenshot's "
          "rows (processed/ ⟺ in sheet)")
    return 0


# ---------------------------------------------------------------------------
# Interactive review (issue #17): the default no-flag human run wraps
# the plan phase (#15) and feeds only the accepted subset into the exact
# commit path (#16).
# ---------------------------------------------------------------------------

def read_key(prompt: str, choices: str, stream=sys.stdout) -> str:
    """One keystroke on a TTY (cbreak — no Enter needed), one typed line
    otherwise (piped sessions and tests work identically). Re-prompts until
    the key is one of `choices`; raises EOFError on closed stdin; lets
    KeyboardInterrupt (Ctrl-C) propagate — callers treat both as abort."""
    while True:
        stream.write(prompt)
        stream.flush()
        ch = None
        if sys.stdin.isatty():
            try:
                import termios
                import tty
            except ImportError:
                pass  # no termios (e.g. Windows): fall through to line input
            else:
                fd = sys.stdin.fileno()
                saved = termios.tcgetattr(fd)
                try:
                    tty.setcbreak(fd)
                    try:
                        ch = sys.stdin.read(1)
                    except UnicodeDecodeError:
                        ch = "?"
                finally:
                    termios.tcsetattr(fd, termios.TCSADRAIN, saved)
                print(ch if ch.isprintable() else "", file=stream)
        if ch is None:
            line = sys.stdin.readline()
            if not line:
                raise EOFError
            ch = line
        if not ch:
            raise EOFError
        key = ch.strip().lower()[:1]
        if key in choices:
            return key
        print(f"  ? press one of: {'/'.join(choices)}", file=stream)


def prompt_field(label: str, stream=sys.stdout) -> str:
    """One raw text input for the edit menu; empty line = keep current,
    '-' = clear (the caller decides). Raises EOFError on closed stdin."""
    stream.write(f"{label}: ")
    stream.flush()
    line = sys.stdin.readline()
    if not line:
        raise EOFError
    return line.strip()


def group_line(num: int, r: dict, fname: str | None = None) -> str:
    """One numbered line for a day-group member (date lives in the group
    header; source file shown only when several screenshots are in play)."""
    conf = r["category_confidence"]
    conf_s = f"{conf:.2f}" if isinstance(conf, (int, float)) else "?"
    src = f"  [{fname}]" if fname else ""
    return (f"  [{num}] {r['amount']:>9.2f}  "
            f"{(r['vendor'] or '—')[:34]:<34} "
            f"{(r['category'] or '—'):<24} ({conf_s})  "
            f"flags={','.join(r['flags']) or '-'}{src}")


def edit_record(r: dict, stream=sys.stdout) -> None:
    """[v]endor / [d]ate / [a]mount / [c]ategory / [k]eep loop; edits apply
    to the record in place (staging + commit pick them up). Empty input
    keeps the current value; '-' clears the field."""
    from run_jev_arm import CATEGORIES

    while True:
        print(f"  editing: vendor={r['vendor'] or '—'}  "
              f"date={display_date(r['date'])}  amount={r['amount']}  "
              f"category={r['category'] or '—'}", file=stream)
        key = read_key("  edit: [v]endor [d]ate [a]mount [c]ategory "
                       "[k]eep: ", "vdack", stream)
        if key == "k":
            return
        if key == "v":
            raw = prompt_field("    vendor", stream)
            if raw == "-":
                r["vendor"] = ""
            elif raw:
                r["vendor"] = raw
            if raw:  # human replaced the string: the machine's flag is moot
                r["flags"] = [f for f in r["flags"] if f != "vendor-suspect"]
        elif key == "d":
            while True:
                raw = prompt_field("    date (dd/mm/yyyy)", stream)
                if raw == "-":
                    r["date"] = None
                    break
                if not raw:
                    break
                date = parse_sheet_date(raw)
                if date:
                    r["date"] = date
                    break
                print("    unparseable date — try 19/9/2026 or 2026-09-19 "
                      "('-' clears)", file=stream)
        elif key == "a":
            while True:
                raw = prompt_field("    amount", stream)
                if not raw:
                    break
                amount = parse_sheet_amount(raw)
                if amount is not None:
                    r["amount"] = amount
                    break
                print("    unparseable amount — try -45.67 or $1,234.50",
                      file=stream)
        elif key == "c":
            print("    enum: " + " | ".join(
                f"{i + 1}={c}" for i, c in enumerate(CATEGORIES)), file=stream)
            while True:
                raw = prompt_field(
                    "    category (number or exact text, '-' clears)", stream)
                if raw == "-":
                    r["category"] = None
                    r["category_confidence"] = None
                    break
                if not raw:
                    break
                if raw.isdigit() and 1 <= int(raw) <= len(CATEGORIES):
                    r["category"] = CATEGORIES[int(raw) - 1]
                    r["category_confidence"] = None
                    break
                match = next((c for c in CATEGORIES
                              if c.lower() == raw.lower()), None)
                if match:
                    r["category"] = match
                    r["category_confidence"] = None
                    break
                print("    not in the category enum — pick a number or the "
                      "exact text", file=stream)


def reflag_accepted(records: list[dict], sheet_rows: list[list[str]]) -> None:
    """Rebuild flags over the accepted subset with the sheet rows already
    fetched at plan time (no extra API call): edits can move a record onto
    or off of a collision, rejects can dissolve a re-shown pair."""
    existing = set()
    for row in sheet_rows[1:]:
        d = parse_sheet_date(row[0] if row else None)
        a = parse_sheet_amount(row[3] if len(row) > 3 else None)
        if d and a is not None:
            existing.add((d, a))

    pairs: dict[tuple[str, float], int] = {}
    for r in records:
        d = parse_sheet_date(r.get("date"))
        a = parse_sheet_amount(r.get("amount"))
        if d and a is not None:
            pairs[(d, a)] = pairs.get((d, a), 0) + 1

    for r in records:
        d = parse_sheet_date(r.get("date"))
        a = parse_sheet_amount(r.get("amount"))
        if not d:
            r["flags"] = ["no-date"]
        else:
            flags = []
            if (d, a) in existing:
                flags.append("already-in-sheet")
            if pairs.get((d, a), 0) > 1:
                flags.append("re-shown")
            r["flags"] = flags


def day_groups(flat: list[tuple[int, int, dict]],
               verdict: dict[tuple[int, int], str]
               ) -> list[tuple[str, list[tuple[int, int, dict]]]]:
    """Undecided records grouped by date, newest day first, unknown dates
    last. flat is (screen_idx, record_idx, record); verdict keys are decided
    and excluded. Groups span screenshots — the same day from two captures
    reviews together."""
    groups: dict[str, list[tuple[int, int, dict]]] = {}
    for si, ri, r in flat:
        if (si, ri) in verdict:
            continue
        key = parse_sheet_date(r.get("date")) or ""
        groups.setdefault(key, []).append((si, ri, r))
    order = sorted((k for k in groups if k), reverse=True)
    if "" in groups:
        order.append("")
    return [(k, groups[k]) for k in order]


def resolve_day(iso: str, items: list[tuple[int, int, dict]], verdict: dict,
                fname_of: dict[int, str], multi: bool,
                stream=sys.stdout) -> str | None:
    """Present one day's records together and resolve every one to accept or
    reject: [a] keep all, [n] keep none, numbers like 1,3,4,5 keep exactly
    those and drop the rest of the day, [e]<num> edits one in place, [q]
    aborts the whole review (None). A selection naming a record without a
    usable date decides nothing — fix the date with e<num> or reselect.
    Records without a usable date can never be kept as-is (commit refuses
    null dates)."""
    head = display_date(iso) if iso else "unknown date — edit or reject"
    wd = ""
    if iso:
        wd = datetime.strptime(iso, "%Y-%m-%d").strftime(" (%a)")
    while True:
        remaining = [(n, si, ri, r)
                     for n, (si, ri, r) in enumerate(items, 1)
                     if (si, ri) not in verdict]
        if not remaining:
            return "ok"
        print(f"\n── {head}{wd} — {len(remaining)} record(s) ──", file=stream)
        for n, si, ri, r in remaining:
            print(group_line(n, r, fname_of[si] if multi else None), file=stream)
        raw = prompt_field("keep which? [a]ll [n]one [e]<num> edit, or "
                           "numbers like 1,3,4,5 ([q]uit)", stream).strip().lower()
        if not raw:
            print("  ? a, n, numbers like 1,3,4,5, e<num>, or q", file=stream)
            continue
        if raw == "q":
            return None
        if raw == "a":
            blocked = 0
            for n, si, ri, r in remaining:
                if not parse_sheet_date(r.get("date")):
                    blocked += 1
                    continue
                verdict[(si, ri)] = "y"
            if blocked:
                print(f"  kept {len(remaining) - blocked}; {blocked} record(s) "
                      "have no usable date — edit (e<num>) or n", file=stream)
            continue
        if raw == "n":
            for n, si, ri, r in remaining:
                verdict[(si, ri)] = "n"
            continue
        if raw.startswith("e"):
            num = raw[1:].strip().strip(",")
            pick = next((x for x in remaining if str(x[0]) == num), None) \
                if num.isdigit() else None
            if pick is None:
                print(f"  ? e<num> with a number shown above (got {raw!r})",
                      file=stream)
                continue
            edit_record(pick[3], stream)  # re-list the day after editing
            continue
        nums = re.split(r"[,\s]+", raw)
        if all(p.isdigit() for p in nums):
            rows = {x[0]: x for x in remaining}
            unknown = [p for p in nums if int(p) not in rows]
            if unknown:
                print(f"  ? no such row(s): {', '.join(unknown)} — nothing "
                      "decided", file=stream)
                continue
            nodate = [rows[int(p)] for p in nums
                      if not parse_sheet_date(rows[int(p)][3].get("date"))]
            if nodate:
                print("  ? row(s) " + ", ".join(f"[{x[0]}]" for x in nodate)
                      + " have no usable date — edit (e<num>) or reselect; "
                        "nothing decided", file=stream)
                continue
            keep = {int(p) for p in nums}
            for n, si, ri, r in remaining:
                verdict[(si, ri)] = "y" if n in keep else "n"
            print(f"  kept {', '.join(str(n) for n in sorted(keep))} · "
                  f"dropped {len(remaining) - len(keep)}", file=stream)
            continue
        print("  ? a, n, numbers like 1,3,4,5, e<num>, or q", file=stream)


def interactive_review(screens: list[dict],
                       stream=sys.stdout) -> list[tuple[dict, list[dict]]] | None:
    """Review day by day, newest first: each day's records are listed
    together and resolved with a/n/numbers/e<num> (q aborts). Edits mutate
    records in place and the grouping re-sorts, so an edited date moves the
    record to its own day. Returns [(screen, [accepted records])] in
    original order, or None on abort."""
    for scr in screens:
        print(f"\n== {scr['file']}  (capture {scr['capture_time']}) ==",
              file=stream)
        print(review_table(scr), file=stream)

    flat = [(si, ri, r) for si, scr in enumerate(screens)
            for ri, r in enumerate(scr["records"])]
    fname_of = {si: Path(scr["file"]).name for si, scr in enumerate(screens)}
    multi = len(screens) > 1
    print(f"\n{len(flat)} record(s) from {len(screens)} screenshot(s), "
          "newest day first. Per day: [a] keep all, [n] none, numbers like "
          "1,3,4,5 keep exactly those (rest dropped), [e]<num> edit, "
          "[q] abort", file=stream)

    verdict: dict[tuple[int, int], str] = {}
    while True:
        groups = day_groups(flat, verdict)
        if not groups:
            break
        iso, items = groups[0]
        if resolve_day(iso, items, verdict, fname_of, multi, stream) is None:
            return None

    return [(scr, [r for ri, r in enumerate(scr["records"])
                   if verdict.get((si, ri)) == "y"])
            for si, scr in enumerate(screens)]


def gate_new_folder(stream=None) -> bool:
    """First step of the direct human run: open data/new/ in the file
    manager so the reviewer can drop this month's screenshots in, then
    wait for Enter to scan. Re-prompts while the folder holds no
    screenshots; q (Ctrl-C/EOF included) aborts → False."""
    stream = stream or sys.stdout
    DATA_NEW.mkdir(parents=True, exist_ok=True)

    def n_images() -> int:
        return sum(1 for p in DATA_NEW.iterdir()
                   if p.suffix.lower() in IMAGE_EXTS | HEIC_EXTS)

    opener = "open" if sys.platform == "darwin" else "xdg-open"
    try:
        subprocess.run([opener, str(DATA_NEW)],
                       capture_output=True, timeout=10)
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        print(f"(could not auto-open a file manager) — put screenshots in "
              f"{DATA_NEW}", file=stream)
    else:
        print(f"opened {DATA_NEW.relative_to(ROOT)} in the file manager",
              file=stream)
    if n := n_images():
        print(f"{n} screenshot(s) already in data/new/", file=stream)
    while True:
        print("drop screenshots there, then press Enter to scan "
              "(q to abort): ", end="", file=stream)
        stream.flush()
        line = sys.stdin.readline()
        if not line:
            return False
        if line.strip().lower()[:1] == "q":
            return False
        if n_images():
            return True
        print(f"  still no screenshots in {DATA_NEW.relative_to(ROOT)} — "
              "drop them in, then Enter again", file=stream)


def interactive(args: argparse.Namespace) -> int:
    """Default run (no mode flag): open data/new/ and wait for screenshots,
    then plan (#15), per-record review (#17), then commit the accepted
    subset through the exact --commit path (#16)."""
    stream = sys.stdout
    if not gate_new_folder(stream):
        print("\nabort: nothing scanned, nothing appended, nothing moved",
              file=stream)
        return 1
    res = plan_extract(args, stream)
    if res is None:
        return 0
    screens, errors, sheet_rows = (res["screens"], res["errors"],
                                   res["sheet_rows"])
    ERRORS_PATH.write_text(json.dumps(errors, indent=2) + "\n", encoding="utf-8")

    n_records = sum(len(s["records"]) for s in screens)
    if not n_records:
        print(f"no records extracted ({len(errors)} error(s)) — "
              "nothing to review", file=stream)
        return 1 if errors else 0

    try:
        decisions = interactive_review(screens, stream)
    except (KeyboardInterrupt, EOFError):
        print(file=stream)
        decisions = None
    if decisions is None:
        print("\nabort: nothing appended, nothing moved — screenshots "
              "remain in data/new/", file=stream)
        return 1

    accepted = [(scr, recs) for scr, recs in decisions if recs]
    all_accepted = [r for _, recs in accepted for r in recs]
    reflag_accepted(all_accepted, sheet_rows)
    n_rejected = n_records - len(all_accepted)
    print(f"\nreview done: {len(all_accepted)} accepted, {n_rejected} rejected",
          file=stream)

    if not accepted:
        print("nothing accepted — no staging written, sheet untouched, "
              "screenshots stay in data/new/", file=stream)
        return 0

    staged = [write_staging(dict(scr, records=recs), res["tab"],
                            len(sheet_rows)) for scr, recs in accepted]
    for (scr, recs), path in zip(accepted, staged):
        print(f"staging: {path}  ({len(recs)} accepted record(s))", file=stream)
    if errors:
        print(f"note: {len(errors)} screenshot(s) failed extraction "
              f"(see {ERRORS_PATH}) — nothing was reviewed for them",
              file=stream)

    commit_args = argparse.Namespace(sheet=args.sheet, tab=res["tab"],
                                     staging=[str(p) for p in staged])
    print(f"\ncommitting {len(all_accepted)} accepted record(s) to "
          f"{res['tab']!r} via the --commit path...", file=stream)
    return commit(commit_args)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import pipeline (issue #13: sheet tracer, #15: plan "
                    "phase, #16: commit phase; no mode flag: interactive "
                    "review, #17)")
    parser.add_argument("--tracer-sheet", action="store_true",
                        help="run the sheet-layer round-trip tracer")
    parser.add_argument("--plan", action="store_true",
                        help="plan phase: scan data/new/, extract + flag, write "
                             "staging JSONs — reads the sheet, never writes it")
    parser.add_argument("--commit", action="store_true",
                        help="commit phase: append reviewed staging rows to "
                             "the tab in ONE values.append, then move the "
                             "screenshots to data/processed/")
    parser.add_argument("--staging", nargs="+", default=None, metavar="JSON",
                        help="with --commit: one or more plan-phase staging "
                             "JSONs (data/out/import/<stem>.json)")
    parser.add_argument("--json", action="store_true",
                        help="with --plan: machine-readable JSON on stdout "
                             "(review table moves to stderr)")
    parser.add_argument("--sheet", default=None,
                        help="spreadsheet ID override (default: SHEET_ID in .env)")
    parser.add_argument("--tab", default=None,
                        help="target tab (default: current month, e.g. 'Sep 26')")
    args = parser.parse_args()

    modes = [m for m in (args.tracer_sheet, args.plan, args.commit) if m]
    if len(modes) > 1:
        raise TracerError("pick one mode: --tracer-sheet, --plan or --commit")
    if args.json and not args.plan:
        raise TracerError("--json only applies to --plan")
    if args.staging and not args.commit:
        raise TracerError("--staging only applies to --commit")
    if args.commit and not args.staging:
        raise TracerError("--commit needs --staging <plan JSON(s)> — "
                          "never commit without the reviewed plan")

    if args.tracer_sheet:
        tracer_sheet(args)
    elif args.plan:
        raise SystemExit(plan(args))
    elif args.commit:
        raise SystemExit(commit(args))
    else:
        # Default: the direct human run (issue #17) — plan, interactive
        # review, commit the accepted subset.
        raise SystemExit(interactive(args))


if __name__ == "__main__":
    main()
