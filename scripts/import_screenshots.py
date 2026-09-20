#!/usr/bin/env python3
"""Sheet tracer (issue #13): prove the sheet layer of the import pipeline with
zero writes to real ledger data.

`--tracer-sheet` runs, in order:

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

The sheet-I/O helpers here (get/append/validation) are the reusable import
block for the later plan/commit slices. No new Python dependencies: everything
shells out to `gws`.

Usage:
    uv run python scripts/import_screenshots.py --tracer-sheet
    uv run python scripts/import_screenshots.py --tracer-sheet --sheet <ID> --tab "Sep 26"
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

EXPECTED_HEADERS = ["date", "vendor", "category", "amount", "Total"]

# USER_ENTERED-safe probe row: no cell parses to a date, numbers read back
# verbatim under General format on a fresh tab.
TRACER_ROW_PREFIX = "sheet-tracer"


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import pipeline tracers (issue #13: sheet layer)")
    parser.add_argument("--tracer-sheet", action="store_true",
                        help="run the sheet-layer round-trip tracer")
    parser.add_argument("--sheet", default=None,
                        help="spreadsheet ID override (default: SHEET_ID in .env)")
    parser.add_argument("--tab", default=None,
                        help="target tab to validate (default: current month, e.g. 'Sep 26')")
    args = parser.parse_args()

    if not args.tracer_sheet:
        parser.print_help()
        raise TracerError("pick a tracer mode, e.g. --tracer-sheet")

    tracer_sheet(args)


if __name__ == "__main__":
    main()
