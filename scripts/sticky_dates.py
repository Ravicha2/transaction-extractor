"""Shared sticky-date-header resolution, used by every extraction arm.

Resolves the bank list's sticky headers ("Today", "Yesterday", "Thu 10 Sep")
to an ISO date against the screenshot's capture timestamp (from the sidecar
written by make_sidecars.py).

Dumb on purpose (this is the control-arm spirit): the header's weekday name is
ignored for resolution; the year comes from the capture date and rolls back one
year if the resolved date would land in the future (Dec screenshots in January).
"""

import re
from datetime import date, datetime, timedelta

MONTHS = {m: i for i, m in enumerate(
    "jan feb mar apr may jun jul aug sep oct nov dec".split(), 1)}

DOW_DAY_MONTH_RE = re.compile(
    r"\b(?:mon|tue|wed|thu|fri|sat|sun)\s*(\d{1,2})\s+"
    r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b",
    re.IGNORECASE,
)
TODAY_RE = re.compile(r"\btoday\b", re.IGNORECASE)
YESTERDAY_RE = re.compile(r"\byesterday\b", re.IGNORECASE)


def parse_capture_time(iso: str | None) -> date:
    if not iso:
        raise ValueError("missing capture timestamp (sidecar)")
    return datetime.fromisoformat(iso).date()


def resolve_header(text: str, capture: date) -> date | None:
    """Return the date a header line refers to, or None if it is not a header."""
    if TODAY_RE.search(text):
        return capture
    if YESTERDAY_RE.search(text):
        return capture - timedelta(days=1)
    m = DOW_DAY_MONTH_RE.search(text)
    if not m:
        return None
    day, month = int(m.group(1)), MONTHS[m.group(2).lower()]
    resolved = date(capture.year, month, day)
    if resolved > capture:
        resolved = date(capture.year - 1, month, day)
    return resolved
