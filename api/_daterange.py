"""
api/_daterange.py
Parse a natural-language question into (date_from, date_to) ISO strings.
date_to defaults to the latest available data date when not explicitly stated.
Returns (None, None) when no range pattern is detected.
"""

from __future__ import annotations

import re
from datetime import date, timedelta

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MONTHS: dict[str, str] = {
    "january": "01", "february": "02", "march": "03", "april": "04",
    "may": "05", "june": "06", "july": "07", "august": "08",
    "september": "09", "october": "10", "november": "11", "december": "12",
    "jan": "01", "feb": "02", "mar": "03", "apr": "04",
    "jun": "06", "jul": "07", "aug": "08",
    "sep": "09", "oct": "10", "nov": "11", "dec": "12",
}

_ORDINAL = r"(?:st|nd|rd|th)?"
_SEP     = r"[\s,/-]+"
_MONTH   = r"(" + "|".join(sorted(_MONTHS, key=len, reverse=True)) + r")"
_DAY     = r"(\d{1,2})" + _ORDINAL
_YEAR    = r"(\d{4})"
_ISO     = r"(\d{4}-\d{2}-\d{2})"
_RANGE_CONN = r"(?:to|through|thru|until|and|-)"


# ---------------------------------------------------------------------------
# Low-level parsers
# ---------------------------------------------------------------------------

def _m(name: str) -> str | None:
    return _MONTHS.get(name.lower())


def _iso(y: str, m: str, d: str) -> str:
    return f"{y}-{m}-{int(d):02d}"


def _parse_explicit_range(q: str) -> tuple[str, str] | None:
    """ISO-to-ISO: 2026-01-01 to 2026-06-04"""
    m = re.search(
        rf"{_ISO}\s*{_RANGE_CONN}\s*{_ISO}",
        q, re.IGNORECASE
    )
    if m:
        return m.group(1), m.group(2)
    return None


def _parse_ordinal_month_year_range(q: str) -> tuple[str, str] | None:
    """
    1st of June to 4th of June 2026
    June 1st to June 4th 2026
    1 June to 4 June 2026
    """
    # "Nth of Month [YYYY] to Nth of Month YYYY"
    pat1 = re.compile(
        rf"{_DAY}\s+of\s+{_MONTH}(?:\s+{_YEAR})?\s*{_RANGE_CONN}\s*"
        rf"{_DAY}\s+of\s+{_MONTH}\s+{_YEAR}",
        re.IGNORECASE,
    )
    m = pat1.search(q)
    if m:
        d1, mo1, y1, d2, mo2, y2 = m.groups()
        y1 = y1 or y2
        mk1, mk2 = _m(mo1), _m(mo2)
        if mk1 and mk2:
            return _iso(y1, mk1, d1), _iso(y2, mk2, d2)

    # "Month Nth [YYYY] to Month Nth YYYY"
    pat2 = re.compile(
        rf"{_MONTH}\s+{_DAY}(?:\s*,?\s*{_YEAR})?\s*{_RANGE_CONN}\s*"
        rf"{_MONTH}\s+{_DAY}(?:\s*,?\s*{_YEAR})?(?:\s+{_YEAR})?",
        re.IGNORECASE,
    )
    m = pat2.search(q)
    if m:
        grps = m.groups()
        # groups: mo1, d1, y1_inline, mo2, d2, y2_inline, y_trailing
        mo1, d1, y1i, mo2, d2, y2i, yt = grps
        y1 = y1i or y2i or yt
        y2 = y2i or yt or y1
        mk1, mk2 = _m(mo1), _m(mo2)
        if mk1 and mk2 and y1 and y2:
            return _iso(y1, mk1, d1), _iso(y2, mk2, d2)

    return None


def _parse_single_month(q: str, data_date: date) -> tuple[str, str] | None:
    """
    "in June 2026" / "during June 2026" / "for June 2026"
    → first to last available day of that month
    """
    m = re.search(
        rf"(?:in|during|for|of)\s+{_MONTH}\s+{_YEAR}",
        q, re.IGNORECASE,
    )
    if m:
        mo, y = m.groups()
        mk = _m(mo)
        if mk:
            d_from = f"{y}-{mk}-01"
            # last day of month = first day of next month - 1
            y_int, m_int = int(y), int(mk)
            if m_int == 12:
                last = date(y_int, 12, 31)
            else:
                last = date(y_int, m_int + 1, 1) - timedelta(days=1)
            d_to = min(last, data_date).isoformat()
            return d_from, d_to
    return None


def _parse_quarter(q: str, data_date: date) -> tuple[str, str] | None:
    """
    Q1 2026 / first quarter 2026 / Q2 of 2026
    """
    qmap = {
        "1": ("01-01", "03-31"), "2": ("04-01", "06-30"),
        "3": ("07-01", "09-30"), "4": ("10-01", "12-31"),
        "first": ("01-01", "03-31"), "second": ("04-01", "06-30"),
        "third": ("07-01", "09-30"), "fourth": ("10-01", "12-31"),
    }
    m = re.search(
        r"(?:Q(\d)|(\bfirst\b|\bsecond\b|\bthird\b|\bfourth\b)\s+quarter)"
        r"(?:\s+of)?\s+(\d{4})",
        q, re.IGNORECASE,
    )
    if m:
        qnum, qword, y = m.groups()
        key = (qnum or qword).lower()
        if key in qmap:
            d_from = f"{y}-{qmap[key][0]}"
            last = date.fromisoformat(f"{y}-{qmap[key][1]}")
            d_to = min(last, data_date).isoformat()
            return d_from, d_to
    return None


def _parse_ytd(q: str, data_date: date) -> tuple[str, str] | None:
    """
    "year to date" / "ytd" / "so far this year" / "for 2026" / "in 2026"
    / "this year" / "since the start of 2026"
    """
    # Explicit year mention
    m = re.search(
        r"(?:for|in|during|of|throughout|all of)\s+(\d{4})\b"
        r"|(?:since\s+(?:the\s+)?(?:start|beginning)\s+of\s+(\d{4}))",
        q, re.IGNORECASE,
    )
    if m:
        y = m.group(1) or m.group(2)
        d_from = f"{y}-01-01"
        last = date(int(y), 12, 31)
        d_to = min(last, data_date).isoformat()
        return d_from, d_to

    # YTD / this year — use data_date's year
    if re.search(r"\bytd\b|year[- ]to[- ]date|so far this year|this year", q, re.IGNORECASE):
        y = str(data_date.year)
        return f"{y}-01-01", data_date.isoformat()

    return None


def _parse_last_n(q: str, data_date: date) -> tuple[str, str] | None:
    """
    last 30 days / past 3 months / previous 2 weeks
    """
    m = re.search(
        r"(?:last|past|previous|trailing)\s+(\d+)\s+(day|week|month|year)s?",
        q, re.IGNORECASE,
    )
    if m:
        n, unit = int(m.group(1)), m.group(2).lower()
        if unit == "day":
            delta = timedelta(days=n)
        elif unit == "week":
            delta = timedelta(weeks=n)
        elif unit == "month":
            delta = timedelta(days=n * 30)
        else:
            delta = timedelta(days=n * 365)
        return (data_date - delta).isoformat(), data_date.isoformat()
    return None


def _parse_since(q: str, data_date: date) -> tuple[str, str] | None:
    """
    since January 2026 / since 2026-01-01 / since Jan 1st 2026
    """
    # since ISO
    m = re.search(r"since\s+(\d{4}-\d{2}-\d{2})", q, re.IGNORECASE)
    if m:
        return m.group(1), data_date.isoformat()

    # since Month YYYY
    m = re.search(
        rf"since\s+{_MONTH}\s+{_YEAR}",
        q, re.IGNORECASE,
    )
    if m:
        mo, y = m.groups()
        mk = _m(mo)
        if mk:
            return f"{y}-{mk}-01", data_date.isoformat()

    # since Day Month YYYY / since Month Day YYYY
    m = re.search(
        rf"since\s+{_DAY}\s+{_MONTH}\s+{_YEAR}",
        q, re.IGNORECASE,
    )
    if m:
        d, mo, y = m.groups()
        mk = _m(mo)
        if mk:
            return _iso(y, mk, d), data_date.isoformat()

    m = re.search(
        rf"since\s+{_MONTH}\s+{_DAY}(?:,?\s*{_YEAR})?",
        q, re.IGNORECASE,
    )
    if m:
        mo, d, y = m.groups()
        mk = _m(mo)
        y = y or str(data_date.year)
        if mk:
            return _iso(y, mk, d), data_date.isoformat()

    return None


def _parse_year_to_month(q: str, data_date: date) -> tuple[str, str] | None:
    """
    from 2024 to June 2026
    2024 to June 2026
    from 2024 through May 2026
    """
    m = re.search(
        rf"(?:from\s+)?(\d{{4}})\s+(?:to|through|thru|until)\s+{_MONTH}\s+{_YEAR}",
        q, re.IGNORECASE,
    )
    if m:
        y1, mo2, y2 = m.groups()
        mk2 = _m(mo2)
        if mk2:
            int_m2 = int(mk2)
            if int_m2 == 12:
                last_day = date(int(y2), 12, 31)
            else:
                last_day = date(int(y2), int_m2 + 1, 1) - timedelta(days=1)
            return f"{y1}-01-01", min(last_day, data_date).isoformat()
    return None


def _parse_year_to_year(q: str, data_date: date) -> tuple[str, str] | None:
    """
    from 2024 to 2026
    2024 to 2026
    2024 through 2025
    """
    m = re.search(
        r"(?:from\s+)?(\d{4})\s+(?:to|through|thru|until)\s+(\d{4})\b",
        q, re.IGNORECASE,
    )
    if m:
        y1, y2 = m.groups()
        last_day = date(int(y2), 12, 31)
        return f"{y1}-01-01", min(last_day, data_date).isoformat()
    return None


def _parse_between_months(q: str, data_date: date) -> tuple[str, str] | None:
    """between April 2026 and June 2026"""
    m = re.search(
        rf"between\s+{_MONTH}\s+{_YEAR}\s+and\s+{_MONTH}\s+{_YEAR}",
        q, re.IGNORECASE,
    )
    if m:
        mo1, y1, mo2, y2 = m.groups()
        mk1, mk2 = _m(mo1), _m(mo2)
        if mk1 and mk2:
            last_mo2 = date(int(y2), int(mk2), 1)
            if int(mk2) == 12:
                last_day = date(int(y2), 12, 31)
            else:
                last_day = date(int(y2), int(mk2) + 1, 1) - timedelta(days=1)
            return f"{y1}-{mk1}-01", min(last_day, data_date).isoformat()
    return None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def extract_date_range(question: str, data_date: date) -> tuple[str | None, str | None]:
    """
    Try all patterns in priority order.
    Returns (date_from, date_to) ISO strings, or (None, None) if no range found.
    """
    q = question.strip()

    for fn in (
        _parse_explicit_range,
        _parse_ordinal_month_year_range,
        lambda q: _parse_year_to_month(q, data_date),
        lambda q: _parse_year_to_year(q, data_date),
        lambda q: _parse_between_months(q, data_date),
        lambda q: _parse_single_month(q, data_date),
        lambda q: _parse_quarter(q, data_date),
        lambda q: _parse_ytd(q, data_date),
        lambda q: _parse_last_n(q, data_date),
        lambda q: _parse_since(q, data_date),
    ):
        result = fn(q)
        if result:
            return result

    return None, None
