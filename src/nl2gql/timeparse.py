"""Resolve time phrases to inclusive date ranges against an injectable ``today``.

The conventions are the ones in ``data/CONVENTIONS.md``: weeks start on Monday, a bare month or
quarter means the most recent one that has started, periods that contain today end today, "last N
days" includes today. Every match reports its character span so later extractors can ignore the
digits inside it ("Q3" is not shift 3, "last 7 days" is not a limit of 7).
"""

from __future__ import annotations

import calendar
import contextlib
import datetime as dt
import re
from dataclasses import dataclass

MONTHS = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3, "april": 4, "apr": 4,
    "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7, "august": 8, "aug": 8, "september": 9,
    "sept": 9, "sep": 9, "october": 10, "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12,
}
NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "a": 1, "an": 1, "a couple of": 2, "couple of": 2, "a couple": 2, "a few": 3, "few": 3,
}
ORDINALS = {"first": 1, "1st": 1, "second": 2, "2nd": 2, "third": 3, "3rd": 3, "fourth": 4, "4th": 4}

_MONTH = r"(?P<{g}>" + "|".join(sorted(MONTHS, key=len, reverse=True)) + r")\.?"
_NUM = r"(?P<{g}>\d{1,3}|" + "|".join(sorted((k for k in NUMBER_WORDS if " " not in k and len(k) > 2),
                                               key=len, reverse=True)) + r")"
_DAY = r"(?P<{g}>[0-3]?\d)(?:st|nd|rd|th)?"
_YEAR = r"(?P<{g}>(?:19|20)\d\d)"

# One date expression. Named groups are suffixed so two dates can sit in one range pattern.
_DATE_FORMS = [
    r"{y}-(?P<{m}n>[01]?\d)-(?P<{d}n>[0-3]?\d)",                           # 2026-11-03
    r"(?P<{m}s>[01]?\d)/(?P<{d}s>[0-3]?\d)(?:/(?P<{y}s>(?:20)?\d\d))?",    # 11/3, 11/3/2026
    _MONTH.replace("{g}", "{m}a") + r"\s+" + _DAY.replace("{g}", "{d}a")
    + r"(?:,?\s+" + _YEAR.replace("{g}", "{y}a") + r")?",                    # nov 3rd, 2026
    _DAY.replace("{g}", "{d}b") + r"(?:\s+of)?\s+" + _MONTH.replace("{g}", "{m}b")
    + r"(?:,?\s+" + _YEAR.replace("{g}", "{y}b") + r")?",                    # 3rd of november 2026
]


def _date_regex(tag: str) -> str:
    forms = []
    for f in _DATE_FORMS:
        f = f.replace("{y}-", _YEAR.replace("{g}", f"y{tag}i") + "-")
        forms.append(f.replace("{m}", f"m{tag}").replace("{d}", f"d{tag}").replace("{y}", f"y{tag}"))
    return "(?:" + "|".join(forms) + ")"


DATE1, DATE2 = _date_regex("1"), _date_regex("2")


@dataclass(frozen=True)
class TimeMatch:
    start: dt.date
    end: dt.date
    span: tuple[int, int]
    phrase: str

    @property
    def iso(self) -> tuple[str, str]:
        return self.start.isoformat(), self.end.isoformat()


def _month_end(year: int, month: int) -> dt.date:
    return dt.date(year, month, calendar.monthrange(year, month)[1])


def _recent_month_year(month: int, today: dt.date) -> int:
    return today.year if month <= today.month else today.year - 1


def _number(text: str) -> int | None:
    text = text.strip().lower()
    if text.isdigit():
        return int(text)
    return NUMBER_WORDS.get(text)


def _date_from(m: re.Match, tag: str, today: dt.date) -> dt.date | None:
    g = m.groupdict()
    if g.get(f"y{tag}i"):
        y, mo, d = int(g[f"y{tag}i"]), int(g[f"m{tag}n"]), int(g[f"d{tag}n"])
        explicit_year = True
    elif g.get(f"m{tag}s"):
        mo, d = int(g[f"m{tag}s"]), int(g[f"d{tag}s"])
        ys = g.get(f"y{tag}s")
        explicit_year = bool(ys)
        y = (2000 + int(ys) if ys and len(ys) == 2 else int(ys)) if ys else today.year
    elif g.get(f"m{tag}a"):
        mo, d = MONTHS[g[f"m{tag}a"].lower().rstrip(".")], int(g[f"d{tag}a"])
        explicit_year = bool(g.get(f"y{tag}a"))
        y = int(g[f"y{tag}a"]) if explicit_year else today.year
    elif g.get(f"m{tag}b"):
        mo, d = MONTHS[g[f"m{tag}b"].lower().rstrip(".")], int(g[f"d{tag}b"])
        explicit_year = bool(g.get(f"y{tag}b"))
        y = int(g[f"y{tag}b"]) if explicit_year else today.year
    else:
        return None
    try:
        date = dt.date(y, mo, d)
    except ValueError:
        return None
    if not explicit_year and date > today:
        try:
            date = dt.date(y - 1, mo, d)
        except ValueError:
            return None
    return date


def _clamp(start: dt.date, end: dt.date, today: dt.date) -> tuple[dt.date, dt.date]:
    return start, (min(end, today) if start <= today else end)


def _quarter(q: int, year: int) -> tuple[dt.date, dt.date]:
    first = 3 * (q - 1) + 1
    return dt.date(year, first, 1), _month_end(year, first + 2)


_B = r"(?<![\w-])"   # left word boundary that also refuses hyphenated keys
_E = r"(?![\w-])"

_RANGE = re.compile(
    _B + r"(?:(?:from|between)\s+)?" + DATE1 + r"\s*(?:-|–|to|through|thru|until|till|and)\s*" + DATE2 + _E, re.I)
_SINCE_DATE = re.compile(_B + r"(?:since|starting|from)\s+(?:the\s+)?" + DATE1 + _E, re.I)
_ON_DATE = re.compile(_B + r"(?:(?:on|for|of|at)\s+(?:the\s+)?)?" + DATE1 + _E, re.I)
_MONTH_YEAR = re.compile(_B + r"(?:(?:the\s+)?month\s+of\s+)?" + _MONTH.replace("{g}", "mon")
                         + r",?\s+" + _YEAR.replace("{g}", "yr") + _E, re.I)
_SINCE_MONTH = re.compile(_B + r"since\s+(?:the\s+(?:start|beginning)\s+of\s+)?"
                          + _MONTH.replace("{g}", "mon") + _E, re.I)
_MONTH_ALONE = re.compile(_B + r"(?P<lead>(?:the\s+)?month\s+of|in|during|for|of|over|throughout|thru|through|last|this)?"
                          r"\s*" + _MONTH.replace("{g}", "mon") + _E, re.I)
_QUARTER = re.compile(
    _B + r"(?:q(?P<q1>[1-4])|(?P<q2>first|second|third|fourth|1st|2nd|3rd|4th)\s+quarter)"
    r"(?:\s*(?:of\s+)?(?:'(?P<yy>\d\d)|" + _YEAR.replace("{g}", "qy") + r"))?" + _E, re.I)
_REL_QUARTER = re.compile(_B + r"(?P<which>this|current|last|previous|prior|past)\s+(?:quarter|qtr)" + _E, re.I)
_REL_N = re.compile(_B + r"(?:the\s+)?(?:last|past|previous|prior|trailing)\s+" + _NUM.replace("{g}", "n")
                    + r"\s+(?P<unit>days?|d|weeks?|wks?|w|months?|mths?)" + _E, re.I)
_N_DAYS_AGO = re.compile(_B + _NUM.replace("{g}", "n") + r"\s+(?P<unit>days?|weeks?)\s+ago" + _E, re.I)
_REL_PERIOD = re.compile(
    _B + r"(?P<which>this|current|last|previous|prior|past)\s+(?P<unit>week|wk|month|mth|mo|year|yr)" + _E, re.I)
_YTD = re.compile(_B + r"(?:ytd|year\s+to\s+date|so\s+far\s+this\s+year|mtd|month\s+to\s+date|wtd|week\s+to\s+date)" + _E, re.I)
_BARE_YEAR = re.compile(_B + r"(?:(?:in|for|during|of|over|throughout)\s+)?(?:the\s+year\s+)?" + _YEAR.replace("{g}", "yr") + _E, re.I)
_TODAY = re.compile(_B + r"(?:today|tdy|so\s+far\s+today|this\s+morning|tonight)" + _E, re.I)
_YESTERDAY = re.compile(_B + r"(?:yesterday|yday|yest|ystrdy|last\s+night)" + _E, re.I)


def find(text: str, today: dt.date) -> TimeMatch | None:
    """The best time expression in ``text``, or None. Earlier patterns take precedence."""
    for match in _find_all(text, today):
        return match
    return None


def _find_all(text: str, today: dt.date):
    for m in _RANGE.finditer(text):
        a, b = _date_from(m, "1", today), _date_from(m, "2", today)
        if a and b:
            first_has_year = any(m.group(g) for g in ("y1i", "y1s", "y1a", "y1b"))
            second_has_year = any(m.group(g) for g in ("y2i", "y2s", "y2a", "y2b"))
            if second_has_year and not first_has_year:
                with contextlib.suppress(ValueError):   # Feb 29 in a non-leap year
                    a = a.replace(year=b.year)          # "January 5 and January 20, 2025"
            if a > b and not (first_has_year or second_has_year):
                a = a.replace(year=a.year - 1)
            yield TimeMatch(a, b, m.span(), m.group(0))
            return
    for m in _SINCE_DATE.finditer(text):
        a = _date_from(m, "1", today)
        if a:
            yield TimeMatch(a, today, m.span(), m.group(0))
            return
    for m in _MONTH_YEAR.finditer(text):
        month, year = MONTHS[m.group("mon").lower().rstrip(".")], int(m.group("yr"))
        yield TimeMatch(*_clamp(dt.date(year, month, 1), _month_end(year, month), today), m.span(), m.group(0))
        return
    for m in _QUARTER.finditer(text):
        q = int(m.group("q1")) if m.group("q1") else ORDINALS[m.group("q2").lower()]
        if m.group("qy"):
            year = int(m.group("qy"))
        elif m.group("yy"):
            year = 2000 + int(m.group("yy"))
        else:
            year = today.year if 3 * (q - 1) + 1 <= today.month else today.year - 1
        yield TimeMatch(*_clamp(*_quarter(q, year), today), m.span(), m.group(0))
        return
    for m in _REL_QUARTER.finditer(text):
        current = (today.month - 1) // 3 + 1
        if m.group("which").lower() in ("this", "current"):
            yield TimeMatch(*_clamp(*_quarter(current, today.year), today), m.span(), m.group(0))
        else:
            q, y = (current - 1, today.year) if current > 1 else (4, today.year - 1)
            yield TimeMatch(*_quarter(q, y), m.span(), m.group(0))
        return
    for m in _REL_N.finditer(text):
        n = _number(m.group("n"))
        unit = m.group("unit").lower()
        if not n:
            continue
        if unit.startswith("d"):
            start = today - dt.timedelta(days=n - 1)
        elif unit.startswith("w"):
            start = today - dt.timedelta(days=7 * n - 1)
        else:  # last N months: same day N months back, plus one day
            month0 = today.month - n
            year = today.year + (month0 - 1) // 12
            month = (month0 - 1) % 12 + 1
            day = min(today.day, calendar.monthrange(year, month)[1])
            start = dt.date(year, month, day) + dt.timedelta(days=1)
        yield TimeMatch(start, today, m.span(), m.group(0))
        return
    for m in _N_DAYS_AGO.finditer(text):
        n = _number(m.group("n"))
        if n:
            days = n * (7 if m.group("unit").lower().startswith("w") else 1)
            d = today - dt.timedelta(days=days)
            yield TimeMatch(d, d, m.span(), m.group(0))
            return
    for m in _REL_PERIOD.finditer(text):
        which, unit = m.group("which").lower(), m.group("unit").lower()
        this = which in ("this", "current")
        if unit in ("week", "wk"):
            monday = today - dt.timedelta(days=today.weekday())
            span = (monday, today) if this else (monday - dt.timedelta(days=7), monday - dt.timedelta(days=1))
        elif unit in ("month", "mth", "mo"):
            first = today.replace(day=1)
            if this:
                span = (first, today)
            else:
                prev_end = first - dt.timedelta(days=1)
                span = (prev_end.replace(day=1), prev_end)
        else:
            span = ((dt.date(today.year, 1, 1), today) if this
                    else (dt.date(today.year - 1, 1, 1), dt.date(today.year - 1, 12, 31)))
        yield TimeMatch(*span, m.span(), m.group(0))
        return
    for m in _YTD.finditer(text):
        word = m.group(0).lower()
        if word.startswith(("mtd", "month")):
            start = today.replace(day=1)
        elif word.startswith(("wtd", "week")):
            start = today - dt.timedelta(days=today.weekday())
        else:
            start = dt.date(today.year, 1, 1)
        yield TimeMatch(start, today, m.span(), m.group(0))
        return
    for m in _TODAY.finditer(text):
        yield TimeMatch(today, today, m.span(), m.group(0))
        return
    for m in _YESTERDAY.finditer(text):
        y = today - dt.timedelta(days=1)
        yield TimeMatch(y, y, m.span(), m.group(0))
        return
    for m in _ON_DATE.finditer(text):
        a = _date_from(m, "1", today)
        if a and _plausible_single_date(m):
            yield TimeMatch(a, a, m.span(), m.group(0))
            return
    for m in _SINCE_MONTH.finditer(text):
        month = MONTHS[m.group("mon").lower().rstrip(".")]
        yield TimeMatch(dt.date(_recent_month_year(month, today), month, 1), today, m.span(), m.group(0))
        return
    for m in _MONTH_ALONE.finditer(text):
        word = m.group("mon").lower().rstrip(".")
        lead = (m.group("lead") or "").lower()
        if word == "may" and not lead:
            continue  # "may" as a verb
        month = MONTHS[word]
        if lead == "last":
            year = today.year if month < today.month else today.year - 1
        else:
            year = _recent_month_year(month, today)
        start, end = _clamp(dt.date(year, month, 1), _month_end(year, month), today)
        yield TimeMatch(start, end, m.span(), m.group(0))
        return
    for m in _BARE_YEAR.finditer(text):
        year = int(m.group("yr"))
        yield TimeMatch(*_clamp(dt.date(year, 1, 1), dt.date(year, 12, 31), today), m.span(), m.group(0))
        return


def _plausible_single_date(m: re.Match) -> bool:
    """A bare 'm/d' could be a ratio; accept it only with a lead word or when it contains a month name/year."""
    s = m.group(0).lower()
    if re.search(r"[a-z]{3,}", s.replace("on", "").replace("for", "").replace("the", "")
                 .replace("of", "").replace("at", "")):
        return True
    return bool(re.match(r"\s*(on|for|of|at)\b", s)) or bool(re.search(r"(19|20)\d\d", s))
