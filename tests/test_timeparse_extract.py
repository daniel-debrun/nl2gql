from __future__ import annotations

import json

import pytest

from nl2gql.extract import extract
from nl2gql.timeparse import find

from .conftest import DATA, TODAY

TIME_CASES = {
    "today": ("2026-11-18", "2026-11-18"), "yesterday": ("2026-11-17", "2026-11-17"),
    "yday": ("2026-11-17", "2026-11-17"), "this week": ("2026-11-16", "2026-11-18"),
    "last week": ("2026-11-09", "2026-11-15"), "last wk": ("2026-11-09", "2026-11-15"),
    "this month": ("2026-11-01", "2026-11-18"), "last month": ("2026-10-01", "2026-10-31"),
    "last mth": ("2026-10-01", "2026-10-31"), "for the month of october": ("2026-10-01", "2026-10-31"),
    "in december": ("2025-12-01", "2025-12-31"), "during november": ("2026-11-01", "2026-11-18"),
    "march 2025": ("2025-03-01", "2025-03-31"), "in may": ("2026-05-01", "2026-05-31"),
    "this quarter": ("2026-10-01", "2026-11-18"), "last quarter": ("2026-07-01", "2026-09-30"),
    "during Q3": ("2026-07-01", "2026-09-30"), "Q4": ("2026-10-01", "2026-11-18"),
    "Q2 2025": ("2025-04-01", "2025-06-30"), "second quarter of 2026": ("2026-04-01", "2026-06-30"),
    "this year": ("2026-01-01", "2026-11-18"), "ytd": ("2026-01-01", "2026-11-18"),
    "last year": ("2025-01-01", "2025-12-31"), "in 2025": ("2025-01-01", "2025-12-31"),
    "last 7 days": ("2026-11-12", "2026-11-18"), "last seven days": ("2026-11-12", "2026-11-18"),
    "past fourteen days": ("2026-11-05", "2026-11-18"), "last 2 weeks": ("2026-11-05", "2026-11-18"),
    "since November 1": ("2026-11-01", "2026-11-18"), "since 2026-10-15": ("2026-10-15", "2026-11-18"),
    "since the beginning of october": ("2026-10-01", "2026-11-18"),
    "on Nov 3": ("2026-11-03", "2026-11-03"), "on Dec 5": ("2025-12-05", "2025-12-05"),
    "on 11/3": ("2026-11-03", "2026-11-03"), "3rd of november": ("2026-11-03", "2026-11-03"),
    "between Oct 1 and Oct 15": ("2026-10-01", "2026-10-15"),
    "from 2026-09-01 to 2026-09-30": ("2026-09-01", "2026-09-30"),
    "between November 2 and November 8, 2026": ("2026-11-02", "2026-11-08"),
    "3 days ago": ("2026-11-15", "2026-11-15"),
}


@pytest.mark.parametrize(("phrase", "expected"), TIME_CASES.items())
def test_time_conventions(phrase, expected):
    match = find(f"oee for Windsor {phrase} please", TODAY)
    assert match is not None and match.iso == expected


@pytest.mark.parametrize("text", ["top 5 lines by oee", "line 3 at Windsor", "shift 2", "you may want oee",
                                  "area 20", "WIN-L03 oee", "top 10 plants by scrap"])
def test_no_false_time_matches(text):
    assert find(text, TODAY) is None


def _worked_examples():
    return [json.loads(line) for line in (DATA / "worked_examples.jsonl").read_text().splitlines()]


def test_worked_example_filters_and_problems():
    expected_problems = {"G-016": "needs_plant", "G-017": "inconsistent_hierarchy", "G-019": "unknown_entity"}
    for ex in _worked_examples():
        ev = extract(ex["utterance"], TODAY)
        reasons = {r for r, _ in ev.problems}
        if ex["id"] in expected_problems:
            assert expected_problems[ex["id"]] in reasons, ex["utterance"]
        else:
            assert not reasons, (ex["utterance"], ev.problems)


@pytest.mark.parametrize(("text", "field", "value"), [
    ("press data raw for stuttgart line 2 shift 1 on dec 5", "shift", 1),       # "2 shift" must not eat "shift"
    ("downtime minutes by line in the saltillo assembly area 50", "filters",
     {"plant": "SAL", "subarea": "SAL-AS", "area": "SAL-A50"}),
    ("downtime minutes for nagoya seating plant", "filters", {"plant": "NAG"}),   # division as a descriptor
    ("whos the worst performing line in nagoya for quality", "agg_level", "line"),
    ("highest tonnage line in the whole company this yr", "singular", True),
    ("give me all press data for the puebla press shop", "raw_cue", True),
    ("cycle time on line 7 at Brno", "location_kind", "machining"),
    ("top 5 lines by oee in Stuttgard", "filters", {"plant": "STU"}),             # typo'd plant name
    ("5 lines with the lowest stroke rate in Graz during Q3", "count", 5),
    ("which plant had the worst scrap rate", "singular", True),
    ("oee for line 14 at Graz", "problems", [("unknown_entity", "line 14")]),
])
def test_extraction_regressions(text, field, value):
    ev = extract(text, TODAY)
    assert getattr(ev, field) == value
