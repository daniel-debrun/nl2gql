"""Schema-driven synthetic data, after Wang, Berant and Liang, "Building a Semantic Parser Overnight" (2015).

Sample a legal slot assignment, render it through templates and synonym choices, and label it with
the query the builder produces. The labels follow ``data/CONVENTIONS.md`` exactly, because the
same conventions are applied in code. Clarify and reject examples are generated too.

This data has a known weakness: its phrasing is only as varied as the templates. It is used for
augmentation and for the "synthetic only" ablation; the held-out test set is a separate fabricated set
in a style the templates never saw.
"""

from __future__ import annotations

import datetime as dt
import random

from nl2gql import hierarchy, lexicon
from nl2gql.builder import to_graphql
from nl2gql.hierarchy import LEVEL_INDEX, LEVELS, NODES
from nl2gql.model import Example
from nl2gql.schema import POLARITY, SPEC
from nl2gql.slots import Slots
from nl2gql.timeparse import find as find_time

LEVEL_SINGULAR = {"group": "group", "region": "region", "division": "division", "plant": "plant",
                  "subarea": "subarea", "area": "area", "line": "line"}
LEVEL_PLURAL = {"group": "groups", "region": "regions", "division": "divisions", "plant": "plants",
                "subarea": "subareas", "area": "areas", "line": "lines"}
LEVEL_PLURAL_ALT = {"plant": ["plants", "sites", "factories"], "subarea": ["subareas", "departments"],
                    "line": ["lines", "production lines"], "area": ["areas", "zones"]}
SUBGRAPH_WORDS = {"kpis": ["kpi", "kpis", "production kpis"], "cnc_data": ["cnc", "cnc data", "machining"],
                  "press_data": ["press", "press data", "stamping"]}
TIME_PHRASES = [
    "today", "yesterday", "this week", "last week", "this month", "last month", "this quarter",
    "last quarter", "this year", "last year", "ytd", "in january", "in march", "for june", "during august",
    "in september", "for october", "the month of october", "in november", "in december", "march 2025",
    "in Q1", "during Q2", "for Q3", "Q4", "Q2 2025", "in 2025", "last 7 days", "the past 14 days",
    "last 30 days", "past 2 weeks", "last 4 weeks", "since November 1", "since oct 15", "on Nov 3",
    "on Oct 28", "between Oct 1 and Oct 15", "from 2026-09-01 to 2026-09-30", "since the beginning of october",
    "last seven days", "yday", "last wk", "last mth",
]
SHIFT_PHRASES = [("shift 1", 1), ("shift 2", 2), ("shift 3", 3), ("first shift", 1), ("second shift", 2),
                 ("third shift", 3), ("day shift", 1), ("afternoon shift", 2), ("night shift", 3),
                 ("2nd shift", 2), ("the 3rd shift", 3)]
NUMBER_WORDS = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 10: "ten", 20: "twenty"}
OFF_TOPIC = [
    "what's the weather in {plant} tomorrow", "book a meeting room for 3pm", "tell me a joke",
    "how many vacation days do I have left", "write a python script to parse csv files",
    "who is the plant manager at {plant}", "translate this email to spanish", "what's for lunch in the cafeteria",
    "remind me to call maintenance at 4", "can you order more safety gloves", "what time is it in {plant}",
    "summarize the latest news", "how do I reset my password", "is the parking lot at {plant} open",
    "send the shift schedule to my team", "what is the capital of austria", "play some music",
    "draft a performance review for my operator", "how do I request overtime pay", "where is the first aid kit",
    "explain how a stamping press works", "what's the best cnc machine to buy", "export my calendar to excel",
    "build me a pivot table", "who won the game last night", "schedule a forklift inspection",
    "what are the covid rules at {plant}", "set an alarm for 6am", "how long is the drive to {plant}",
    "recommend a good book about lean manufacturing",
]


def _pick(rng: random.Random, xs):
    return xs[rng.randrange(len(xs))]


def _name(key: str) -> str:
    return NODES[key].name


def _metric_word(rng: random.Random, metric: str) -> str:
    phrases = [p for p in lexicon.METRIC_PHRASES[metric] if len(p) > 2 or metric in ("tool_wear",)]
    return _pick(rng, phrases)


class Generator:
    def __init__(self, today: dt.date, seed: int = 0) -> None:
        self.today = today
        self.rng = random.Random(seed)

    # --- pieces -----------------------------------------------------------------------------------

    def filter_chain(self, max_levels: int, min_finest: str | None = None,
                     subgraph: str | None = None) -> dict[str, str]:
        """0..max_levels filter levels on one path. Levels below plant always come with their plant.

        Machine data is usually asked about where the machines are: CNC queries mostly filter to
        Machining, press queries to the Press Shop, KPI queries anywhere.
        """
        rng = self.rng
        n = rng.choices([0, 1, 2, 3, 4], weights=[3, 5, 4, 2, 1])[0]
        n = min(n, max_levels)
        if n == 0:
            return {}
        leaf_level = rng.choice(LEVELS if min_finest is None else LEVELS[: LEVEL_INDEX[min_finest] + 1])
        candidates = hierarchy.keys_at(leaf_level)
        home = {"cnc_data": "machining", "press_data": "press_shop"}.get(subgraph)
        if home and LEVEL_INDEX[leaf_level] >= LEVEL_INDEX["subarea"] and rng.random() < 0.85:
            candidates = [k for k in candidates if hierarchy.subarea_kind(k) == home]
        leaf = rng.choice(candidates)
        path = list(reversed(hierarchy.ancestors(leaf)))       # group .. leaf
        chosen = {NODES[leaf].level: leaf}
        if LEVEL_INDEX[leaf_level] > LEVEL_INDEX["plant"]:
            chosen["plant"] = hierarchy.plant_of(leaf)
        others = [k for k in path if NODES[k].level not in chosen]
        rng.shuffle(others)
        for k in others[: max(0, n - len(chosen))]:
            chosen[NODES[k].level] = k
        return chosen

    def location_text(self, filters: dict[str, str]) -> str:
        rng = self.rng
        parts = []
        plant = filters.get("plant")
        for level in sorted(filters, key=LEVEL_INDEX.get, reverse=True):
            key = filters[level]
            if level == "line":
                parts.append(_pick(rng, [f"line {int(key[-2:])}", f"Line {int(key[-2:])}", f"line #{int(key[-2:])}"]))
            elif level == "area":
                parts.append(f"area {key.split('-A')[1]}")
            elif level == "subarea":
                parts.append(_pick(rng, {"PS": ["the press shop", "press shop"], "MC": ["the machining subarea",
                                          "machining department"], "AS": ["assembly", "the assembly hall"]}[key[-2:]]))
            elif level == "plant":
                parts.append(_pick(rng, [_name(key), f"the {_name(key)} plant", _name(key) + " site"]))
            elif level == "region":
                parts.append(_pick(rng, [_name(key), {"AMER": "the americas", "EUR": "europe", "APAC": "APAC"}[key]]))
            else:
                parts.append(_name(key))
        if not parts:
            return ""
        joiner = _pick(rng, [" in ", " at ", ", ", " of "])
        text = joiner.join(parts)
        assert plant is None or plant in filters.values()
        return _pick(rng, ["in ", "at ", "for ", "across "]) + text

    def time_text(self) -> tuple[str, tuple[str, str] | None]:
        if self.rng.random() < 0.25:
            return "", None
        phrase = _pick(self.rng, TIME_PHRASES)
        match = find_time(phrase, self.today)
        return phrase, match.iso if match else None

    def shift_text(self) -> tuple[str, int | None]:
        if self.rng.random() < 0.8:
            return "", None
        phrase, n = _pick(self.rng, SHIFT_PHRASES)
        return _pick(self.rng, ["on ", "", "for "]) + phrase, n

    # --- examples ---------------------------------------------------------------------------------

    def query(self) -> Example:
        rng = self.rng
        subgraph = rng.choices(list(SPEC.subgraphs), weights=[4, 3, 3])[0]
        metrics = SPEC.subgraph(subgraph).metrics
        qtype = rng.choices(["ranked", "aggregate", "raw"], weights=[35, 40, 25])[0]
        metric = rng.choice(metrics) if (qtype == "ranked" or rng.random() < 0.9) else None
        filters = self.filter_chain(max_levels=4, subgraph=subgraph,
                                    min_finest="plant" if qtype != "raw" and rng.random() < 0.6 else None)
        finest = max(filters, key=LEVEL_INDEX.get) if filters else None
        time_phrase, time_range = self.time_text()
        shift_phrase, shift = self.shift_text()
        cue = ""
        if metric == "cycle_time":
            loc_kind = hierarchy.subarea_kind(filters[finest]) if finest and LEVEL_INDEX[finest] >= 4 else None
            decided = {"machining": "cnc_data", "press_shop": "press_data"}.get(loc_kind)
            if decided != subgraph:
                cue = _pick(rng, SUBGRAPH_WORDS[subgraph][:2]) if subgraph != "kpis" else ""
        mword = _metric_word(rng, metric) if metric else ""
        subword = _pick(rng, SUBGRAPH_WORDS[subgraph])
        loc = self.location_text(filters)
        limit = None
        agg = direction = None
        if qtype == "ranked":
            finer = [lv for lv in LEVELS if finest is None or LEVEL_INDEX[lv] > LEVEL_INDEX[finest]]
            agg = rng.choice(finer) if finer and rng.random() < 0.9 else None
            polarity = POLARITY[metric]
            cues = ["top", "highest", "bottom", "lowest", "rank"] + (["best", "worst"] if polarity != "neutral" else [])
            word = rng.choice(cues)
            direction = {"top": "DESC", "highest": "DESC", "bottom": "ASC", "lowest": "ASC", "rank": "DESC",
                         "best": "DESC" if polarity == "higher" else "ASC",
                         "worst": "ASC" if polarity == "higher" else "DESC"}[word]
            count = rng.choice([None, None, 3, 5, 10, 20, 2, 4])
            singular = count is None and agg is not None and rng.random() < 0.35
            limit = count if count else (1 if singular else 10)
            num = "" if count is None else (NUMBER_WORDS.get(count) if rng.random() < 0.3 else str(count)) or str(count)
            what = "readings" if agg is None else (LEVEL_SINGULAR[agg] if singular else
                                                   _pick(rng, LEVEL_PLURAL_ALT.get(agg, [LEVEL_PLURAL[agg]])))
            m = f"{cue} {mword}".strip()
            if singular:
                text = _pick(rng, [f"which {what} had the {word if word not in ('top', 'bottom', 'rank') else 'highest'} {m}",
                                   f"the {what} with the {word if word in ('highest', 'lowest', 'best', 'worst') else 'highest'} {m}"])
                if word in ("top", "bottom", "rank"):
                    direction = "DESC"
                if word == "bottom":
                    text = f"which {what} had the lowest {m}"
                    direction = "ASC"
            elif word == "rank":
                text = _pick(rng, [f"rank {what} by {m}", f"rank the {num} {what} by {m}".replace("  ", " "),
                                   f"{what} ranked by {m}"])
            else:
                text = _pick(rng, [f"{word} {num} {what} by {m}", f"{num} {what} with the {word} {m}",
                                   f"show the {word} {num} {what} for {m}", f"give me {word} {num} {what} by {m}"])
        elif qtype == "aggregate":
            if finest is None and rng.random() < 0.35:
                agg, grouping = "group", ""          # "scrap rate last month": no place named -> GROUP
            elif rng.random() < 0.55 or finest is None:
                finer = [lv for lv in LEVELS if finest is None or LEVEL_INDEX[lv] > LEVEL_INDEX[finest]]
                agg = rng.choice(finer) if finer else finest
                grouping = _pick(rng, [f"by {agg}", f"per {agg}", f"for each {agg}", f"broken down by {agg}",
                                       f"grouped by {agg}", f"across {LEVEL_PLURAL[agg]}"])
            else:
                agg = finest
                grouping = ""
            body = f"{cue} {mword}".strip() if metric else _pick(rng, [f"{subword} summary", f"all {subword}"])
            if grouping:
                text = _pick(rng, [f"{body} {grouping}", f"average {body} {grouping}", f"show {body} {grouping}"])
            else:
                text = _pick(rng, [f"what was the {body}", f"how was {body}", f"{body}", f"give me the {body}"])
        else:
            count = rng.choice([None, None, None, 20, 50, 100])
            limit = count
            body = f"{cue} {mword}".strip() if metric else f"all {subword} data"
            what = _pick(rng, ["readings", "records", "rows", "entries", "log", "raw data"])
            prefix = f"first {count} " if count else ""
            text = _pick(rng, [f"show me {prefix}{body} {what}", f"list {prefix}{body} {what}",
                               f"pull {prefix}{body} {what}", f"{body} by shift", f"{body} for each day"])
            if count and "first" not in text:
                text = f"first {count} " + text
        text = " ".join(x for x in (text, loc, time_phrase, shift_phrase) if x)
        if metric is None and qtype != "raw" and subword not in text:
            text = f"{subword} {text}"
        slots = Slots(subgraph=subgraph, metric=metric, aggregate_by=agg, direction=direction,
                      filters=tuple(filters.items()), time_range=time_range, shift=shift, limit=limit)
        to_graphql(slots)
        return Example(text, "query", slots, source="synthetic")

    def clarify(self) -> Example:
        rng = self.rng
        reason = rng.choice(["needs_plant", "ambiguous_metric", "missing_metric", "inconsistent_hierarchy",
                             "unknown_entity"])
        t, _ = self.time_text()
        if reason == "needs_plant":
            text = _pick(rng, [f"oee for line {rng.randint(1, 12)}", f"scrap rate in area {rng.choice([10, 20, 50])}",
                               "downtime in the press shop", f"alarm count on line {rng.randint(5, 8)}"])
        elif reason == "ambiguous_metric":
            text = _pick(rng, ["top lines by cycle time", "cycle time by plant", "average cycle time in europe",
                               "which plant had the longest cycle time"])
        elif reason == "missing_metric":
            text = _pick(rng, ["best 5 lines", "top plants", "rank the regions", "which line did the best",
                               "worst areas in " + _name(rng.choice(hierarchy.keys_at("plant")))])
        elif reason == "inconsistent_hierarchy":
            plant = rng.choice(hierarchy.keys_at("plant"))
            region = rng.choice([r for r in hierarchy.keys_at("region") if r != hierarchy.ancestor_at(plant, "region")])
            text = f"{_pick(rng, ['oee', 'scrap rate', 'downtime'])} for {_name(plant)} in {_name(region)}"
        else:
            text = _pick(rng, ["downtime at the detroit plant", "oee for line 14 at Graz",
                               "scrap rate in area 70 of Brno", "kpis for the Houston plant"])
        return Example(" ".join(x for x in (text, t) if x), "clarify", reason=reason, source="synthetic")

    def reject(self) -> Example:
        plant = _name(self.rng.choice(hierarchy.keys_at("plant")))
        return Example(_pick(self.rng, OFF_TOPIC).format(plant=plant), "reject", reason="out_of_domain",
                       source="synthetic")

    def examples(self, n: int) -> list[Example]:
        out = []
        for _ in range(n):
            r = self.rng.random()
            out.append(self.query() if r < 0.78 else self.clarify() if r < 0.9 else self.reject())
        return out
