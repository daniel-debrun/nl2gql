"""Deterministic evidence extraction: spans, values and cues. It proposes; the network decides.

Order matters. Times, hierarchy keys and names, shifts and counts are found first and their spans are
masked, so the digits inside "Q3", "line 3" or "last 7 days" cannot leak into another slot. Cue words
(metrics, subgraph hints, ranking and grouping words) are then read from what is left.
"""

from __future__ import annotations

import datetime as dt
import difflib
import re
from dataclasses import dataclass, field

from nl2gql import hierarchy, lexicon, timeparse
from nl2gql.hierarchy import LEVEL_INDEX, LEVELS

_NUMBER_WORDS = {k: v for k, v in timeparse.NUMBER_WORDS.items() if " " not in k and len(k) > 2}
_NUM = r"(\d{1,4}|" + "|".join(sorted(_NUMBER_WORDS, key=len, reverse=True)) + r")"
_L, _R = r"(?<![\w-])", r"(?![\w-])"
PLANT_NAMES = {hierarchy.NODES[k].name.lower(): k for k in hierarchy.keys_at("plant")}


def _alternation(phrases) -> str:
    return "|".join(re.escape(p) for p in sorted(set(phrases), key=len, reverse=True))


def _to_int(text: str) -> int | None:
    text = text.lower()
    return int(text) if text.isdigit() else _NUMBER_WORDS.get(text)


@dataclass
class Evidence:
    text: str
    time: timeparse.TimeMatch | None = None
    shift: int | None = None
    count: int | None = None
    singular: bool = False
    metric_lexemes: list[str] = field(default_factory=list)
    subgraph_cues: list[str] = field(default_factory=list)
    rank_cues: list[str] = field(default_factory=list)
    agg_level: str | None = None
    agg_form: str | None = None
    raw_cue: bool = False
    raw_weak: bool = False
    unknown_measure: str | None = None
    filters: dict[str, str] = field(default_factory=dict)
    problems: list[tuple[str, str]] = field(default_factory=list)
    masked: str = ""

    @property
    def metric_lexeme(self) -> str | None:
        return self.metric_lexemes[0] if self.metric_lexemes else None

    @property
    def subgraph_cue(self) -> str | None:
        return self.subgraph_cues[0] if self.subgraph_cues else None

    @property
    def rank_cue(self) -> str | None:
        """The most decisive ranking cue: best/worst and top/bottom beat a bare 'rank'."""
        for cue in ("best", "worst", "bottom", "top", "rank"):
            if cue in self.rank_cues:
                return cue
        return None

    @property
    def raw_strength(self) -> str:
        return "strong" if self.raw_cue else "weak" if self.raw_weak else "no"

    @property
    def finest_filter(self) -> str | None:
        return max(self.filters, key=LEVEL_INDEX.get) if self.filters else None

    @property
    def location_kind(self) -> str | None:
        """press_shop / machining / assembly when a filter sits at or below a subarea."""
        finest = self.finest_filter
        if finest and LEVEL_INDEX[finest] >= LEVEL_INDEX["subarea"]:
            return hierarchy.subarea_kind(self.filters[finest])
        return None

    @property
    def domain_signals(self) -> int:
        return sum(bool(x) for x in (self.metric_lexemes, self.subgraph_cues, self.filters, self.agg_level,
                                     self.rank_cues, self.time, self.shift, self.raw_cue))


class _Text:
    """Lower-cased text with a mask of consumed characters."""

    def __init__(self, text: str) -> None:
        self.original = text
        # "scrap-rate" reads as "scrap rate"; same length, so spans still line up with the original
        self.lower = re.sub(r"(?<=[a-z])-(?=[a-z])", " ", text.lower())
        self.used = [False] * len(text)
        self.labels: list[tuple[int, int, str]] = []

    def free(self, start: int, end: int) -> bool:
        return not any(self.used[start:end])

    def take(self, start: int, end: int, label: str) -> None:
        for i in range(start, end):
            self.used[i] = True
        self.labels.append((start, end, label))

    def finditer(self, pattern: re.Pattern, original: bool = False):
        """Matches that avoid consumed text. A match that overlaps consumed text is retried one
        character later, so "line 2 shift 1" still finds "shift 1" after "2 shift" is refused."""
        text = self.original if original else self.lower
        pos = 0
        while pos <= len(text):
            m = pattern.search(text, pos)
            if m is None:
                return
            if self.free(*m.span()):
                yield m
                pos = max(m.end(), m.start() + 1)
            else:
                pos = m.start() + 1

    def masked(self) -> str:
        out, i = [], 0
        for start, end, label in sorted(self.labels):
            out.append(self.lower[i:start])
            out.append(f" __{label}__ ")
            i = end
        out.append(self.lower[i:])
        return re.sub(r"\s+", " ", "".join(out)).strip()


# --- patterns -------------------------------------------------------------------------------------

_KEY = re.compile(r"(?<![\w-])([A-Z]{2,4}(?:-(?:PS|MC|AS|A[1-6]0|L\d\d))?)(?![\w-])")
_LINE = re.compile(_L + r"(?:production\s+)?(?:line|ln)\.?\s*(?:#|no\.?|number|num)?\s*" + _NUM
                   + r"(?:\s*(?:,|and|&|or|-|to)\s*" + _NUM + r")?" + _R, re.I)
_LINE_CODE = re.compile(r"(?<![\w-])L0?(\d{1,2})(?![\w-])")
_AREA = re.compile(_L + r"(?:area|zone|cell)\s*(?:#|no\.?|number)?\s*" + _NUM + _R, re.I)
_AREA_CODE = re.compile(r"(?<![\w-])A([1-9]0)(?![\w-])")
_SHIFT = re.compile(
    _L + r"(?:shift\s*(?:#|no\.?|number)?\s*(?P<n>[123]|one|two|three)"
    r"|(?P<o>first|second|third|1st|2nd|3rd)\s+shift"
    r"|(?P<w>day|morning|afternoon|swing|evening|late|night|graveyard|overnight)\s+shift"
    r"|s(?P<s>[123])\b|(?P<n2>[123])(?:st|nd|rd)?\s+shift)" + _R, re.I)
_SHIFT_WORDS = {"day": 1, "morning": 1, "afternoon": 2, "swing": 2, "evening": 2, "late": 2,
                "night": 3, "graveyard": 3, "overnight": 3}
_ORD = {"first": 1, "1st": 1, "second": 2, "2nd": 2, "third": 3, "3rd": 3, "one": 1, "two": 2, "three": 3}

_PLURAL_ALT = _alternation(lexicon.PLURAL_LEVEL_WORDS)
_RANK_WORDS = r"(?:top|bottom|best|worst|highest|lowest|leading|first|last|latest|most|least|biggest|largest|smallest|fewest|poorest|worst-performing|best-performing|top-performing)"
_COUNT = [
    re.compile(_L + _RANK_WORDS + r"\s+" + _NUM + _R, re.I),
    re.compile(_L + _NUM + r"\s+(?:" + _RANK_WORDS[3:-1] + r"|worst|best)" + _R, re.I),
    re.compile(_L + _NUM + r"\s+(?:[a-z]+\s+){0,2}?(?:" + _PLURAL_ALT + r"|records|rows|readings|entries|results|items|data points)" + _R, re.I),
    re.compile(_L + r"(?:limit(?:ed)?(?:\s+(?:it|to|results\s+to))*|max(?:imum)?\s+of|only|just)\s+" + _NUM + _R, re.I),
]

_SINGULAR_LEVEL = r"(group|region|division|business unit|plant|site|factory|facility|location|subarea|sub-area|department|section|shop|area|zone|cell|line|production line)"
_SINGULAR = [
    re.compile(_L + r"(?:which|what)\s+(?:one\s+|single\s+)?" + _SINGULAR_LEVEL + _R, re.I),
    re.compile(_L + r"the\s+(?:single\s+)?(?:best|worst|top|highest|lowest|poorest|strongest|weakest|best[- ]performing|worst[- ]performing|top[- ]performing)"
               r"(?:\s+\w+)?\s+" + _SINGULAR_LEVEL + _R, re.I),
    re.compile(_L + _SINGULAR_LEVEL + r"\s+(?:with|that had|that has|had|has|having|showed|shows|saw|that saw|ran)\s+the\s+"
               r"(?:most|highest|lowest|least|best|worst|biggest|largest|smallest|fewest|longest|shortest|max|min|top)" + _R, re.I),
    re.compile(_L + r"(?:who|which\s+one)\s+(?:had|has|ran|was|is)\s+the\s+(?:most|highest|lowest|least|best|worst)" + _R, re.I),
]

_ANY_LEVEL_ALT = _alternation(p for ps in lexicon.LEVEL_WORDS.values() for p in ps)
_LEVEL_OF = {p: lvl for lvl, ps in lexicon.LEVEL_WORDS.items() for p in ps}
_BY_FORM = re.compile(
    _L + r"(?:by|per|each|every|across|over|among|between|for\s+each|for\s+every|for\s+all|grouped\s+by|group\s+by|"
    r"broken\s+down\s+by|broken\s+out\s+by|break\s*down\s+by|breakdown\s+by|split\s+by|by\s+each|at\s+the|at|all)\s+"
    r"(?:the\s+|individual\s+|different\s+|our\s+)?(" + _ANY_LEVEL_ALT + r")(?:\s+level)?" + _R, re.I)
_WHICH_FORM = re.compile(_L + r"(?:which|what)\s+(?:one\s+|single\s+)?(" + _ANY_LEVEL_ALT + r")" + _R, re.I)
_PLURAL_FORM = re.compile(_L + r"(" + _PLURAL_ALT + r")" + _R, re.I)

_METRIC = re.compile(_L + r"(" + _alternation(p for ps in lexicon.METRIC_PHRASES.values() for p in ps) + r")" + _R, re.I)
_METRIC_OF = {p: lex for lex, ps in lexicon.METRIC_PHRASES.items() for p in ps}
_SUBGRAPH = re.compile(_L + r"(" + _alternation(p for ps in lexicon.SUBGRAPH_PHRASES.values() for p in ps) + r")" + _R, re.I)
_SUBGRAPH_OF = {p: s for s, ps in lexicon.SUBGRAPH_PHRASES.items() for p in ps}
_RANK = re.compile(_L + r"(" + _alternation(p for ps in lexicon.RANK_PHRASES.values() for p in ps) + r")" + _R, re.I)
_RANK_OF = {p: r for r, ps in lexicon.RANK_PHRASES.items() for p in ps}
_RAW = re.compile(_L + r"(" + _alternation(lexicon.RAW_PHRASES) + r")" + _R, re.I)
_ALL_DATA = re.compile(_L + r"(?:all|everything|full|complete|entire)\s+(?:of\s+)?(?:the\s+)?(?:[a-z]+\s+){0,2}(?:data|dataset|data set|numbers|info|information)" + _R, re.I)
_RAW_WEAK = re.compile(_L + r"(?:data|dataset|data set|dump|list|values)" + _R, re.I)
_RANK_SINGULAR_LEVEL = [
    re.compile(r"(?:best|worst|top|bottom|highest|lowest|poorest|strongest|weakest|most|least|biggest|largest|smallest|fewest|longest|shortest)"
               r"(?:[- ]performing)?(?:\s+(?:__metric_[a-z_]+__|[a-z]+)){0,2}?\s+" + _SINGULAR_LEVEL + r"(?!s)\b"),
    re.compile(_SINGULAR_LEVEL + r"\s+(?:with|that had|that has|had|has|having|showed|shows|saw|ran)\s+the\s+"
               r"(?:most|highest|lowest|least|best|worst|biggest|largest|smallest|fewest|longest|shortest|max|min|top)\b"),
]
_WHICH_MASKED = re.compile(r"\b(?:which|what)\s+(?:(?:__[a-z_]+__|[a-z]+)\s+){0,2}?" + _SINGULAR_LEVEL + r"\b(?!s)")
_COUNT_MASKED = re.compile(r"(?<![\w-])" + _NUM + r"\s+(?:(?:__[a-z_]+__|[a-z]+)\s+){1,3}?(?:" + _PLURAL_ALT
                           + r"|records|rows|readings|entries|results|logs|values)\b")
_KPI_BEFORE_METRIC = re.compile(_L + r"(performance\s+metrics|production\s+metrics|key\s+performance\s+indicators)" + _R, re.I)

_ABOVE_PLANT = {alias: key for key, aliases in lexicon.NAME_ALIASES.items() for alias in aliases}
_ABOVE_PLANT_RE = re.compile(_L + r"(" + _alternation(_ABOVE_PLANT) + r")" + _R, re.I)
_PLANT_RE = re.compile(_L + r"(" + _alternation(PLANT_NAMES) + r")(?:'s)?" + _R, re.I)
_SUBAREA_OF = {alias: code for code, aliases in lexicon.SUBAREA_ALIASES.items() for alias in aliases}
_SUBAREA_RE = re.compile(_L + r"(" + _alternation(_SUBAREA_OF) + r")" + _R, re.I)
_MACHINING_SUBAREA = re.compile(_L + r"(?:the\s+)?(?:" + _alternation(PLANT_NAMES) + r")(?:'s)?\s+(machining)" + _R
                                + r"|" + _L + r"(?:in|at|within|inside)\s+(?:the\s+)?(machining)(?!\s+(?:" + _ANY_LEVEL_ALT + r"|data|centers?))" + _R, re.I)
_MACHINING_LISTED = re.compile(r"(?:,|;)\s*(machining)\s*(?:,|;|$)", re.I)
_UNKNOWN_MEASURE = re.compile(
    _L + r"(?:energy|power|electricity|vibration|humidity|noise|voltage|current|coolant|oil|air|water|carbon|emissions?|"
    r"headcount|head count|safety|incidents?|injur(?:y|ies)|cost|costs|revenue|"
    r"profit|margin|throughput|output|units|parts produced|production volume|pressure|flow|footprint|consumption|usage|"
    r"temperature of the (?:room|air|building))" + _R, re.I)
_UNKNOWN_PLANT = re.compile(_L + r"([a-z][a-z'-]{3,})\s+(?:plant|site|facility|factory)" + _R, re.I)
_WORD = re.compile(r"[a-z][a-z'-]{4,}")


def _problem(ev: Evidence, reason: str, detail: str) -> None:
    if (reason, detail) not in ev.problems:
        ev.problems.append((reason, detail))


_CACHE: dict[tuple[str, dt.date], Evidence] | None = None


class caching:
    """Context manager: memoise ``extract`` while tuning. Off by default so measured latencies include extraction."""

    def __enter__(self):
        global _CACHE
        self._previous, _CACHE = _CACHE, {} if _CACHE is None else _CACHE
        return self

    def __exit__(self, *exc):
        global _CACHE
        _CACHE = self._previous


def extract(text: str, today: dt.date) -> Evidence:
    if _CACHE is not None:
        key = (text, today)
        if key not in _CACHE:
            _CACHE[key] = _extract(text, today)
        return _CACHE[key]
    return _extract(text, today)


def _extract(text: str, today: dt.date) -> Evidence:
    ev = Evidence(text=text)
    t = _Text(text)

    tm = timeparse.find(text, today)
    if tm:
        ev.time = tm
        t.take(*tm.span, "time")

    mentions: dict[str, set[str]] = {}          # level -> keys (resolved)
    descriptors: list[str] = []                  # "the Seating plant": a division used to describe a plant
    pending: list[tuple[str, str]] = []          # (level, suffix) needing a plant: ("line", "L03")

    for m in t.finditer(_KEY, original=True):
        key = m.group(1)
        n = hierarchy.NODES.get(key)
        if n:
            mentions.setdefault(n.level, set()).add(key)
            t.take(*m.span(), n.level)
    for m in t.finditer(_SUBAREA_RE):
        pending.append(("subarea", _SUBAREA_OF[m.group(1).lower()]))
        t.take(*m.span(), "subarea")
    plant_named = bool(_PLANT_RE.search(t.lower)) or any(
        hierarchy.NODES.get(m.group(1), None) is not None and hierarchy.NODES[m.group(1)].level == "plant"
        for m in _KEY.finditer(text))
    for m in t.finditer(_MACHINING_SUBAREA):
        span = m.span(1) if m.group(1) else m.span(2)
        if t.free(*span):
            pending.append(("subarea", "MC"))
            t.take(*span, "subarea")
    if plant_named:
        # "line 6, machining, Saltillo": machining listed among the locations names the subarea.
        # "the CNC lines at Wroclaw" only describes the lines, so it adds no filter (see data/CONVENTIONS.md).
        for m in t.finditer(_MACHINING_LISTED):
            if not any(level == "subarea" for level, _ in pending):
                pending.append(("subarea", "MC"))
            t.take(*m.span(1), "subarea")
    for m in t.finditer(_PLANT_RE):
        mentions.setdefault("plant", set()).add(PLANT_NAMES[m.group(1).lower()])
        t.take(*m.span(), "plant")
    for m in t.finditer(_ABOVE_PLANT_RE):
        key = _ABOVE_PLANT[m.group(1).lower()]
        level = hierarchy.NODES[key].level
        described = level == "division" and re.match(r"\s+(?:plant|site|factory|facility)\b", t.lower[m.end():])
        if described:
            descriptors.append(key)
        else:
            mentions.setdefault(level, set()).add(key)
        t.take(*m.span(), level)
    for m in list(t.finditer(_WORD)):   # typo'd plant names: "stuttgard", "wroclow"
        word = m.group(0)
        if word in PLANT_NAMES or word in _METRIC_OF:
            continue
        close = difflib.get_close_matches(word, PLANT_NAMES, n=1, cutoff=0.82)
        if close:
            mentions.setdefault("plant", set()).add(PLANT_NAMES[close[0]])
            t.take(*m.span(), "plant")
    for m in t.finditer(_LINE):
        numbers = [x for x in (m.group(1), m.group(2)) if x]
        for num in numbers:
            n = _to_int(num)
            if n is None or not 1 <= n <= 12:
                _problem(ev, "unknown_entity", f"line {num}")
            else:
                pending.append(("line", f"L{n:02d}"))
        if len(numbers) > 1:
            _problem(ev, "multiple_values", "several lines")
        t.take(*m.span(), "line")
    for m in t.finditer(_LINE_CODE, original=True):
        n = int(m.group(1))
        if 1 <= n <= 12:
            pending.append(("line", f"L{n:02d}"))
            t.take(*m.span(), "line")
    for m in t.finditer(_AREA):
        n = _to_int(m.group(1))
        if n in (10, 20, 30, 40, 50, 60):
            pending.append(("area", f"A{n}"))
        else:
            _problem(ev, "unknown_entity", f"area {m.group(1)}")
        t.take(*m.span(), "area")
    for m in t.finditer(_AREA_CODE, original=True):
        pending.append(("area", f"A{m.group(1)}"))
        t.take(*m.span(), "area")
    for m in t.finditer(_UNKNOWN_PLANT):
        word = m.group(1).lower()
        if word not in lexicon.STOP_BEFORE_PLANT and word not in _METRIC_OF and word not in _SUBGRAPH_OF \
                and word not in _LEVEL_OF and not word.endswith(("ing", "est")):
            _problem(ev, "unknown_entity", f"{word} plant")

    for m in t.finditer(_SHIFT):
        g = m.groupdict()
        if g["n"] or g["o"]:
            ev.shift = _ORD.get((g["n"] or g["o"]).lower()) or int(g["n"])
        elif g["w"]:
            ev.shift = _SHIFT_WORDS[g["w"].lower()]
        else:
            ev.shift = int(g["s"] or g["n2"])
        t.take(*m.span(), "shift")
        break

    for pattern in _COUNT:
        for m in t.finditer(pattern):
            n = _to_int(m.group(1))
            if n:
                ev.count = n
                s, e = m.span(1)
                t.take(s, e, "num")
                break
        if ev.count:
            break
    ev.singular = ev.count is None and any(p.search(t.lower) for p in _SINGULAR)

    _resolve(ev, mentions, pending, descriptors, t.lower)

    for m in t.finditer(_KPI_BEFORE_METRIC):
        ev.subgraph_cues.append("kpi")
        t.take(*m.span(), "kpi")
    for m in t.finditer(_METRIC):
        ev.metric_lexemes.append(_METRIC_OF[m.group(1).lower()])
        t.take(*m.span(), "metric_" + _METRIC_OF[m.group(1).lower()])
    for m in t.finditer(_SUBGRAPH):
        ev.subgraph_cues.append(_SUBGRAPH_OF[m.group(1).lower()])
    for m in t.finditer(_RANK):
        ev.rank_cues.append(_RANK_OF[m.group(1).lower()])
    ev.raw_cue = any(True for _ in t.finditer(_RAW)) or bool(_ALL_DATA.search(t.lower))
    ev.raw_weak = bool(_RAW_WEAK.search(t.lower))

    for pattern, form in ((_BY_FORM, "by"), (_WHICH_FORM, "which"), (_PLURAL_FORM, "plural")):
        found = [m for m in t.finditer(pattern)]
        if found:
            word = found[0].group(1).lower()
            ev.agg_level = lexicon.PLURAL_LEVEL_WORDS.get(word) or _LEVEL_OF.get(word)
            ev.agg_form = form
            break

    unknown = next(t.finditer(_UNKNOWN_MEASURE), None)
    ev.unknown_measure = unknown.group(0) if unknown else None
    ev.masked = t.masked()
    if ev.agg_level is None or ev.agg_form == "plural":
        m = _WHICH_MASKED.search(ev.masked)
        if m:
            ev.agg_level = _LEVEL_OF.get(m.group(1).lower())
            ev.agg_form = "which"
    if ev.count is None:
        m = _COUNT_MASKED.search(ev.masked)
        if m:
            ev.count = _to_int(m.group(1))
    if ev.agg_form == "which" and ev.count is None:
        ev.singular = True
    if ev.agg_level is None:
        # "the worst performing line", "highest tonnage line", "line with the most downtime"
        for pattern in _RANK_SINGULAR_LEVEL:
            m = pattern.search(ev.masked)
            if m:
                ev.agg_level = _LEVEL_OF.get(m.group(1).lower())
                ev.agg_form = "which"
                ev.singular = ev.count is None
                break
    return ev


_KIND_ACROSS_PLANTS = re.compile(r"(?:which|what|each|every|per|by|all|across|among)\s+(?:\w+\s+)?(?:plants?|sites?|factories|facilities|regions?|divisions?|groups?)\b|\bits\b")


def _resolve(ev: Evidence, mentions: dict[str, set[str]], pending: list[tuple[str, str]],
             descriptors: list[str], lower: str) -> None:
    if pending and "plant" not in mentions and _KIND_ACROSS_PLANTS.search(lower):
        # "which plant has the fewest alarms in its machining shop": a kind of subarea across plants,
        # not a filter the schema can express, so it describes the question instead of filtering it
        pending = [(lvl, suffix) for lvl, suffix in pending if lvl != "subarea"]
    for key in descriptors:
        plants = mentions.get("plant", set())
        if plants and not all(key in hierarchy.ancestors(p) for p in plants):
            mentions.setdefault("division", set()).add(key)   # contradicts the plant: surface it
    for level, keys in mentions.items():
        if len(keys) > 1:
            _problem(ev, "multiple_values", f"{level}: {sorted(keys)}")
        ev.filters[level] = sorted(keys)[0]
    plant = ev.filters.get("plant")
    if plant is None and pending:
        finer = [lvl for lvl, _ in pending]
        # a finer key (line code) may already have come in as a full key
        full = [k for lvl in ("subarea", "area", "line") for k in mentions.get(lvl, ())]
        if full:
            plant = hierarchy.plant_of(full[0])
        else:
            _problem(ev, "needs_plant", ", ".join(sorted(set(finer))))
    if plant is not None:
        for level, suffix in pending:
            key = f"{plant}-{suffix}"
            if key not in hierarchy.NODES:
                _problem(ev, "unknown_entity", key)
                continue
            if level in ev.filters and ev.filters[level] != key:
                _problem(ev, "multiple_values", f"{level}: {ev.filters[level]}, {key}")
            ev.filters.setdefault(level, key)
    keys = list(ev.filters.values())
    if len(keys) > 1 and not hierarchy.on_one_path(keys):
        _problem(ev, "inconsistent_hierarchy", ", ".join(sorted(keys)))
    for level in list(ev.filters):
        assert level in LEVELS
