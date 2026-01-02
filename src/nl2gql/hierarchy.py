"""The example factory hierarchy: group > region > division > plant > subarea > area > line.

Keys are what queries filter on. Names are what people say. Group, region, division and plant names
are unique. Subarea, area and line names repeat in every plant ("Line 3", "Press Shop"), so resolving
them needs a plant, exactly as it would in a real organisation.

Every plant has the same layout:

    Press Shop (PS)  Area 10: Line 1, Line 2    Area 20: Line 3, Line 4
    Machining (MC)   Area 30: Line 5, Line 6    Area 40: Line 7, Line 8
    Assembly (AS)    Area 50: Line 9, Line 10   Area 60: Line 11, Line 12
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cache

LEVELS = ("group", "region", "division", "plant", "subarea", "area", "line")
LEVEL_INDEX = {level: i for i, level in enumerate(LEVELS)}

_GROUPS = {"MOB": "Acme Mobility", "IND": "Acme Industrial"}
_REGIONS = {"AMER": ("Americas", "MOB"), "EUR": ("Europe", "MOB"), "APAC": ("Asia Pacific", "IND")}
_DIVISIONS = {
    "BODY": ("Body Systems", "AMER"), "PWT": ("Powertrain", "AMER"),
    "CHS": ("Chassis", "EUR"), "INT": ("Interiors", "EUR"),
    "ELX": ("Electronics", "APAC"), "SEAT": ("Seating", "APAC"),
}
_PLANTS = {
    "WIN": ("Windsor", "BODY"), "PUE": ("Puebla", "BODY"),
    "TOL": ("Toledo", "PWT"), "SAL": ("Saltillo", "PWT"),
    "GRZ": ("Graz", "CHS"), "WRO": ("Wroclaw", "CHS"),
    "STU": ("Stuttgart", "INT"), "BRN": ("Brno", "INT"),
    "SUZ": ("Suzhou", "ELX"), "PEN": ("Penang", "ELX"),
    "CHE": ("Chennai", "SEAT"), "NAG": ("Nagoya", "SEAT"),
}
# subarea code, name, [(area number, [line numbers])]
PLANT_LAYOUT = (
    ("PS", "Press Shop", ((10, (1, 2)), (20, (3, 4)))),
    ("MC", "Machining", ((30, (5, 6)), (40, (7, 8)))),
    ("AS", "Assembly", ((50, (9, 10)), (60, (11, 12)))),
)
SUBAREA_KINDS = {"PS": "press_shop", "MC": "machining", "AS": "assembly"}


@dataclass(frozen=True)
class Node:
    key: str
    name: str
    level: str
    parent: str | None
    children: tuple[str, ...] = field(default=(), compare=False)


def _build() -> dict[str, Node]:
    raw: list[tuple[str, str, str, str | None]] = []
    raw += [(k, n, "group", None) for k, n in _GROUPS.items()]
    raw += [(k, n, "region", p) for k, (n, p) in _REGIONS.items()]
    raw += [(k, n, "division", p) for k, (n, p) in _DIVISIONS.items()]
    for plant, (name, division) in _PLANTS.items():
        raw.append((plant, name, "plant", division))
        for code, sub_name, areas in PLANT_LAYOUT:
            sub_key = f"{plant}-{code}"
            raw.append((sub_key, sub_name, "subarea", plant))
            for area, lines in areas:
                area_key = f"{plant}-A{area}"
                raw.append((area_key, f"Area {area}", "area", sub_key))
                raw += [(f"{plant}-L{line:02d}", f"Line {line}", "line", area_key) for line in lines]
    children: dict[str, list[str]] = {}
    for key, _, _, parent in raw:
        if parent is not None:
            children.setdefault(parent, []).append(key)
    return {key: Node(key, name, level, parent, tuple(children.get(key, ())))
            for key, name, level, parent in raw}


NODES: dict[str, Node] = _build()


def node(key: str) -> Node:
    try:
        return NODES[key]
    except KeyError:
        raise KeyError(f"unknown hierarchy key {key!r}") from None


@cache
def ancestors(key: str) -> tuple[str, ...]:
    """Keys from the node itself up to its group, nearest first."""
    out, current = [], NODES[key]
    while True:
        out.append(current.key)
        if current.parent is None:
            return tuple(out)
        current = NODES[current.parent]


def ancestor_at(key: str, level: str) -> str | None:
    return next((k for k in ancestors(key) if NODES[k].level == level), None)


def on_one_path(keys: list[str]) -> bool:
    """True when every key is an ancestor or descendant of every other (a consistent filter)."""
    ordered = sorted(keys, key=lambda k: LEVEL_INDEX[NODES[k].level])
    return all(ordered[i] in ancestors(ordered[i + 1]) for i in range(len(ordered) - 1))


def plant_of(key: str) -> str | None:
    return ancestor_at(key, "plant")


def subarea_kind(key: str) -> str | None:
    """press_shop, machining or assembly for keys at or below a subarea."""
    sub = ancestor_at(key, "subarea")
    return SUBAREA_KINDS[sub.rsplit("-", 1)[1]] if sub else None


def keys_at(level: str) -> list[str]:
    return [k for k, n in NODES.items() if n.level == level]


def by_name(level: str, name: str) -> list[str]:
    """All keys at a level whose display name matches, case-insensitively."""
    wanted = name.casefold()
    return [k for k, n in NODES.items() if n.level == level and n.name.casefold() == wanted]


def describe() -> str:
    """Human-readable table of the hierarchy, for documentation."""
    lines = ["group > region > division > plant (key: name)"]
    for g in keys_at("group"):
        lines.append(f"{g}: {NODES[g].name}")
        for r in NODES[g].children:
            lines.append(f"  {r}: {NODES[r].name}")
            for d in NODES[r].children:
                lines.append(f"    {d}: {NODES[d].name}")
                for p in NODES[d].children:
                    lines.append(f"      {p}: {NODES[p].name}")
    return "\n".join(lines)
