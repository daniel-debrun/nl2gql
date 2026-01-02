from __future__ import annotations

import datetime as dt

import pytest
from graphql import parse, print_ast
from hypothesis import given, settings
from hypothesis import strategies as st

from nl2gql import hierarchy
from nl2gql.builder import BuildError, to_graphql
from nl2gql.checker import InvalidQuery, check, to_slots
from nl2gql.hierarchy import LEVELS
from nl2gql.schema import SPEC
from nl2gql.slots import Slots

# Hand-written expected queries: the builder must produce exactly these ASTs.
GOLDEN = [
    (Slots("kpis", "oee", "line", "DESC", (("region", "AMER"),), ("2026-10-01", "2026-10-31"), limit=10),
     'query { kpis(filter: {region: "AMER"}, aggregateBy: LINE, timeRange: {start: "2026-10-01", end: "2026-10-31"},'
     ' orderBy: {metric: OEE, direction: DESC}, limit: 10) { line oee } }'),
    (Slots("kpis", "scrap_rate", "plant", "DESC", time_range=("2026-10-01", "2026-10-31"), limit=1),
     'query { kpis(aggregateBy: PLANT, timeRange: {start: "2026-10-01", end: "2026-10-31"},'
     ' orderBy: {metric: SCRAP_RATE, direction: DESC}, limit: 1) { plant scrap_rate } }'),
    (Slots("cnc_data", "spindle_load", filters=(("plant", "TOL"), ("line", "TOL-L06")), shift=3),
     'query { cnc_data(filter: {plant: "TOL", line: "TOL-L06"}, shift: 3) { date shift line spindle_load } }'),
    (Slots("press_data", "cycle_time", "area", filters=(("plant", "SAL"), ("subarea", "SAL-PS"))),
     'query { press_data(filter: {plant: "SAL", subarea: "SAL-PS"}, aggregateBy: AREA) { area cycle_time } }'),
    (Slots("cnc_data", None, filters=(("plant", "PUE"), ("area", "PUE-A30")), shift=2),
     'query { cnc_data(filter: {plant: "PUE", area: "PUE-A30"}, shift: 2) { date shift line spindle_load'
     ' spindle_speed feed_rate tool_wear cycle_time alarm_count } }'),
    (Slots("press_data", "reject_count", "region", "DESC", (("group", "MOB"),), limit=3),
     'query { press_data(filter: {group: "MOB"}, aggregateBy: REGION, orderBy: {metric: REJECT_COUNT,'
     ' direction: DESC}, limit: 3) { region reject_count } }'),
    (Slots("kpis", "availability", "division", "DESC", time_range=("2025-01-01", "2025-12-31"), limit=10),
     'query { kpis(aggregateBy: DIVISION, timeRange: {start: "2025-01-01", end: "2025-12-31"},'
     ' orderBy: {metric: AVAILABILITY, direction: DESC}, limit: 10) { division availability } }'),
    (Slots("press_data", "stroke_rate", None, "ASC", (("plant", "GRZ"), ("line", "GRZ-L03")), limit=5),
     'query { press_data(filter: {plant: "GRZ", line: "GRZ-L03"}, orderBy: {metric: STROKE_RATE, direction: ASC},'
     ' limit: 5) { date shift line stroke_rate } }'),
    (Slots("kpis", None, "plant", filters=(("region", "EUR"), ("division", "CHS"))),
     'query { kpis(filter: {region: "EUR", division: "CHS"}, aggregateBy: PLANT) { plant oee availability'
     ' performance quality scrap_rate downtime_minutes } }'),
    (Slots("cnc_data", "alarm_count", "line", filters=(("group", "IND"), ("region", "APAC"), ("plant", "NAG"),
                                                       ("subarea", "NAG-MC")), shift=1),
     'query { cnc_data(filter: {group: "IND", region: "APAC", plant: "NAG", subarea: "NAG-MC"}, aggregateBy: LINE,'
     ' shift: 1) { line alarm_count } }'),
]


@pytest.mark.parametrize(("slots", "expected"), GOLDEN)
def test_golden_queries(slots, expected):
    built = to_graphql(slots)
    assert print_ast(parse(built)) == print_ast(parse(expected))
    assert check(built) == []
    assert to_slots(expected) == slots


def test_every_subgraph_metric_and_query_type_builds_valid():
    for sub in SPEC.subgraphs.values():
        for metric in (*sub.metrics, None):
            for level in (None, *LEVELS):
                for direction in (None, "ASC", "DESC"):
                    if direction and metric is None:
                        continue
                    slots = Slots(sub.name, metric, level, direction, limit=5 if direction else None)
                    query = to_graphql(slots)
                    assert check(query) == []
                    assert to_slots(query) == slots


@pytest.mark.parametrize("slots", [
    Slots("kpis", "spindle_load"),                     # metric from another subgraph
    Slots("cnc_data", "oee", direction="DESC"),
    Slots("nope", "oee"),
])
def test_builder_refuses_what_the_schema_lacks(slots):
    with pytest.raises(BuildError):
        to_graphql(slots)


def test_slots_reject_ranked_without_metric():
    with pytest.raises(ValueError):
        Slots("kpis", None, direction="DESC")


@pytest.mark.parametrize(("query", "code"), [
    ("query { kpis { oee ", "syntax"),
    ('query { kpis(orderBy: {metric: SPINDLE_LOAD, direction: DESC}) { spindle_load } }', "schema"),
    ("mutation { kpis { oee } }", "shape"),
    ("query { schemaVersion }", "shape"),
    ("query($p: ID) { kpis(filter: {plant: $p}) { oee } }", "shape"),
    ('query { kpis(filter: {plant: "DET"}) { oee } }', "filter_key_unknown"),
    ('query { kpis(filter: {plant: "AMER"}) { oee } }', "filter_level_mismatch"),
    ('query { kpis(filter: {region: "AMER", plant: "GRZ"}) { oee } }', "filter_inconsistent"),
    ('query { kpis(filter: {subarea: "WIN-PS", line: "WIN-L09"}) { oee } }', "filter_inconsistent"),
    ('query { kpis(filter: {group: "MOB", region: "AMER", division: "BODY", plant: "WIN", line: "WIN-L01"}) { oee } }',
     "filter_too_many"),
    ('query { kpis(timeRange: {start: "2026-10-31", end: "2026-10-01"}) { oee } }', "time_order"),
    ('query { kpis(timeRange: {start: "2026-02-30", end: "2026-03-01"}) { oee } }', "time_invalid"),
    ("query { kpis(shift: 4) { oee } }", "shift_range"),
    ("query { kpis(limit: 0) { oee } }", "limit_range"),
    ("query { kpis(orderBy: {metric: OEE, direction: DESC}) { quality } }", "order_not_selected"),
    ("query { kpis(aggregateBy: PLANT) { oee } }", "group_not_selected"),
    ("query { kpis { date line } }", "no_metric"),
])
def test_checker_catches(query, code):
    assert code in {i.code for i in check(query)}


def test_multi_metric_selection_is_not_representable():
    with pytest.raises(InvalidQuery) as exc:
        to_slots("query { kpis(aggregateBy: PLANT) { plant oee quality } }")
    assert exc.value.issues[0].code == "multi_metric"


def _chains():
    keys = list(hierarchy.NODES)

    @st.composite
    def chain(draw):
        leaf = draw(st.sampled_from(keys))
        path = hierarchy.ancestors(leaf)
        chosen = draw(st.lists(st.sampled_from(path), min_size=1, max_size=4, unique=True))
        return tuple((hierarchy.NODES[k].level, k) for k in chosen)

    return chain()


@st.composite
def slot_values(draw):
    sub = draw(st.sampled_from(list(SPEC.subgraphs.values())))
    direction = draw(st.sampled_from([None, "ASC", "DESC"]))
    metric = draw(st.sampled_from(sub.metrics)) if direction else draw(st.sampled_from([*sub.metrics, None]))
    start = draw(st.dates(dt.date(2024, 1, 1), dt.date(2026, 11, 18)))
    days = draw(st.integers(0, 120))
    return Slots(
        subgraph=sub.name, metric=metric, aggregate_by=draw(st.sampled_from([None, *LEVELS])), direction=direction,
        filters=draw(st.one_of(st.just(()), _chains())),
        time_range=draw(st.sampled_from([None, (start.isoformat(), (start + dt.timedelta(days)).isoformat())])),
        shift=draw(st.sampled_from([None, 1, 2, 3])), limit=draw(st.sampled_from([None, 1, 5, 10, 1000])),
    )


@settings(max_examples=400, deadline=None)
@given(slot_values())
def test_property_every_legal_slot_assignment_builds_a_valid_query(slots):
    query = to_graphql(slots)
    assert check(query) == []
    assert to_slots(query) == slots
