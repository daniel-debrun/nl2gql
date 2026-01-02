"""Load the subgraph SDL modules into one schema and derive what the rest of the engine needs from it.

Nothing downstream hard-codes a subgraph, metric or argument name: the builder, the checker and the
Bayesian network's hard constraints all read them from ``SPEC``.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from importlib import resources

from graphql import GraphQLEnumType, GraphQLInputObjectType, GraphQLList, GraphQLNonNull, GraphQLSchema, build_schema

SUBGRAPH_FILES = ("shared.graphql", "kpis.graphql", "cnc_data.graphql", "press_data.graphql")
ROOT_ARGUMENTS = ("filter", "aggregateBy", "timeRange", "shift", "orderBy", "limit")
RECORD_KEY_FIELDS = ("group", "region", "division", "plant", "subarea", "area", "line", "date", "shift")

# Which direction is "better". Not expressible in SDL, so it lives here, next to the schema it annotates.
# Neutral metrics have no better direction: "best spindle load" means nothing.
POLARITY = {
    "oee": "higher", "availability": "higher", "performance": "higher", "quality": "higher",
    "scrap_rate": "lower", "downtime_minutes": "lower",
    "spindle_load": "neutral", "spindle_speed": "neutral", "feed_rate": "neutral",
    "tool_wear": "lower", "cycle_time": "lower", "alarm_count": "lower",
    "tonnage": "neutral", "stroke_rate": "higher", "die_temperature": "neutral", "reject_count": "lower",
}


@dataclass(frozen=True)
class SubgraphSpec:
    name: str                      # root field, e.g. "kpis"
    record_type: str               # e.g. "KpiRecord"
    order_type: str                # e.g. "KpiOrder"
    metric_enum: str               # e.g. "KpiMetric"
    metrics: tuple[str, ...]       # record field names, e.g. ("oee", ...)
    arguments: tuple[str, ...]     # root field argument names


@dataclass(frozen=True)
class SchemaSpec:
    schema: GraphQLSchema
    subgraphs: dict[str, SubgraphSpec]
    levels: tuple[str, ...]        # Level enum values, e.g. ("GROUP", ..., "LINE")
    directions: tuple[str, ...]

    def subgraph(self, name: str) -> SubgraphSpec:
        try:
            return self.subgraphs[name]
        except KeyError:
            raise KeyError(f"unknown subgraph {name!r}; expected one of {sorted(self.subgraphs)}") from None

    @property
    def metric_slots(self) -> tuple[tuple[str, str], ...]:
        """Every (subgraph, metric) pair. ``cycle_time`` appears twice, once per machine subgraph."""
        return tuple((s.name, m) for s in self.subgraphs.values() for m in s.metrics)


def sdl() -> str:
    folder = resources.files("nl2gql") / "schema"
    return "\n".join((folder / name).read_text("utf-8") for name in SUBGRAPH_FILES)


def _unwrap(t):
    while isinstance(t, (GraphQLNonNull, GraphQLList)):
        t = t.of_type
    return t


@cache
def load() -> SchemaSpec:
    schema = build_schema(sdl())
    subgraphs = {}
    for name, field in schema.query_type.fields.items():
        if "orderBy" not in field.args:
            continue  # schemaVersion and other non-data fields
        order = _unwrap(field.args["orderBy"].type)
        assert isinstance(order, GraphQLInputObjectType)
        metric_enum = _unwrap(order.fields["metric"].type)
        assert isinstance(metric_enum, GraphQLEnumType)
        record = _unwrap(field.type)
        metrics = tuple(v.lower() for v in metric_enum.values)
        missing = [m for m in metrics if m not in record.fields]
        if missing:
            raise ValueError(f"{record.name} lacks fields for metric enum values {missing}")
        subgraphs[name] = SubgraphSpec(name, record.name, order.name, metric_enum.name, metrics, tuple(field.args))
    level = schema.get_type("Level")
    direction = schema.get_type("Direction")
    return SchemaSpec(schema, subgraphs, tuple(level.values), tuple(direction.values))


SPEC = load()
