"""Slots -> GraphQL AST -> query text.

The builder never assembles query text from strings. It builds AST nodes, and every argument, input
field, enum value and selected field is looked up in the loaded schema first, so there is no code
path that emits a name the schema does not define. ``print_ast`` renders the text.
"""

from __future__ import annotations

from graphql import GraphQLEnumType, GraphQLInputObjectType, print_ast
from graphql.language import ast

from nl2gql.schema import SPEC, _unwrap
from nl2gql.slots import Slots

RAW_KEY_FIELDS = ("date", "shift", "line")


class BuildError(ValueError):
    """The slots describe something the schema cannot express."""


def _name(value: str) -> ast.NameNode:
    return ast.NameNode(value=value)


def _enum(enum_type, value: str) -> ast.EnumValueNode:
    if not isinstance(enum_type, GraphQLEnumType) or value not in enum_type.values:
        raise BuildError(f"{value!r} is not a value of {enum_type}")
    return ast.EnumValueNode(value=value)


def _object(input_type, fields: dict[str, ast.ValueNode]) -> ast.ObjectValueNode:
    if not isinstance(input_type, GraphQLInputObjectType):
        raise BuildError(f"{input_type} is not an input object type")
    unknown = set(fields) - set(input_type.fields)
    if unknown:
        raise BuildError(f"{input_type.name} has no fields {sorted(unknown)}")
    return ast.ObjectValueNode(fields=tuple(
        ast.ObjectFieldNode(name=_name(k), value=v) for k, v in fields.items()))


def _string(value: str) -> ast.StringValueNode:
    return ast.StringValueNode(value=value, block=False)


def _int(value: int) -> ast.IntValueNode:
    return ast.IntValueNode(value=str(int(value)))


def arguments(slots: Slots) -> list[tuple[str, ast.ValueNode]]:
    """Argument nodes for the root field, in canonical order."""
    root = SPEC.schema.query_type.fields[slots.subgraph]
    args = root.args
    out: list[tuple[str, ast.ValueNode]] = []
    if slots.filters:
        out.append(("filter", _object(_unwrap(args["filter"].type),
                                      {level: _string(key) for level, key in slots.filters})))
    if slots.aggregate_by:
        out.append(("aggregateBy", _enum(_unwrap(args["aggregateBy"].type), slots.aggregate_by.upper())))
    if slots.time_range:
        start, end = slots.time_range
        out.append(("timeRange", _object(_unwrap(args["timeRange"].type),
                                         {"start": _string(start), "end": _string(end)})))
    if slots.shift is not None:
        out.append(("shift", _int(slots.shift)))
    if slots.direction:
        order_type = _unwrap(args["orderBy"].type)
        metric_enum = _unwrap(order_type.fields["metric"].type)
        direction_enum = _unwrap(order_type.fields["direction"].type)
        out.append(("orderBy", _object(order_type, {
            "metric": _enum(metric_enum, slots.metric.upper()),
            "direction": _enum(direction_enum, slots.direction),
        })))
    if slots.limit is not None:
        out.append(("limit", _int(slots.limit)))
    for name, _ in out:
        if name not in args:
            raise BuildError(f"{slots.subgraph} has no argument {name!r}")
    return out


def selection(slots: Slots) -> list[str]:
    """Fields to select: the grouping key (or the raw row key) followed by the metric(s)."""
    sub = SPEC.subgraph(slots.subgraph)
    metrics = [slots.metric] if slots.metric else list(sub.metrics)
    keys = [slots.aggregate_by] if slots.aggregate_by else list(RAW_KEY_FIELDS)
    return keys + metrics


def build(slots: Slots) -> ast.DocumentNode:
    try:
        SPEC.subgraph(slots.subgraph)
    except KeyError as exc:
        raise BuildError(str(exc)) from None
    if slots.metric is not None and slots.metric not in SPEC.subgraph(slots.subgraph).metrics:
        raise BuildError(f"{slots.subgraph} has no metric {slots.metric!r}")
    record = _unwrap(SPEC.schema.query_type.fields[slots.subgraph].type)
    fields = selection(slots)
    unknown = [f for f in fields if f not in record.fields]
    if unknown:
        raise BuildError(f"{record.name} has no fields {unknown}")
    root = ast.FieldNode(
        alias=None, name=_name(slots.subgraph), directives=(),
        arguments=tuple(ast.ArgumentNode(name=_name(n), value=v) for n, v in arguments(slots)),
        selection_set=ast.SelectionSetNode(selections=tuple(
            ast.FieldNode(alias=None, name=_name(f), arguments=(), directives=(), selection_set=None)
            for f in fields)),
    )
    operation = ast.OperationDefinitionNode(
        operation=ast.OperationType.QUERY, name=None, variable_definitions=(), directives=(),
        selection_set=ast.SelectionSetNode(selections=(root,)))
    return ast.DocumentNode(definitions=(operation,))


def to_graphql(slots: Slots) -> str:
    return print_ast(build(slots))
