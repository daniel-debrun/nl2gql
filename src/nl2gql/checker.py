"""Validation checker for GraphQL query text, and the inverse of the builder (query text -> slots).

Three layers:

1. Syntax: the text parses as GraphQL.
2. Schema: ``graphql.validate`` against the composed subgraph schema, plus the expected shape (one
   query operation, one data root field, literal arguments only).
3. Semantics the SDL cannot express: hierarchy keys exist, sit at the level they are filtered on and
   lie on one path of the tree; at most 4 filter levels; valid, ordered dates; shift in 1..3; a
   positive limit; the ordered-by metric and the grouping level are selected.

Used on the engine's own output, on hand-written golden queries, and on every generated training
example before it is allowed into the training set.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

from graphql import GraphQLError, parse, validate
from graphql.language import ast

from nl2gql import hierarchy
from nl2gql.schema import SPEC
from nl2gql.slots import Slots

MAX_FILTER_LEVELS = 4
MAX_LIMIT = 1000


@dataclass(frozen=True)
class Issue:
    code: str
    message: str

    def __str__(self) -> str:
        return f"{self.code}: {self.message}"


class InvalidQuery(ValueError):
    def __init__(self, issues: list[Issue]) -> None:
        super().__init__("; ".join(map(str, issues)))
        self.issues = issues


def _root_field(doc: ast.DocumentNode) -> tuple[ast.FieldNode | None, list[Issue]]:
    ops = [d for d in doc.definitions if isinstance(d, ast.OperationDefinitionNode)]
    if len(ops) != 1 or len(doc.definitions) != 1:
        return None, [Issue("shape", "expected exactly one operation and no fragments")]
    op = ops[0]
    if op.operation != ast.OperationType.QUERY:
        return None, [Issue("shape", f"expected a query, got {op.operation.value}")]
    if op.variable_definitions:
        return None, [Issue("shape", "variables are not supported; use literal arguments")]
    roots = op.selection_set.selections
    if len(roots) != 1 or not isinstance(roots[0], ast.FieldNode) or roots[0].name.value not in SPEC.subgraphs:
        return None, [Issue("shape", f"expected one root field from {sorted(SPEC.subgraphs)}")]
    root = roots[0]
    if root.alias is not None:
        return None, [Issue("shape", "aliases on the root field are not supported")]
    return root, []


def _value(node: ast.ValueNode):
    if isinstance(node, ast.ObjectValueNode):
        return {f.name.value: _value(f.value) for f in node.fields}
    if isinstance(node, ast.IntValueNode):
        return int(node.value)
    if isinstance(node, (ast.StringValueNode, ast.EnumValueNode)):
        return node.value
    if isinstance(node, ast.NullValueNode):
        return None
    if isinstance(node, ast.ListValueNode):
        return [_value(v) for v in node.values]
    return getattr(node, "value", None)


def _date(text, issues: list[Issue], label: str) -> dt.date | None:
    try:
        return dt.date.fromisoformat(text)
    except (TypeError, ValueError):
        issues.append(Issue("time_invalid", f"{label} {text!r} is not a YYYY-MM-DD date"))
        return None


def _semantic(root: ast.FieldNode) -> list[Issue]:
    issues: list[Issue] = []
    args = {a.name.value: _value(a.value) for a in root.arguments}
    selected = {s.name.value for s in root.selection_set.selections if isinstance(s, ast.FieldNode)}
    filters = args.get("filter") or {}
    filters = {k: v for k, v in filters.items() if v is not None}
    if len(filters) > MAX_FILTER_LEVELS:
        issues.append(Issue("filter_too_many", f"{len(filters)} filter levels; at most {MAX_FILTER_LEVELS}"))
    known = []
    for level, key in filters.items():
        n = hierarchy.NODES.get(key)
        if n is None:
            issues.append(Issue("filter_key_unknown", f"{level}: {key!r} is not a hierarchy key"))
        elif n.level != level:
            issues.append(Issue("filter_level_mismatch", f"{key!r} is a {n.level}, filtered as {level}"))
        else:
            known.append(key)
    if len(known) > 1 and not hierarchy.on_one_path(known):
        issues.append(Issue("filter_inconsistent", f"filters {sorted(known)} are not on one hierarchy path"))
    if "timeRange" in args:
        tr = args["timeRange"] or {}
        start, end = _date(tr.get("start"), issues, "start"), _date(tr.get("end"), issues, "end")
        if start and end and start > end:
            issues.append(Issue("time_order", f"start {start} is after end {end}"))
    if "shift" in args and args["shift"] not in (1, 2, 3):
        issues.append(Issue("shift_range", f"shift must be 1, 2 or 3, not {args['shift']!r}"))
    if "limit" in args and not (isinstance(args["limit"], int) and 0 < args["limit"] <= MAX_LIMIT):
        issues.append(Issue("limit_range", f"limit must be in 1..{MAX_LIMIT}, not {args['limit']!r}"))
    order = args.get("orderBy")
    if order and str(order.get("metric", "")).lower() not in selected:
        issues.append(Issue("order_not_selected", f"ordered by {order.get('metric')} but it is not selected"))
    agg = args.get("aggregateBy")
    if agg and agg.lower() not in selected:
        issues.append(Issue("group_not_selected", f"aggregated by {agg} but the {agg.lower()} field is not selected"))
    metrics = set(SPEC.subgraph(root.name.value).metrics)
    if not selected & metrics:
        issues.append(Issue("no_metric", "the query selects no metric"))
    return issues


def check(query: str | ast.DocumentNode) -> list[Issue]:
    """All issues with a query; an empty list means valid."""
    if isinstance(query, str):
        try:
            doc = parse(query)
        except GraphQLError as exc:
            return [Issue("syntax", exc.message)]
    else:
        doc = query
    errors = validate(SPEC.schema, doc)
    if errors:
        return [Issue("schema", e.message) for e in errors]
    root, issues = _root_field(doc)
    if root is None:
        return issues
    return _semantic(root)


def to_slots(query: str | ast.DocumentNode) -> Slots:
    """Parse a valid query back into slots. Raises ``InvalidQuery`` when the checker finds issues.

    The metric is the ordered-by metric when there is one; otherwise the single selected metric, or
    ``None`` when every metric of the subgraph is selected. Selecting some but not all of several
    metrics is outside what the slots can represent and is rejected.
    """
    doc = parse(query) if isinstance(query, str) else query
    issues = check(doc)
    if issues:
        raise InvalidQuery(issues)
    root, _ = _root_field(doc)
    args = {a.name.value: _value(a.value) for a in root.arguments}
    sub = SPEC.subgraph(root.name.value)
    selected = [s.name.value for s in root.selection_set.selections]
    chosen = [m for m in sub.metrics if m in selected]
    order = args.get("orderBy")
    if order:
        metric = order["metric"].lower()
    elif len(chosen) == 1:
        metric = chosen[0]
    elif set(chosen) == set(sub.metrics):
        metric = None
    else:
        raise InvalidQuery([Issue("multi_metric", f"selects {chosen}; one metric or all of them is supported")])
    tr = args.get("timeRange")
    filters = {k: v for k, v in (args.get("filter") or {}).items() if v is not None}
    return Slots(
        subgraph=sub.name, metric=metric,
        aggregate_by=args["aggregateBy"].lower() if args.get("aggregateBy") else None,
        direction=order["direction"] if order else None,
        filters=tuple(filters.items()),
        time_range=(tr["start"], tr["end"]) if tr else None,
        shift=args.get("shift"), limit=args.get("limit"),
    )


def main(argv: list[str] | None = None) -> int:
    """``python -m nl2gql.checker FILE.jsonl``: check the ``graphql`` field of every JSONL line."""
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("files", nargs="+")
    args = parser.parse_args(argv)
    bad = total = 0
    for path in args.files:
        with open(path, encoding="utf-8") as fh:
            for n, line in enumerate(fh, 1):
                if not line.strip():
                    continue
                total += 1
                try:
                    ex = json.loads(line)
                except json.JSONDecodeError as exc:
                    bad += 1
                    print(f"{path}:{n}: invalid JSON: {exc}")
                    continue
                if ex.get("kind", "query") != "query":
                    continue
                try:
                    to_slots(ex["graphql"])
                except InvalidQuery as exc:
                    bad += 1
                    print(f"{path}:{n}: {exc}")
                except (KeyError, TypeError) as exc:
                    bad += 1
                    print(f"{path}:{n}: missing or malformed field: {exc}")
    print(f"{total - bad}/{total} lines valid", file=sys.stderr)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
