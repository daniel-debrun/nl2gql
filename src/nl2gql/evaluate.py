"""Load annotated JSONL, and score a translator against it.

Gold queries are compared as slots, not as text: argument order, formatting and the choice of extra
selected fields do not matter; subgraph, metric, grouping, direction, filters, dates, shift and limit
all must.
"""

from __future__ import annotations

import datetime as dt
import json
import statistics
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from nl2gql.checker import InvalidQuery, check, to_slots
from nl2gql.model import Example

SLOT_FIELDS = ("subgraph", "metric", "query_type", "aggregate_by", "direction", "filters", "time_range",
               "shift", "limit")


@dataclass
class LoadReport:
    path: str
    total: int = 0
    kinds: Counter = field(default_factory=Counter)
    invalid: list[tuple[str, str]] = field(default_factory=list)   # (id, issue)


def load_jsonl(path: str | Path, source: str | None = None) -> tuple[list[Example], LoadReport]:
    report = LoadReport(str(path))
    examples = []
    for n, line in enumerate(Path(path).read_text("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        report.total += 1
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            report.invalid.append((f"line {n}", f"json: {exc}"))
            continue
        kind = raw.get("kind", "query")
        ex_id = raw.get("id", f"line {n}")
        if kind == "query":
            try:
                slots = to_slots(raw["graphql"])
            except InvalidQuery as exc:
                report.invalid.append((ex_id, str(exc)))
                continue
            except (KeyError, TypeError) as exc:
                report.invalid.append((ex_id, f"missing graphql: {exc}"))
                continue
            examples.append(Example(raw["utterance"], "query", slots, source=source or Path(path).stem, id=ex_id))
        elif kind in ("clarify", "reject"):
            examples.append(Example(raw["utterance"], kind, reason=raw.get("reason"),
                                    source=source or Path(path).stem, id=ex_id))
        else:
            report.invalid.append((ex_id, f"unknown kind {kind!r}"))
            continue
        report.kinds[kind] += 1
    return examples, report


def _slot_values(slots) -> dict:
    return {
        "subgraph": slots.subgraph, "metric": slots.metric, "query_type": slots.query_type,
        "aggregate_by": slots.aggregate_by, "direction": slots.direction, "filters": slots.filters,
        "time_range": slots.time_range, "shift": slots.shift, "limit": slots.limit,
    }


@dataclass
class Score:
    n: int = 0
    kind_correct: int = 0
    reason_correct: int = 0
    reason_total: int = 0
    queries_gold: int = 0
    queries_emitted: int = 0
    queries_valid: int = 0
    exact: int = 0
    slot_correct: Counter = field(default_factory=Counter)
    kind_confusion: Counter = field(default_factory=Counter)
    latencies: list[float] = field(default_factory=list)
    errors: list[dict] = field(default_factory=list)

    def summary(self) -> dict:
        q = max(self.queries_gold, 1)
        lat = sorted(self.latencies) or [0.0]
        return {
            "examples": self.n,
            "decision_accuracy": round(self.kind_correct / max(self.n, 1), 4),
            "clarify_reason_accuracy": round(self.reason_correct / max(self.reason_total, 1), 4),
            "gold_queries": self.queries_gold,
            "exact_match": round(self.exact / q, 4),
            "slot_accuracy": {f: round(self.slot_correct[f] / q, 4) for f in SLOT_FIELDS},
            "validity_of_emitted_queries": round(self.queries_valid / max(self.queries_emitted, 1), 4),
            "emitted_queries": self.queries_emitted,
            "latency_ms_median": round(statistics.median(lat), 3),
            "latency_ms_p95": round(lat[int(0.95 * (len(lat) - 1))], 3),
            "kind_confusion": {f"{g}->{p}": c for (g, p), c in sorted(self.kind_confusion.items())},
        }


def score(predict, examples: list[Example], today: dt.date, keep_errors: int = 200) -> Score:
    """``predict(utterance, today)`` returns an object with ``kind``, ``slots``, ``reason``, ``graphql``, ``millis``."""
    s = Score()
    for ex in examples:
        r = predict(ex.utterance, today)
        s.n += 1
        s.latencies.append(getattr(r, "millis", 0.0))
        s.kind_confusion[(ex.kind, r.kind)] += 1
        if r.kind == "query":
            s.queries_emitted += 1
            s.queries_valid += bool(r.graphql) and not check(r.graphql)
        if r.kind == ex.kind:
            s.kind_correct += 1
        if ex.kind == "clarify":
            s.reason_total += 1
            s.reason_correct += r.kind == "clarify" and r.reason == ex.reason
        if ex.kind != "query":
            if r.kind != ex.kind and len(s.errors) < keep_errors:
                s.errors.append({"id": ex.id, "utterance": ex.utterance, "gold": ex.kind, "gold_reason": ex.reason,
                                 "pred": r.kind, "pred_reason": r.reason})
            continue
        s.queries_gold += 1
        if r.kind != "query" or r.slots is None:
            if len(s.errors) < keep_errors:
                s.errors.append({"id": ex.id, "utterance": ex.utterance, "gold": "query", "pred": r.kind,
                                 "pred_reason": r.reason})
            continue
        gold, pred = _slot_values(ex.slots), _slot_values(r.slots)
        wrong = [f for f in SLOT_FIELDS if gold[f] != pred[f]]
        for f in SLOT_FIELDS:
            s.slot_correct[f] += f not in wrong
        if not wrong:
            s.exact += 1
        elif len(s.errors) < keep_errors:
            s.errors.append({"id": ex.id, "utterance": ex.utterance, "wrong": {
                f: {"gold": gold[f], "pred": pred[f]} for f in wrong}})
    return s
