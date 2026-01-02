"""The full pipeline: utterance -> evidence -> posterior -> decision -> slots -> AST -> checked query."""

from __future__ import annotations

import datetime as dt
import time
from dataclasses import dataclass, field, replace

import numpy as np

from nl2gql.builder import BuildError, to_graphql
from nl2gql.checker import check
from nl2gql.extract import Evidence, extract
from nl2gql.model import METRICS, QTYPES, Translator
from nl2gql.slots import Slots

DEFAULT_LIMIT = 10
PROBLEM_PRIORITY = ("unknown_entity", "inconsistent_hierarchy", "multiple_values", "needs_plant")
MESSAGES = {
    "out_of_domain": "That doesn't look like a request for production data (KPIs, CNC or press data).",
    "unknown_entity": "I don't recognise {detail}. Plants are Windsor, Puebla, Toledo, Saltillo, Graz, Wroclaw, "
                      "Stuttgart, Brno, Suzhou, Penang, Chennai and Nagoya; each has lines 1-12 and areas 10-60. "
                      "Metrics are the KPIs (OEE, availability, performance, quality, scrap rate, downtime), "
                      "CNC data and press data.",
    "inconsistent_hierarchy": "These locations don't fit together: {detail}. Which one did you mean?",
    "multiple_values": "I can filter on one value per level, but found {detail}. Which one?",
    "needs_plant": "Every plant has that {detail}. Which plant?",
    "ambiguous_metric": "Cycle time exists for CNC machines and for presses. Which one?",
    "missing_metric": "Which metric? For example OEE, scrap rate, downtime, spindle load or stroke rate.",
    "invalid": "I could not build a valid query for that: {detail}",
}


@dataclass
class Result:
    kind: str                                   # query | clarify | reject
    utterance: str
    graphql: str | None = None
    slots: Slots | None = None
    reason: str | None = None
    message: str | None = None
    confidence: dict[str, float] = field(default_factory=dict)
    assumptions: list[str] = field(default_factory=list)
    defaulted: list[str] = field(default_factory=list)   # slots filled with a standard value, not recognised
    millis: float = 0.0

    def to_dict(self) -> dict:
        return {
            "kind": self.kind, "utterance": self.utterance, "graphql": self.graphql,
            "slots": self.slots.to_dict() if self.slots else None, "reason": self.reason,
            "message": self.message, "confidence": {k: round(v, 4) for k, v in self.confidence.items()},
            "assumptions": self.assumptions, "defaulted": self.defaulted, "millis": round(self.millis, 3),
        }


def _clarify(utterance: str, reason: str, detail: str = "", **extra) -> Result:
    kind = "reject" if reason == "out_of_domain" else "clarify"
    return Result(kind, utterance, reason=reason, message=MESSAGES[reason].format(detail=detail), **extra)


def _restricted_map(post, exclude_q: str = "ood", **fixed: str) -> tuple[dict[str, str], float]:
    """Joint MAP over assignments whose query type is not ``exclude_q``, optionally with some nodes fixed.

    The probability is conditional on the restriction.
    """
    joint = post.joint.copy()
    net = post.net
    index = [slice(None)] * joint.ndim
    index[net.latent.index("Q")] = QTYPES.index(exclude_q)
    joint[tuple(index)] = 0.0
    for name, state in fixed.items():
        mask = np.zeros(joint.shape[net.latent.index(name)], dtype=bool)
        mask[net.nodes[name].index(state)] = True
        shape = [1] * joint.ndim
        shape[net.latent.index(name)] = -1
        joint = joint * mask.reshape(shape)
    total = joint.sum()
    idx = np.unravel_index(int(np.argmax(joint)), joint.shape)
    assignment = {name: net.nodes[name].states[i] for name, i in zip(net.latent, idx, strict=True)}
    return assignment, float(joint[idx] / total) if total > 0 else 0.0


def slots_from(assignment: dict[str, str], ev: Evidence) -> tuple[Slots, list[str]]:
    subgraph = assignment["S"]
    metric = None if assignment["M"] == "none" else assignment["M"].split(".", 1)[1]
    qtype = assignment["Q"]
    notes = []
    limit = ev.count
    if qtype == "ranked" and limit is None:
        limit = 1 if ev.singular else DEFAULT_LIMIT
        notes.append(f"no number given; limit {limit} ({'singular' if ev.singular else 'default'})")
    if ev.time:
        notes.append(f"'{ev.time.phrase.strip()}' -> {ev.time.start} to {ev.time.end}")
    slots = Slots(
        subgraph=subgraph, metric=metric,
        aggregate_by=None if assignment["A"] == "none" else assignment["A"],
        direction=None if assignment["D"] == "none" else assignment["D"],
        filters=tuple(ev.filters.items()), time_range=ev.time.iso if ev.time else None,
        shift=ev.shift, limit=limit,
    )
    return slots, notes


def translate(translator: Translator, utterance: str, today: dt.date | None = None,
              strict: bool | None = None) -> Result:
    """Translate one request. ``strict`` overrides the model's setting for this call only."""
    started = time.perf_counter()
    today = today or dt.date.today()
    ev = extract(utterance, today)
    post = translator.posterior(ev)
    cfg = translator.config if strict is None else replace(translator.config, strict=strict)
    q_marginal = post.marginal("Q")
    elapsed = lambda: (time.perf_counter() - started) * 1000  # noqa: E731

    # Naming factory locations (even wrongly) or asking for a measure at one is in-domain: those are
    # clarified before the out-of-domain test.
    for reason in PROBLEM_PRIORITY:
        details = [d for r, d in ev.problems if r == reason]
        if details:
            return _clarify(utterance, reason, "; ".join(details), millis=elapsed())
    measure_at_location = bool(ev.unknown_measure and ev.metric_lexeme is None and
                               (ev.filters or ev.time or ev.agg_level))
    if measure_at_location and cfg.strict:
        return _clarify(utterance, "unknown_entity", ev.unknown_measure, millis=elapsed())
    if q_marginal["ood"] > cfg.tau_ood and not measure_at_location:
        return _clarify(utterance, "out_of_domain", confidence={"ood": q_marginal["ood"]}, millis=elapsed())

    defaulted: list[str] = []
    notes: list[str] = []
    fixed: dict[str, str] = {}
    if ev.metric_lexeme is None and ev.subgraph_cue is None:
        # Nothing names a metric or a data source: the request asks for something the schema does not
        # have ("energy usage"), or ranks without saying by what ("who's leading").
        if cfg.strict:
            if ev.unknown_measure:
                return _clarify(utterance, "unknown_entity", ev.unknown_measure, millis=elapsed())
            return _clarify(utterance, "missing_metric", millis=elapsed())
        fixed["S"] = cfg.default_subgraph
        defaulted.append("subgraph")
        notes.append(f"no data source recognised; used the standard one, {cfg.default_subgraph}")
        if ev.unknown_measure:
            notes.append(f"'{ev.unknown_measure}' is not a metric in the schema")

    assignment, joint_p = _restricted_map(post, **fixed)
    wants_metric = assignment["Q"] == "ranked" or bool(ev.unknown_measure)
    if "S" in fixed and not wants_metric and assignment["M"] != "none":
        defaulted.append("metric")
        notes.append(f"no metric named; inferred {assignment['M'].split('.', 1)[1]} from the wording")
    if "S" in fixed and wants_metric and ev.metric_lexeme is None:
        metric = cfg.default_metrics[fixed["S"]]
        fixed["M"] = f"{fixed['S']}.{metric}"
        assignment, joint_p = _restricted_map(post, **fixed)
        defaulted.append("metric")
        notes.append(f"no metric recognised; used the standard one, {metric}")

    m_marginal = post.marginal("M")
    if assignment["M"].endswith(".cycle_time") and "M" not in fixed:
        cnc, press = m_marginal["cnc_data.cycle_time"], m_marginal["press_data.cycle_time"]
        if min(cnc, press) / max(cnc + press, 1e-12) >= cfg.tau_ambiguous:
            if cfg.strict:
                return _clarify(utterance, "ambiguous_metric", confidence={"cnc_data": cnc, "press_data": press},
                                millis=elapsed())
            share = max(cnc, press) / max(cnc + press, 1e-12)
            defaulted.append("subgraph")
            notes.append(f"cycle time could be CNC or press; chose {assignment['S']} "
                         f"({share:.0%} of the network's belief)")
    if (assignment["Q"] == "ranked" and ev.metric_lexeme is None and "M" not in fixed
            and (not cfg.strict or m_marginal[assignment["M"]] < cfg.tau_missing)):
        if cfg.strict:
            return _clarify(utterance, "missing_metric", confidence={"metric": m_marginal[assignment["M"]]},
                            millis=elapsed())
        metric = cfg.default_metrics[assignment["S"]]
        fixed.update(S=assignment["S"], M=f"{assignment['S']}.{metric}")
        assignment, joint_p = _restricted_map(post, **fixed)
        defaulted.append("metric")
        notes.append(f"no metric recognised; used the standard one for {assignment['S']}, {metric}")

    slots, slot_notes = slots_from(assignment, ev)
    notes += slot_notes
    if slots.query_type == "ranked" and ev.count is None:
        defaulted.append("limit")
    confidence = {"joint": joint_p}
    for node in ("S", "Q", "M", "A", "D"):
        confidence[node] = post.marginal(node)[assignment[node]]
    try:
        query = to_graphql(slots)
    except BuildError as exc:
        return _clarify(utterance, "invalid", str(exc), millis=elapsed())
    issues = check(query)
    if issues:
        return _clarify(utterance, "invalid", "; ".join(map(str, issues)), millis=elapsed())
    return Result("query", utterance, graphql=query, slots=slots, confidence=confidence, assumptions=notes,
                  defaulted=defaulted, millis=elapsed())


__all__ = ["METRICS", "Result", "translate"]
