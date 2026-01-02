"""Two baselines for the ablation, both using the same extractor and the same decision rules.

- ``IndependentSlots``: one naive-Bayes classifier per slot over the same evidence (all evidence nodes
  and the same text soft evidence), trained on the same rows, each slot decided on its own. No joint
  inference, no schema constraints between slots. This is what the Bayesian network is measured against.
- ``Rules``: hand-written mapping from evidence to slots, no learning. The "templates only" baseline.

Both return ``translate.Result`` objects. When their slots cannot be built into a valid query (a CNC
metric under ``kpis``, a ranked query with no direction) the result keeps the slots, has no query text,
and counts as an invalid emitted query.
"""

from __future__ import annotations

import datetime as dt
import time

import numpy as np

from nl2gql.builder import BuildError, to_graphql
from nl2gql.checker import check
from nl2gql.extract import Evidence, extract
from nl2gql.model import (
    DIRECTIONS,
    LEVEL_STATES,
    METRICS,
    QTYPES,
    SUBGRAPHS,
    Config,
    Example,
    build_network,
    labels,
    observations,
)
from nl2gql.nb import NaiveBayes, tokens
from nl2gql.schema import POLARITY, SPEC
from nl2gql.slots import Slots
from nl2gql.translate import DEFAULT_LIMIT, PROBLEM_PRIORITY, Result, _clarify

SLOT_NODES = {"S": SUBGRAPHS, "Q": QTYPES, "M": METRICS, "A": LEVEL_STATES, "D": DIRECTIONS}
EVIDENCE_NODES = ("F", "e_met", "e_sub", "e_loc", "e_rank", "e_agg", "e_form", "e_raw", "e_num", "e_sing")


def _assemble(values: dict[str, str], ev: Evidence) -> Slots | None:
    metric = None if values["M"] == "none" else values["M"].split(".", 1)[1]
    direction = None if values["D"] == "none" else values["D"]
    if direction and metric is None:
        return None
    limit = ev.count
    if values["Q"] == "ranked" and limit is None:
        limit = 1 if ev.singular else DEFAULT_LIMIT
    return Slots(subgraph=values["S"], metric=metric, aggregate_by=None if values["A"] == "none" else values["A"],
                 direction=direction, filters=tuple(ev.filters.items()),
                 time_range=ev.time.iso if ev.time else None, shift=ev.shift, limit=limit)


def _finish(utterance: str, slots: Slots | None, started: float) -> Result:
    ms = (time.perf_counter() - started) * 1000
    if slots is None:
        return Result("query", utterance, reason="unbuildable", millis=ms)
    try:
        query = to_graphql(slots)
    except BuildError:
        return Result("query", utterance, slots=slots, reason="unbuildable", millis=ms)
    if check(query):
        return Result("query", utterance, slots=slots, reason="invalid", millis=ms)
    return Result("query", utterance, graphql=query, slots=slots, millis=ms)


class IndependentSlots:
    def __init__(self, tables, nbs, config: Config) -> None:
        self.tables = tables      # slot -> (log prior, {evidence node: log P(e | slot) array [slot, e]})
        self.nbs = nbs
        self.config = config
        self.states = {name: build_network().nodes[name].states for name in EVIDENCE_NODES}

    @classmethod
    def train(cls, examples: list[Example], today: dt.date, config: Config | None = None) -> IndependentSlots:
        config = config or Config()
        rows, docs = [], {k: ([], []) for k in SLOT_NODES}
        for ex in examples:
            if ex.kind == "clarify":
                continue
            ev = extract(ex.utterance, today)
            row = observations(ev) | labels(ex.slots, ex.kind)
            rows.append(row)
            for k in SLOT_NODES:
                if row[k] is not None:
                    docs[k][0].append(tokens(ev.masked))
                    docs[k][1].append(row[k])
        net = build_network()
        tables = {}
        for k, states in SLOT_NODES.items():
            prior = np.full(len(states), config.alpha)
            cond = {e: np.full((len(states), len(net.nodes[e].states)), config.alpha) for e in EVIDENCE_NODES}
            for row in rows:
                if row[k] is None:
                    continue
                i = states.index(row[k])
                prior[i] += 1
                for e in EVIDENCE_NODES:
                    cond[e][i, net.nodes[e].states.index(row[e])] += 1
            tables[k] = (np.log(prior / prior.sum()),
                         {e: np.log(c / c.sum(axis=1, keepdims=True)) for e, c in cond.items()})
        nbs = {k: NaiveBayes(states, alpha=config.nb_alpha).fit(*docs[k]) for k, states in SLOT_NODES.items()}
        return cls(tables, nbs, config)

    def predict(self, utterance: str, today: dt.date) -> Result:
        started = time.perf_counter()
        ev = extract(utterance, today)
        obs = observations(ev)
        toks = tokens(ev.masked)
        values, probs = {}, {}
        for k, states in SLOT_NODES.items():
            prior, cond = self.tables[k]
            score = prior.copy()
            for e in EVIDENCE_NODES:
                score += cond[e][:, self.states[e].index(obs[e])]
            if self.config.use_nb:
                ll = self.nbs[k].log_likelihood(toks)
                score += self.config.nb_weight * (ll - ll.max())
            p = np.exp(score - score.max())
            p /= p.sum()
            values[k], probs[k] = states[int(np.argmax(p))], p
        for reason in PROBLEM_PRIORITY:
            details = [d for r, d in ev.problems if r == reason]
            if details:
                return _clarify(utterance, reason, "; ".join(details))
        if ev.unknown_measure and ev.metric_lexeme is None and (ev.filters or ev.time or ev.agg_level):
            return _clarify(utterance, "unknown_entity", ev.unknown_measure)
        if probs["Q"][QTYPES.index("ood")] > self.config.tau_ood:
            return _clarify(utterance, "out_of_domain", millis=(time.perf_counter() - started) * 1000)
        if ev.metric_lexeme is None and ev.subgraph_cue is None:
            return _clarify(utterance, "unknown_entity" if ev.unknown_measure else "missing_metric")
        if values["Q"] == "ood":
            q = probs["Q"].copy()
            q[QTYPES.index("ood")] = 0
            values["Q"] = QTYPES[int(np.argmax(q))]
        return _finish(utterance, _assemble(values, ev), started)


class Rules:
    """Hand-written evidence -> slots mapping. No learning, no probabilities."""

    def predict(self, utterance: str, today: dt.date) -> Result:
        started = time.perf_counter()
        ev = extract(utterance, today)
        for reason in PROBLEM_PRIORITY:
            details = [d for r, d in ev.problems if r == reason]
            if details:
                return _clarify(utterance, reason, "; ".join(details))
        if ev.unknown_measure and ev.metric_lexeme is None and (ev.filters or ev.time or ev.agg_level):
            return _clarify(utterance, "unknown_entity", ev.unknown_measure)
        if ev.metric_lexeme is None and ev.subgraph_cue is None:
            if ev.domain_signals == 0 or not (ev.filters or ev.agg_level or ev.rank_cues):
                return _clarify(utterance, "out_of_domain")
            return _clarify(utterance, "missing_metric")
        lex = ev.metric_lexeme
        owners = [s for s, spec in SPEC.subgraphs.items() if lex in spec.metrics] if lex else []
        cue = {"kpi": "kpis", "cnc": "cnc_data", "press": "press_data"}.get(ev.subgraph_cue or "")
        loc = {"machining": "cnc_data", "press_shop": "press_data"}.get(ev.location_kind or "")
        if len(owners) == 1:
            subgraph = owners[0]
        elif len(owners) > 1:
            subgraph = cue if cue in owners else loc if loc in owners else None
            if subgraph is None:
                return _clarify(utterance, "ambiguous_metric")
        else:
            subgraph = cue or "kpis"
        rank = ev.rank_cue
        if rank and lex is None:
            return _clarify(utterance, "missing_metric")
        if rank:
            polarity = POLARITY[lex]
            direction = {"top": "DESC", "rank": "DESC", "bottom": "ASC",
                         "best": "ASC" if polarity == "lower" else "DESC",
                         "worst": "DESC" if polarity == "lower" else "ASC"}[rank]
            values = {"Q": "ranked", "A": ev.agg_level or "none", "D": direction}
        elif ev.agg_form in ("by", "which", "plural") and ev.agg_level:
            values = {"Q": "aggregate", "A": ev.agg_level, "D": "none"}
        elif ev.raw_cue or not ev.filters:
            values = {"Q": "raw", "A": "none", "D": "none"}
        else:
            values = {"Q": "aggregate", "A": ev.finest_filter, "D": "none"}
        values["S"] = subgraph
        values["M"] = f"{subgraph}.{lex}" if lex and lex in SPEC.subgraph(subgraph).metrics else "none"
        return _finish(utterance, _assemble(values, ev), started)
