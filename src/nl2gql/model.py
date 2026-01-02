"""The translation network: structure from the schema, parameters from data, soft evidence from text.

Latent nodes (what the query means):

    S  subgraph           kpis | cnc_data | press_data
    Q  query type         raw | aggregate | ranked | ood (out of domain)
    M  metric             "<subgraph>.<metric>" for every metric of every subgraph, or "none" (all)
    P  metric polarity    higher | lower | neutral        (deterministic given M)
    A  aggregation level  none | group ... line
    D  direction          none | ASC | DESC

Observed nodes (evidence from the extractor):

    F       finest filter level            parent of A
    e_met   metric word ("cycle_time")     child of M
    e_sub   subgraph cue ("cnc", "press")  child of S
    e_loc   kind of the filtered subarea   child of S
    e_rank  ranking cue ("best", "top")    child of Q, D, P
    e_agg   level in a grouping phrase     child of A and F (no phrase when A = F: "oee for Windsor")
    e_form  form of that phrase            child of Q
    e_raw   raw-record cue (none/weak/strong)  child of Q
    e_num, e_sing                          children of Q

Structural zeros come from the schema: M must belong to S (so ``spindle_load`` can never be a KPI),
raw queries have no aggregation or direction, aggregates have an aggregation level, ranked queries
have a direction and a metric, out-of-domain requests have none of them. Word evidence that the
extractor's lexicon does not cover enters as naive-Bayes soft evidence on S, Q and M.
"""

from __future__ import annotations

import datetime as dt
import io
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from nl2gql import lexicon
from nl2gql.bn import BayesNet, Node, Posterior
from nl2gql.extract import Evidence, extract
from nl2gql.hierarchy import LEVELS
from nl2gql.nb import NaiveBayes, tokens
from nl2gql.schema import POLARITY, SPEC
from nl2gql.slots import Slots

SUBGRAPHS = tuple(SPEC.subgraphs)
METRICS = tuple(f"{s}.{m}" for s, m in SPEC.metric_slots) + ("none",)
QTYPES = ("raw", "aggregate", "ranked", "ood")
LEVEL_STATES = ("none",) + LEVELS
DIRECTIONS = ("none", "ASC", "DESC")
POLARITIES = ("higher", "lower", "neutral")
LEXEMES = ("none",) + tuple(lexicon.METRIC_PHRASES)
SUB_CUES = ("none", "kpi", "cnc", "press")
LOCATIONS = ("none", "press_shop", "machining", "assembly")
RANK_CUES = ("none", "top", "bottom", "best", "worst", "rank")
FORMS = ("none", "by", "which", "plural")
YESNO = ("no", "yes")
RAW_LEVELS = ("no", "weak", "strong")
LATENT = ["S", "Q", "M", "P", "A", "D"]


def _mask(shape, allowed) -> np.ndarray:
    m = np.zeros(shape, dtype=bool)
    for idx in np.ndindex(*shape):
        m[idx] = allowed(*idx)
    return m


def build_network() -> BayesNet:
    """The network structure with schema-derived structural zeros; parameters come from ``fit``."""
    none_m = METRICS.index("none")

    def m_allowed(s, q, m):
        if m == none_m:
            return QTYPES[q] != "ranked"
        if QTYPES[q] == "ood":
            return False
        return METRICS[m].split(".")[0] == SUBGRAPHS[s]

    def p_allowed(m, p):
        want = "neutral" if m == none_m else POLARITY[METRICS[m].split(".")[1]]
        return POLARITIES[p] == want

    def a_allowed(q, f, a):
        kind = QTYPES[q]
        if kind in ("raw", "ood"):
            return a == 0
        if kind == "aggregate":
            return a != 0
        return True

    def d_allowed(q, d):
        return (d != 0) if QTYPES[q] == "ranked" else (d == 0)

    n = len
    nodes = [
        Node("S", SUBGRAPHS),
        Node("Q", QTYPES),
        Node("M", METRICS, ("S", "Q"), _mask((n(SUBGRAPHS), n(QTYPES), n(METRICS)), m_allowed)),
        Node("P", POLARITIES, ("M",), _mask((n(METRICS), n(POLARITIES)), p_allowed)),
        Node("F", LEVEL_STATES),
        Node("A", LEVEL_STATES, ("Q", "F"), _mask((n(QTYPES), n(LEVEL_STATES), n(LEVEL_STATES)), a_allowed)),
        Node("D", DIRECTIONS, ("Q",), _mask((n(QTYPES), n(DIRECTIONS)), d_allowed)),
        Node("e_met", LEXEMES, ("M",)),
        Node("e_sub", SUB_CUES, ("S",)),
        Node("e_loc", LOCATIONS, ("S",)),
        Node("e_rank", RANK_CUES, ("Q", "D", "P")),
        Node("e_agg", LEVEL_STATES, ("A", "F")),
        Node("e_form", FORMS, ("Q",)),
        Node("e_raw", RAW_LEVELS, ("Q",)),
        Node("e_num", YESNO, ("Q",)),
        Node("e_sing", YESNO, ("Q",)),
    ]
    return BayesNet(nodes, LATENT)


def observations(ev: Evidence) -> dict[str, str]:
    """The observed nodes for one utterance."""
    return {
        "F": ev.finest_filter or "none",
        "e_met": ev.metric_lexeme or "none",
        "e_sub": ev.subgraph_cue or "none",
        "e_loc": ev.location_kind or "none",
        "e_rank": ev.rank_cue or "none",
        "e_agg": ev.agg_level or "none",
        "e_form": ev.agg_form or "none",
        "e_raw": ev.raw_strength,
        "e_num": YESNO[ev.count is not None],
        "e_sing": YESNO[ev.singular],
    }


def labels(slots: Slots | None, kind: str) -> dict[str, str | None]:
    """The latent nodes' true values for a training example (``None`` where undefined)."""
    if kind == "reject":
        return {"S": None, "Q": "ood", "M": "none", "P": "neutral", "A": "none", "D": "none"}
    metric = f"{slots.subgraph}.{slots.metric}" if slots.metric else "none"
    return {
        "S": slots.subgraph, "Q": slots.query_type, "M": metric,
        "P": POLARITY[slots.metric] if slots.metric else "neutral",
        "A": slots.aggregate_by or "none", "D": slots.direction or "none",
    }


@dataclass
class Example:
    utterance: str
    kind: str                      # query | clarify | reject
    slots: Slots | None = None
    reason: str | None = None
    source: str = ""
    id: str = ""


@dataclass
class Config:
    alpha: float = 0.5            # CPT smoothing
    nb_alpha: float = 0.5
    nb_weight: float = 0.35       # tempering of naive-Bayes soft evidence (it double-counts shared words)
    use_nb: bool = True
    use_network: bool = True      # False: independent per-slot classifiers (ablation)
    tau_ood: float = 0.5          # reject when P(Q = ood) exceeds this
    tau_ambiguous: float = 0.25   # clarify cycle time when the runner-up subgraph has this much mass
    tau_missing: float = 0.5      # a ranked request with no metric word below this confidence has no metric
    # When a parameter is not recognised, fill in a standard value and say so (strict=False), or ask (strict=True).
    # Location problems and off-topic requests always ask or refuse: a guessed plant returns the wrong data.
    strict: bool = False
    default_subgraph: str = "kpis"
    default_metrics: dict[str, str] = field(default_factory=lambda: {
        "kpis": "oee", "cnc_data": "cycle_time", "press_data": "stroke_rate"})


class Translator:
    """Evidence -> posterior over the latent nodes."""

    def __init__(self, net: BayesNet, nbs: dict[str, NaiveBayes], config: Config) -> None:
        self.net = net
        self.nbs = nbs
        self.config = config

    @classmethod
    def train(cls, examples: list[Example], today: dt.date, config: Config | None = None) -> Translator:
        config = config or Config()
        net = build_network()
        rows, docs = [], {"S": ([], []), "Q": ([], []), "M": ([], [])}
        for ex in examples:
            if ex.kind == "clarify":
                continue
            ev = extract(ex.utterance, today)
            row = observations(ev) | labels(ex.slots, ex.kind)
            rows.append(row)
            toks = tokens(ev.masked)
            for node in ("S", "Q", "M"):
                if row[node] is not None:
                    docs[node][0].append(toks)
                    docs[node][1].append(row[node])
        net.fit(rows, alpha=config.alpha)
        nbs = {node: NaiveBayes(net.nodes[node].states, alpha=config.nb_alpha).fit(*docs[node])
               for node in ("S", "Q", "M")}
        return cls(net, nbs, config)

    def soft_evidence(self, ev: Evidence) -> dict[str, np.ndarray]:
        if not self.config.use_nb:
            return {}
        toks = tokens(ev.masked)
        out = {}
        for node, nb in self.nbs.items():
            ll = nb.log_likelihood(toks)
            out[node] = self.config.nb_weight * (ll - ll.max())
        return out

    def posterior(self, ev: Evidence) -> Posterior:
        return self.net.posterior(observations(ev), self.soft_evidence(ev))

    # --- persistence ------------------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        arrays = {f"cpt.{name}": node.cpt for name, node in self.net.nodes.items()}
        for node, nb in self.nbs.items():
            arrays |= nb.to_arrays(f"nb.{node}")
        arrays["config"] = np.array(json.dumps(self.config.__dict__))
        buf = io.BytesIO()
        np.savez_compressed(buf, **arrays)
        Path(path).write_bytes(buf.getvalue())

    @classmethod
    def load(cls, path: str | Path) -> Translator:
        with np.load(path, allow_pickle=False) as data:
            arrays = {k: data[k] for k in data.files}
        net = build_network()
        for name, node in net.nodes.items():
            node.cpt = arrays[f"cpt.{name}"]
        nbs = {node: NaiveBayes.from_arrays(arrays, f"nb.{node}") for node in ("S", "Q", "M")}
        return cls(net, nbs, Config(**json.loads(str(arrays["config"]))))
