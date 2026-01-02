"""A small discrete Bayesian network with exact inference by enumeration, in numpy.

The networks here have a handful of latent nodes with a few dozen states between them, so the full
joint over the latent nodes fits in a small array (thousands of cells) and exact inference is one
broadcasted sum of log-factors: no approximation, and well under a millisecond per query.

Conditional probability tables (CPTs) are learned by counting with additive smoothing. Structural
zeros stay zero: a ``mask`` marks which cells are allowed at all, so a hard constraint derived from
the schema ("``spindle_load`` is not a KPI") cannot be smoothed away.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

NEG_INF = -np.inf


@dataclass
class Node:
    name: str
    states: tuple[str, ...]
    parents: tuple[str, ...] = ()
    mask: np.ndarray | None = None      # bool, shape (*parent_sizes, n_states); None = all allowed
    cpt: np.ndarray | None = None       # probabilities, same shape
    counts: np.ndarray | None = field(default=None, repr=False)

    def index(self, state: str) -> int:
        try:
            return self.states.index(state)
        except ValueError:
            raise ValueError(f"{self.name}: unknown state {state!r}") from None


class BayesNet:
    def __init__(self, nodes: list[Node], latent: list[str]) -> None:
        self.nodes = {n.name: n for n in nodes}
        self.order = [n.name for n in nodes]
        self.latent = list(latent)
        for n in nodes:
            for p in n.parents:
                if p not in self.nodes:
                    raise ValueError(f"{n.name}: unknown parent {p}")
            if n.mask is None:
                n.mask = np.ones(self._shape(n), dtype=bool)
            if n.mask.shape != self._shape(n):
                raise ValueError(f"{n.name}: mask shape {n.mask.shape} != {self._shape(n)}")

    def _shape(self, n: Node) -> tuple[int, ...]:
        return tuple(len(self.nodes[p].states) for p in n.parents) + (len(n.states),)

    # --- learning ---------------------------------------------------------------------------------

    def fit(self, rows: list[dict[str, str | None]], alpha: float = 0.5) -> BayesNet:
        """Maximum a posteriori CPTs from fully or partly observed rows (``None`` = unobserved).

        A row updates a node's table only when the node and all its parents are observed in it.
        """
        for n in self.nodes.values():
            n.counts = np.zeros(self._shape(n))
        for row in rows:
            for n in self.nodes.values():
                values = [row.get(p) for p in n.parents] + [row.get(n.name)]
                if any(v is None for v in values):
                    continue
                idx = tuple(self.nodes[v_name].index(v) for v_name, v in zip((*n.parents, n.name), values, strict=True))
                n.counts[idx] += 1
        for n in self.nodes.values():
            smoothed = np.where(n.mask, n.counts + alpha, 0.0)
            totals = smoothed.sum(axis=-1, keepdims=True)
            with np.errstate(invalid="ignore", divide="ignore"):
                n.cpt = np.where(totals > 0, smoothed / np.where(totals > 0, totals, 1), 0.0)
        return self

    # --- inference --------------------------------------------------------------------------------

    def _broadcast(self, table: np.ndarray, axes: list[str]) -> np.ndarray:
        """Reshape a table over ``axes`` (latent names) to broadcast against the full latent joint."""
        shape = [1] * len(self.latent)
        order = sorted(range(len(axes)), key=lambda i: self.latent.index(axes[i]))
        table = np.transpose(table, order)
        for i in order:
            shape[self.latent.index(axes[i])] = table.shape[order.index(i)]
        return table.reshape(shape)

    def log_joint(self, evidence: dict[str, str], soft: dict[str, np.ndarray] | None = None) -> np.ndarray:
        """log P(latent, evidence) over every latent assignment, as an array with one axis per latent node.

        ``soft`` adds likelihood vectors on latent nodes (virtual evidence), in log space.
        """
        sizes = [len(self.nodes[name].states) for name in self.latent]
        total = np.zeros(sizes)
        with np.errstate(divide="ignore"):
            for n in self.nodes.values():
                if n.name not in self.latent and n.name not in evidence:
                    continue  # unobserved leaf: sums to one, contributes nothing
                table = np.log(n.cpt)
                axes = list(n.parents) + [n.name]
                # slice observed variables out of the table
                for i in reversed(range(len(axes))):
                    name = axes[i]
                    if name not in self.latent:
                        if name not in evidence:
                            raise ValueError(f"{n.name}: parent {name} must be observed or latent")
                        table = np.take(table, self.nodes[name].index(evidence[name]), axis=i)
                        axes.pop(i)
                total = total + self._broadcast(table, axes)
            for name, vector in (soft or {}).items():
                total = total + self._broadcast(np.asarray(vector, dtype=float), [name])
        return total

    def posterior(self, evidence: dict[str, str], soft: dict[str, np.ndarray] | None = None) -> Posterior:
        lj = self.log_joint(evidence, soft)
        finite = np.isfinite(lj)
        if not finite.any():
            raise ValueError("evidence has zero probability under every latent assignment")
        top = lj[finite].max()
        p = np.where(finite, np.exp(lj - top), 0.0)
        p /= p.sum()
        return Posterior(self, p)


class Posterior:
    def __init__(self, net: BayesNet, joint: np.ndarray) -> None:
        self.net = net
        self.joint = joint

    def marginal(self, name: str) -> dict[str, float]:
        axis = self.net.latent.index(name)
        other = tuple(i for i in range(self.joint.ndim) if i != axis)
        values = self.joint.sum(axis=other)
        return dict(zip(self.net.nodes[name].states, values.tolist(), strict=True))

    def map(self) -> tuple[dict[str, str], float]:
        """The single most probable joint assignment and its posterior probability."""
        flat = int(np.argmax(self.joint))
        idx = np.unravel_index(flat, self.joint.shape)
        assignment = {name: self.net.nodes[name].states[i] for name, i in zip(self.net.latent, idx, strict=True)}
        return assignment, float(self.joint[idx])

    def map_given(self, **fixed: str) -> tuple[dict[str, str], float]:
        """Most probable assignment with some latent nodes held fixed (probability is conditional)."""
        sl = []
        for name in self.net.latent:
            sl.append(self.net.nodes[name].index(fixed[name]) if name in fixed else slice(None))
        sub = self.joint[tuple(sl)]
        free = [n for n in self.net.latent if n not in fixed]
        if sub.sum() <= 0:
            raise ValueError(f"{fixed} has zero posterior probability")
        idx = np.unravel_index(int(np.argmax(sub)), sub.shape)
        out = dict(fixed)
        out.update({name: self.net.nodes[name].states[i] for name, i in zip(free, idx, strict=True)})
        return out, float(sub[idx] / sub.sum())
