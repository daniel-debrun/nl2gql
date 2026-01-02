"""Multinomial naive Bayes over the masked utterance, used as soft evidence for network nodes.

Binary token presence over unigrams and bigrams, additive smoothing, numpy only. The input is the
extractor's masked text, where times, hierarchy names, numbers and known metric words are replaced by
placeholders ("__time__", "__plant__", "__metric__"), so the model learns phrasing ("readings", "how
did", "rank") rather than memorising plant names, and does not double-count the metric words the
network already sees.
"""

from __future__ import annotations

import re

import numpy as np

_TOKEN = re.compile(r"__[a-z_]+__|[a-z0-9%]+(?:'[a-z]+)?")


def tokens(masked: str) -> list[str]:
    words = [re.sub(r"^__metric_[a-z_]+__$", "__metric__", w) for w in _TOKEN.findall(masked)]
    out = set(words)
    out.update(f"{a}|{b}" for a, b in zip(words, words[1:], strict=False))
    return sorted(out)


class NaiveBayes:
    """Multinomial NB with binary token counts: log P(doc | c) = sum over present tokens of log theta_c,t."""

    def __init__(self, classes: tuple[str, ...], alpha: float = 0.5, min_count: int = 2) -> None:
        self.classes = tuple(classes)
        self.alpha = alpha
        self.min_count = min_count
        self.vocab: dict[str, int] = {}
        self.log_theta: np.ndarray | None = None   # (n_classes, n_tokens)

    def fit(self, docs: list[list[str]], labels: list[str]) -> NaiveBayes:
        counts: dict[str, int] = {}
        for doc in docs:
            for tok in doc:
                counts[tok] = counts.get(tok, 0) + 1
        self.vocab = {t: i for i, t in enumerate(sorted(t for t, c in counts.items() if c >= self.min_count))}
        table = np.zeros((len(self.classes), len(self.vocab)))
        for doc, label in zip(docs, labels, strict=True):
            c = self.classes.index(label)
            for tok in doc:
                j = self.vocab.get(tok)
                if j is not None:
                    table[c, j] += 1.0
        smoothed = table + self.alpha
        self.log_theta = np.log(smoothed / smoothed.sum(axis=1, keepdims=True))
        return self

    def log_likelihood(self, doc: list[str]) -> np.ndarray:
        """log P(doc | class) for every class (no prior: the network supplies it)."""
        idx = [self.vocab[t] for t in doc if t in self.vocab]
        if not idx:
            return np.zeros(len(self.classes))
        return self.log_theta[:, idx].sum(axis=1)

    def to_arrays(self, prefix: str) -> dict[str, np.ndarray]:
        return {f"{prefix}.log_theta": self.log_theta,
                f"{prefix}.vocab": np.array(sorted(self.vocab, key=self.vocab.get)),
                f"{prefix}.classes": np.array(self.classes)}

    @classmethod
    def from_arrays(cls, arrays: dict[str, np.ndarray], prefix: str) -> NaiveBayes:
        nb = cls(tuple(str(c) for c in arrays[f"{prefix}.classes"]))
        nb.vocab = {str(t): i for i, t in enumerate(arrays[f"{prefix}.vocab"])}
        nb.log_theta = arrays[f"{prefix}.log_theta"]
        return nb
