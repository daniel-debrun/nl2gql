"""The structured meaning of a request: what the translator predicts and the builder turns into GraphQL."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from nl2gql.hierarchy import LEVEL_INDEX, LEVELS

QUERY_TYPES = ("raw", "aggregate", "ranked")


@dataclass(frozen=True)
class Slots:
    """One GraphQL query, as slot values.

    ``query_type`` is determined by the other slots: ``ranked`` has a direction, ``aggregate`` has an
    aggregation level but no direction, ``raw`` has neither. ``metric`` is a record field name, or
    ``None`` for "all metrics of the subgraph" (not allowed for ranked queries, which order by it).
    ``filters`` holds (level, key) pairs ordered from coarse to fine.
    """

    subgraph: str
    metric: str | None = None
    aggregate_by: str | None = None
    direction: str | None = None
    filters: tuple[tuple[str, str], ...] = ()
    time_range: tuple[str, str] | None = None
    shift: int | None = None
    limit: int | None = None

    def __post_init__(self) -> None:
        ordered = tuple(sorted(self.filters, key=lambda f: LEVEL_INDEX[f[0]]))
        object.__setattr__(self, "filters", ordered)
        if self.aggregate_by is not None and self.aggregate_by not in LEVELS:
            raise ValueError(f"unknown aggregation level {self.aggregate_by!r}")
        if self.direction is not None and self.direction not in ("ASC", "DESC"):
            raise ValueError(f"direction must be ASC or DESC, not {self.direction!r}")
        if self.direction is not None and self.metric is None:
            raise ValueError("a ranked query needs a metric to order by")

    @property
    def query_type(self) -> str:
        if self.direction is not None:
            return "ranked"
        return "aggregate" if self.aggregate_by is not None else "raw"

    @property
    def filter_dict(self) -> dict[str, str]:
        return dict(self.filters)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["filters"] = dict(self.filters)
        d["time_range"] = list(self.time_range) if self.time_range else None
        d["query_type"] = self.query_type
        return d

    @classmethod
    def from_dict(cls, d: dict) -> Slots:
        tr = d.get("time_range")
        return cls(
            subgraph=d["subgraph"], metric=d.get("metric"), aggregate_by=d.get("aggregate_by"),
            direction=d.get("direction"), filters=tuple((d.get("filters") or {}).items()),
            time_range=tuple(tr) if tr else None, shift=d.get("shift"), limit=d.get("limit"),
        )
