"""Training, tuning and the ablation study.

Data:
- ``data/examples/{A,B,C,D}.jsonl``: fabricated example sets used for training, development and tuning.
- ``data/examples/E.jsonl``: the fabricated test set, written after development finished and scored once,
  at the end. Never used for training or tuning.
- synthetic examples from ``generate.Generator``.

Tuning uses leave-one-set-out over A-D: train on three plus synthetic, score on the fourth, average.
"""

from __future__ import annotations

import datetime as dt
import itertools
import json
from dataclasses import replace
from pathlib import Path

from nl2gql.baselines import IndependentSlots, Rules
from nl2gql.evaluate import load_jsonl, score
from nl2gql.extract import caching
from nl2gql.generate import Generator
from nl2gql.model import Config, Example, Translator
from nl2gql.translate import translate

REFERENCE_DATE = dt.date(2026, 11, 18)       # "today" for every annotated example
TRAIN_SETS = ("A", "B", "C", "D")
TEST_SET = "E"
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
MODEL_PATH = Path(__file__).resolve().parent / "models" / "nl2gql.npz"
SYNTHETIC_N = 6000


def load_set(name: str, data_dir: Path = DATA_DIR) -> tuple[list[Example], dict]:
    examples, report = load_jsonl(data_dir / "examples" / f"{name}.jsonl", source=name)
    return examples, {"file": f"{name}.jsonl", "lines": report.total, "kinds": dict(report.kinds),
                      "invalid": len(report.invalid), "invalid_examples": report.invalid[:10]}


def synthetic(n: int = SYNTHETIC_N, seed: int = 1) -> list[Example]:
    return Generator(REFERENCE_DATE, seed=seed).examples(n)


def bn_predictor(train: list[Example], config: Config):
    model = Translator.train(train, REFERENCE_DATE, config)
    return lambda u, today: translate(model, u, today, strict=True)   # the labels encode strict behaviour


def _objective(summary: dict) -> float:
    return 0.5 * summary["exact_match"] + 0.5 * summary["decision_accuracy"]


def tune(annotated: dict[str, list[Example]], extra: list[Example]) -> tuple[Config, list[dict]]:
    """Grid search over the soft-evidence weight and decision thresholds, leave-one-set-out."""
    grid = {
        "nb_weight": (0.0, 0.2, 0.35, 0.5, 0.8),
        "tau_ood": (0.4, 0.5, 0.7),
        "tau_ambiguous": (0.15, 0.25, 0.35),
        "tau_missing": (0.3, 0.5, 0.7),
    }
    # Thresholds only change decisions, not training, so train once per (fold, nb_weight).
    folds = []
    for held in annotated:
        train = [ex for name, exs in annotated.items() if name != held for ex in exs] + extra
        folds.append((held, train, annotated[held]))
    trials = []
    for nb_weight in grid["nb_weight"]:
        models = [(held, Translator.train(train, REFERENCE_DATE, Config(nb_weight=nb_weight, use_nb=nb_weight > 0)),
                   test) for held, train, test in folds]
        for tau_ood, tau_amb, tau_miss in itertools.product(grid["tau_ood"], grid["tau_ambiguous"], grid["tau_missing"]):
            values = []
            for _, model, test in models:
                model.config = Config(nb_weight=nb_weight, use_nb=nb_weight > 0, tau_ood=tau_ood,
                                      tau_ambiguous=tau_amb, tau_missing=tau_miss)
                values.append(_objective(score(lambda u, d, m=model: translate(m, u, d, strict=True), test, REFERENCE_DATE).summary()))
            trials.append({"nb_weight": nb_weight, "tau_ood": tau_ood, "tau_ambiguous": tau_amb,
                           "tau_missing": tau_miss, "objective": round(sum(values) / len(values), 4)})
    best = max(trials, key=lambda t: t["objective"])
    config = Config(nb_weight=best["nb_weight"], use_nb=best["nb_weight"] > 0, tau_ood=best["tau_ood"],
                    tau_ambiguous=best["tau_ambiguous"], tau_missing=best["tau_missing"])
    return config, sorted(trials, key=lambda t: -t["objective"])[:10]


def run(out_dir: Path, tune_grid: bool = True) -> dict:
    annotated, reports = {}, {}
    for name in (*TRAIN_SETS, TEST_SET):
        annotated[name], reports[name] = load_set(name)
    test = annotated.pop(TEST_SET)
    syn = synthetic()
    example_train = [ex for exs in annotated.values() for ex in exs]
    names = "+".join(TRAIN_SETS)

    with caching():
        config, top_trials = tune(annotated, syn) if tune_grid else (Config(), [])
    no_nb = replace(config, use_nb=False, nb_weight=0.0)

    training_sets = {"synthetic only": syn, f"sets {names}": example_train,
                     f"sets {names} + synthetic": example_train + syn}
    results = []
    for set_name, train in training_sets.items():
        systems = {
            "bayes-net + text evidence": bn_predictor(train, config),
            "bayes-net, no text evidence": bn_predictor(train, no_nb),
            "independent per-slot": IndependentSlots.train(train, REFERENCE_DATE, config).predict,
        }
        for system, predict in systems.items():
            s = score(predict, test, REFERENCE_DATE)
            results.append({"train": set_name, "system": system, **s.summary()})
    rules = score(Rules().predict, test, REFERENCE_DATE)
    results.append({"train": "(none)", "system": "rules only", **rules.summary()})

    # robustness across sets: leave one training set out
    cross = []
    for held in TRAIN_SETS:
        train = [ex for name, exs in annotated.items() if name != held for ex in exs] + syn
        s = score(bn_predictor(train, config), annotated[held], REFERENCE_DATE).summary()
        cross.append({"held_out": held, "exact_match": s["exact_match"], "decision_accuracy": s["decision_accuracy"]})

    final_train = example_train + syn
    final = Translator.train(final_train, REFERENCE_DATE, config)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    final.save(MODEL_PATH)
    final_score = score(lambda u, d: translate(final, u, d, strict=True), test, REFERENCE_DATE)
    default_score = score(lambda u, d: translate(final, u, d, strict=False), test, REFERENCE_DATE)

    out = {
        "reference_date": REFERENCE_DATE.isoformat(),
        "data": reports,
        "synthetic_examples": len(syn),
        "config": config.__dict__,
        "tuning_top_trials": top_trials,
        "ablation": results,
        "leave_one_set_out": cross,
        "final_model": {"path": str(MODEL_PATH.relative_to(MODEL_PATH.parents[2])),
                        "bytes": MODEL_PATH.stat().st_size, "train_examples": len(final_train),
                        "test": final_score.summary(), "test_default_mode": default_score.summary()},
        "test_errors": final_score.errors,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(out, indent=2, default=str) + "\n", "utf-8")
    (out_dir / "results.md").write_text(to_markdown(out), "utf-8")
    return out


def to_markdown(out: dict) -> str:
    lines = ["# Results", "",
             f"Held-out test set: set {TEST_SET} ({out['data'][TEST_SET]['lines']} lines, "
             f"{out['data'][TEST_SET]['kinds']}), never used for training or tuning. Reference date "
             f"{out['reference_date']}.", "",
             "## Ablation on the held-out set", "",
             "| training data | system | decision acc. | exact match | subgraph | metric | query type | grouping | direction | filters | time | limit | valid queries |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in out["ablation"]:
        sa = r["slot_accuracy"]
        lines.append(f"| {r['train']} | {r['system']} | {r['decision_accuracy']:.3f} | {r['exact_match']:.3f} | "
                     f"{sa['subgraph']:.3f} | {sa['metric']:.3f} | {sa['query_type']:.3f} | {sa['aggregate_by']:.3f} | "
                     f"{sa['direction']:.3f} | {sa['filters']:.3f} | {sa['time_range']:.3f} | {sa['limit']:.3f} | "
                     f"{r['validity_of_emitted_queries']:.3f} ({r['emitted_queries']}) |")
    lines += ["", "## Leave one set out (train on the other three + synthetic)", "",
              "| held out | exact match | decision acc. |", "|---|---|---|"]
    lines += [f"| {c['held_out']} | {c['exact_match']:.3f} | {c['decision_accuracy']:.3f} |" for c in out["leave_one_set_out"]]
    f = out["final_model"]
    t = f["test"]
    lines += ["", "## Final model", "",
              f"Trained on {f['train_examples']} examples, {f['bytes'] / 1024:.0f} KB on disk. On the held-out set: "
              f"decision accuracy {t['decision_accuracy']:.3f}, exact match {t['exact_match']:.3f}, "
              f"valid emitted queries {t['validity_of_emitted_queries']:.3f}, median latency {t['latency_ms_median']} ms "
              f"(p95 {t['latency_ms_p95']} ms).", "",
              "Decision confusion (gold -> predicted): " + ", ".join(f"{k}: {v}" for k, v in t["kind_confusion"].items()),
              "", "## Example data", "", "| file | lines | kinds | failed the checker |", "|---|---|---|---|"]
    lines += [f"| {d['file']} | {d['lines']} | {d['kinds']} | {d['invalid']} |" for d in out["data"].values()]
    lines += ["", f"Tuned configuration: `{json.dumps(out['config'])}`", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    import sys

    result = run(Path(sys.argv[1] if len(sys.argv) > 1 else "results"))
    print(to_markdown(result))
