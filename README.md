# nl2gql

Natural language to GraphQL for a closed manufacturing schema, with no language model at inference
time. A schema-derived AST builder guarantees that every query it emits is valid; a small Bayesian
network decides what the request means and says how sure it is. When the request is ambiguous or out
of scope, it asks or refuses instead of guessing.

```
$ nl2gql translate --today 2026-11-18 "give me top lines by oee for the month of october in the region of Americas"
{
  kpis(
    filter: {region: "AMER"}
    aggregateBy: LINE
    timeRange: {start: "2026-10-01", end: "2026-10-31"}
    orderBy: {metric: OEE, direction: DESC}
    limit: 10
  ) {
    line
    oee
  }
}
# no number given; limit 10 (default)
# 'the month of october' -> 2026-10-01 to 2026-10-31
# confidence 0.954, 3.7 ms

$ nl2gql translate --today 2026-11-18 "best 5 lines in Chennai last month"
...
# no data source recognised; used the standard one, kpis
# no metric recognised; used the standard one, oee
# defaulted: subgraph, metric

$ nl2gql translate --strict --today 2026-11-18 "cycle time for line 11 at nagoya this month"
clarify (ambiguous_metric): Cycle time exists for CNC machines and for presses. Which one?
```

On a held-out test set the model never saw (300 requests: 225 queries, 45
that need clarification, 30 off-topic), the final model gets **95.6% of queries exactly right, makes
the right query / clarify / refuse decision 98.3% of the time, and every one of the queries it emitted
is valid**. It is 62 KB on disk and answers in about 3 ms on a laptop CPU. Details and caveats are in
[Results](#results).

## Contents

1. [Why](#why)
2. [The schema](#the-schema)
3. [How it works](#how-it-works)
4. [Data](#data)
5. [Results](#results)
6. [Limitations](#limitations)
7. [Papers it stands on](#papers-it-stands-on)
8. [Usage](#usage)
9. [Development](#development)

## Why

General text-to-GraphQL is hard for large models. In IBM's benchmark (Kesarwani et al., EMNLP 2024
Industry Track) the best of ten LLMs, a Granite code model, reaches about 50% accuracy one-shot, and
many outputs are not valid queries at all. That work is LLM inference over arbitrary schemas.

Many real deployments are the opposite case: one fixed schema, a known vocabulary (plants, lines,
KPIs), users who want an answer in milliseconds, and a hard requirement that nothing invalid reaches
the API. There the right tool is a structured model that cannot produce an invalid query, is cheap
enough to run anywhere, and knows when it does not know. This project demonstrates that approach end
to end on a small, realistic schema.

## The schema

Three subgraphs, each its own SDL module in `src/nl2gql/schema/`, composed into one schema with
`extend type Query`:

| subgraph | metrics |
|---|---|
| `kpis` | oee, availability, performance, quality, scrap_rate, downtime_minutes |
| `cnc_data` | spindle_load, spindle_speed, feed_rate, tool_wear, cycle_time, alarm_count |
| `press_data` | tonnage, stroke_rate, die_temperature, cycle_time, reject_count |

Every subgraph takes the same arguments: `filter` on hierarchy keys (up to 4 levels), `aggregateBy`
a hierarchy level, `timeRange`, `shift` (1-3), `orderBy {metric, direction}` and `limit`. Each
subgraph has its own metric enum, so ordering KPIs by a CNC metric is a schema error, not just a
semantic one. `cycle_time` is deliberately in both machine subgraphs, so the engine has a real
ambiguity to resolve from context.

The factory hierarchy is a tree, `group > region > division > plant > subarea > area > line`: 2
groups, 3 regions, 6 divisions, 12 plants and 144 lines (`src/nl2gql/hierarchy.py`). Filters use keys
(`WIN-L03`); people use names ("line 3 at Windsor"). Subarea, area and line names repeat in every
plant, so "line 4" alone is ambiguous, exactly as in a real organisation.

There is no database. Queries are validated against the schema and the hierarchy, never executed.

## How it works

```
utterance
  -> 1. extractor        times, hierarchy names -> keys, shift, counts, metric / subgraph / ranking cues
  -> 2. Bayesian network joint MAP over subgraph, query type, metric, grouping level, direction
  -> 3. decision         query, or clarify (which plant? which cycle time? which metric?), or refuse
  -> 4. AST builder      slots -> GraphQL AST, every name looked up in the schema first
  -> 5. checker          graphql-core validation + hierarchy and value checks
```

**1. Extractor** (`extract.py`, `timeparse.py`, `lexicon.py`). Deterministic. It resolves time
phrases against an injectable "today" (weeks start Monday, "october" means the most recent October,
"last 7 days" includes today), maps names to hierarchy keys with typo tolerance, and checks them
against the tree: "Graz in the Americas" is a contradiction, "line 4" without a plant needs a plant,
"line 14" does not exist. It masks what it has consumed, so the 3 in "Q3" or "line 3" never becomes a
shift or a limit. It produces evidence, not decisions.

**2. Bayesian network** (`bn.py`, `model.py`). Six latent nodes (subgraph S, query type Q, metric M,
metric polarity P, grouping level A, direction D) and ten observed evidence nodes from the extractor:

```
         S ──────────► e_sub (cnc/press/kpi words), e_loc (kind of filtered subarea)
         │
    Q ──►M ──► P (polarity, deterministic) ──┐
    │    └──► e_met (metric word)            │
    ├──► D ──────────────────────────────────┴──► e_rank (top/best/worst/...)
    ├──► A ◄── F (finest filter level)
    │    └──► e_agg (level in a grouping phrase) ◄── F
    └──► e_form, e_raw, e_num, e_sing
```

The structure encodes how the slots constrain each other, and the hard facts are structural zeros
taken from the schema, which smoothing cannot remove: a metric must belong to its subgraph, raw
queries have no grouping or direction, ranked queries have a direction and a metric. "best" means
DESC for OEE and ASC for scrap rate because `e_rank` depends on the metric's polarity. Words the
lexicon does not cover enter as tempered naive-Bayes soft evidence on S, Q and M. Parameters are
learned by counting. Inference is exact enumeration over about 15,000 joint states in numpy. The
implementation is cross-checked against pgmpy's variable elimination in the tests (agreement to 1e-9).

Why a network rather than one classifier per slot: slots depend on each other. "oee" makes `kpis`
near-certain, which fixes the legal metrics. "line 7 at Brno" is in Machining, which makes "cycle
time" mean CNC cycle time. A request with a grouping phrase is grouped finer than its filter, and one
without is a summary at the filter's level. Joint inference uses all of that; independent classifiers
cannot, and the ablation below measures the difference.

**3. Decision** (`translate.py`). When a parameter is not recognised, the engine fills in a standard
value, lists the assumption, and names the slot in `defaulted`, so a caller can always tell a guess
from a reading:

| not recognised | default |
|---|---|
| no metric and no data source ("best 5 lines in Chennai") | `kpis`, and OEE if a metric is needed |
| a metric the schema does not have ("energy usage per line") | the same, with a note naming the unknown word |
| a data source but no metric for a ranking ("top 3 press lines") | that subgraph's standard metric: OEE, CNC cycle time, press stroke rate |
| cycle time with nothing deciding CNC or press | whichever the network finds likelier, with its share of belief |
| no number for a ranking | 10, or 1 for a singular ("which plant") |
| no number for records or aggregates | no limit: the argument is left out |

The standard subgraph and metrics are settings (`Config.default_subgraph`, `default_metrics`). They
are fixed choices rather than learned popularity, because the training data is balanced across
subgraphs by design, so "most popular" there would be arbitrary.

Two things are never defaulted: locations and topic. A line without a plant, contradictory places
("Graz in the Americas") or a plant that does not exist get a clarifying question, because a guessed
plant returns another plant's data. Requests that are not about production data are refused. With
`strict` on (`--strict`, or `"strict": true` over HTTP), the engine asks instead of defaulting in
every case above. Thresholds are tuned by cross-validation.

**4. AST builder** (`builder.py`). Slots become graphql-core AST nodes; every argument, input field,
enum value and selected field is looked up in the loaded schema before it is emitted, and the text is
printed from the AST. There is no string-templating path.

**5. Checker** (`checker.py`). Syntax, `graphql.validate` against the composed schema, the expected
shape, and the semantics SDL cannot express: keys exist at the level they filter and lie on one path
of the tree, at most 4 levels, ordered valid dates, shift 1-3, a positive limit, the ordered-by metric
is selected. The same checker validates every training label, and it runs on every emitted query.

## Data

All data in this repository is fabricated. There are no real users, plants or logs behind it: five
example sets of 300 requests each (`data/examples/A-E.jsonl`), written to the conventions in
`data/CONVENTIONS.md`, each with 225 queries, 45 requests that need clarification and 30
off-topic requests. Every GraphQL label passes the checker.

Sets A-D are used for training, development and tuning. Set E is the test set: it was written
after development finished, scored once, and nothing was changed after seeing it. A schema-driven
generator (`generate.py`, after "Building a Semantic Parser Overnight") adds 6,000 synthetic examples.

One convention is inconsistent across the sets: whether "the stamping lines at Brno" adds a Press
Shop filter (15 labels say yes, 18 say no). The engine follows the written convention (only a named
subarea is a filter), so those labels cap exact match.

## Results

All numbers are on set E, the held-out test set, which was written after development ended
and scored once. Exact match means every slot is right: subgraph, metric, grouping, direction,
filters, dates, shift and limit. Full output: [`results/results.md`](results/results.md).

| training data | system | decision acc. | exact match | query type | grouping | valid queries |
|---|---|---|---|---|---|---|
| synthetic only | Bayesian network + text evidence | 0.987 | 0.956 | 0.982 | 0.987 | 100% |
| sets A-D | Bayesian network + text evidence | 0.983 | 0.916 | 0.982 | 0.956 | 100% |
| sets A-D | Bayesian network, no text evidence | 0.983 | 0.929 | 0.982 | 0.956 | 100% |
| sets A-D | independent per-slot classifiers | 0.957 | 0.720 | 0.889 | 0.836 | 96.1% |
| **sets A-D + synthetic** | **Bayesian network + text evidence (final)** | **0.983** | **0.956** | 0.982 | 0.987 | **100%** |
| sets A-D + synthetic | independent per-slot classifiers | 0.953 | 0.787 | 0.844 | 0.836 | 100% |
| none | hand-written rules on the same evidence | 0.940 | 0.964 | 0.982 | 0.987 | 100% |

Leave-one-set-out on the four training sets (train on three plus synthetic, test on the
fourth) gives 95.6-99.1% exact match and 99.3-100% decision accuracy.

What this shows:

- **Joint inference is what makes the evidence usable.** Independent per-slot classifiers, given
  exactly the same evidence and training data, reach 72-79% exact match; the network reaches 95.6%.
  The gap is in the slots that depend on each other: query type (84-89% vs 98%) and grouping level
  (84% vs 99%). Trained on the example sets alone, the independent classifiers also emit invalid slot
  combinations (3.9% of their queries), which the network's structural zeros rule out.
- **Hand-written rules on the same evidence are as good on slots.** Rules reach 96.4% exact match,
  slightly above the network's 95.6%, because the extractor already does most of the work for a
  vocabulary this small. The network's advantage is the decision layer: 98.3% correct
  query / clarify / refuse decisions against 94.0% for rules, plus calibrated confidences, and
  learned behaviour instead of rules someone has to write and maintain. On a closed domain like this
  one a careful rule system is a serious baseline, and the table says so.
- **The naive-Bayes text evidence adds little here.** With synthetic data in the mix it makes no
  measurable difference; on the example sets alone it slightly hurts exact match (91.6% vs 92.9%). The
  tuned weight is small (0.2). It is kept because it is what carries words outside the lexicon, but
  this test set does not show a benefit.
- **Synthetic data alone is nearly as good as the full training set** (95.6% either way on E),
  which says more about how much of the task the schema and the extractor encode than about the model.

The 12 errors on the test set are all phrasings the extractor does not cover: "on nights" for shift 3,
"fewest alarms by line" as a ranking, "which lines" (plural) read as singular, "from November 5th" read
as "since", "the 5 fastest lines by spindle speed" (speed words follow polarity, and spindle speed has
none), "8 raw kpi entries" picking up "quality" from "quality engineer here", and "a poem about robots
on the assembly line" asking which plant instead of refusing. None was fixed after the test set was
scored.

**Default mode versus strict mode.** The labelled clarifications encode strict behaviour, so the
table above is scored in strict mode. In the default mode the same model answers 21 of the 45
requests labelled as needing clarification with a defaulted query, marked as such; exact
match on real queries is unchanged (95.6%), every one of the 245 emitted queries is valid, and
decision accuracy against the strict labels is 92.3% by design.

Speed and size: 62 KB model file, median 2.7 ms and p95 3.9 ms per request on a laptop CPU, most of it
in the regex extractor; inference itself is a numpy sum over about 15,000 states.

## Limitations

- **One metric per query and one value per hierarchy level.** "OEE and scrap rate by plant" and
  "Windsor versus Puebla" are outside what the slots represent; they get a clarifying question or a
  single-metric answer.
- **Defaults can hide a misunderstanding.** A defaulted query is valid but may not be what the person
  meant; callers should surface `defaulted` and `assumptions` to the user, or run in strict mode.
- **The domain is fixed.** Adding a metric means editing the SDL (the builder, checker and network
  constraints follow automatically) and the lexicon, then retraining. A new schema with a different
  argument shape needs a new slot design.
- **The extractor is hand-written.** It covers the conventions in `data/CONVENTIONS.md` and the phrasing of the example
  sets. Real users will find phrasings it misses; the naive-Bayes evidence carries unknown words,
  but new entity names are not learned on their own.
- **The data is fabricated.** It is a proxy for real requests, not a substitute for logs from real
  users. The labels contain some noise (for example one "this quarter" labelled as Q3), which is
  left in rather than hand-corrected.
- **English only, one calendar convention** (Monday weeks, month-first ambiguous dates).
- **No execution.** Correct here means "the query matches the annotated meaning and is valid".

## Papers it stands on

- **Belief networks for closed-domain language understanding:** H. Meng, W. Lam, C. Wai, "To believe
  is to understand", Eurospeech 1999; H. Meng, C. Wai, R. Pieraccini, "The use of belief networks for
  mixed-initiative dialog modeling", IEEE Trans. Speech and Audio Processing, 2003. A belief network
  over semantic-concept evidence, with out-of-domain rejection and detection of missing concepts
  followed by clarification. This is the main precedent for the decision layer.
- **A network over slot dependencies:** B. Thomson, S. Young, "Bayesian update of dialogue state: A
  POMDP framework for spoken dialogue systems", Computer Speech & Language 24(4), 2010.
- **Grammar- and type-constrained output:** P. Yin, G. Neubig, "TRANX: A Transition-based Neural
  Abstract Syntax Parser", EMNLP 2018 (AST construction driven by a grammar specification);
  J. Krishnamurthy, P. Dasigi, M. Gardner, "Neural Semantic Parsing with Type Constraints for
  Semi-Structured Tables", EMNLP 2017.
- **Schema-derived training data:** Y. Wang, J. Berant, P. Liang, "Building a Semantic Parser
  Overnight", ACL 2015.
- **The task and the LLM baseline:** A. Kesarwani et al. (IBM Research), "GraphQL Query Generation: A
  Large Training and Benchmarking Dataset", EMNLP 2024 Industry Track; A. Gupta et al., "Schema and
  Natural Language Aware In-Context Learning for Improved GraphQL Query Generation", NAACL 2025
  Industry Track.

No prior work known to this project combines a Bayesian network with a schema-derived AST for
text-to-query. It assembles established parts; it does not claim a new method.

## Usage

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e '.[dev]'

nl2gql translate "which plant had the worst scrap rate last month?"
nl2gql translate --json --today 2026-11-18 "5 lines with the lowest stroke rate in Graz during Q3"
nl2gql translate --strict "top lines by cycle time"   # ask instead of filling in defaults
nl2gql serve --port 8080          # POST /translate, POST /check, GET /schema, GET /health
nl2gql check data/examples/E.jsonl  # validate a JSONL file of labelled queries
nl2gql schema                     # print the composed SDL
```

HTTP:

```bash
curl -s localhost:8080/translate -d '{"text": "oee by plant in Europe this week", "today": "2026-11-18"}'
curl -s localhost:8080/translate -d '{"text": "top lines by cycle time", "strict": true}'
curl -s localhost:8080/check -d '{"query": "query { kpis(shift: 4) { oee } }"}'
```

Docker (the image contains only numpy, graphql-core and the trained model):

```bash
docker build -t nl2gql .
docker run --rm -p 8080:8080 nl2gql
# or: docker compose up
```

## Development

```bash
pytest                 # builder/checker properties, time conventions, extractor regressions,
                       # network inference (incl. the pgmpy cross-check), translation, HTTP
ruff check .
nl2gql train           # tune, run the ablation, write results/ and the bundled model
```

`nl2gql train` reproduces `results/results.md` and `src/nl2gql/models/nl2gql.npz` from the data in
this repository (about 15 minutes on a laptop, most of it the tuning grid).

Apache-2.0.
