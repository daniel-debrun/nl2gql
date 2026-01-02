# Labelling conventions

The rules every labelled example in `data/examples/` follows: how a plain-English request about
factory production data maps to a GraphQL query, or to a clarify / reject label. The translator
implements the same rules. `data/worked_examples.jsonl` has 20 worked examples; the schema is in
`src/nl2gql/schema/`.

## Reference date

All examples use **Wednesday 2026-11-18** as today; relative dates resolve against it.

## The data

Three subgraphs (root query fields), all with the same arguments:

| subgraph | metrics (record fields) | ordering enum |
|---|---|---|
| `kpis` | oee, availability, performance, quality, scrap_rate, downtime_minutes | `KpiMetric` |
| `cnc_data` | spindle_load, spindle_speed, feed_rate, tool_wear, cycle_time, alarm_count | `CncMetric` |
| `press_data` | tonnage, stroke_rate, die_temperature, cycle_time, reject_count | `PressMetric` |

Arguments: `filter: {group, region, division, plant, subarea, area, line}` (hierarchy **keys**),
`aggregateBy: GROUP|REGION|DIVISION|PLANT|SUBAREA|AREA|LINE`, `timeRange: {start, end}` (inclusive
ISO dates), `shift: 1|2|3`, `orderBy: {metric, direction: ASC|DESC}`, `limit`.

Which direction is better (used for "best"/"worst"):
- higher is better: oee, availability, performance, quality, stroke_rate
- lower is better: scrap_rate, downtime_minutes, tool_wear, cycle_time, alarm_count, reject_count
- neutral (no better direction): spindle_load, spindle_speed, feed_rate, tonnage, die_temperature.
  Examples never use "best"/"worst" with these; they use highest/lowest.

## The factory hierarchy

group > region > division > plant (key: name):

```
MOB: Acme Mobility
  AMER: Americas
    BODY: Body Systems      plants WIN: Windsor, PUE: Puebla
    PWT: Powertrain         plants TOL: Toledo, SAL: Saltillo
  EUR: Europe
    CHS: Chassis            plants GRZ: Graz, WRO: Wroclaw
    INT: Interiors          plants STU: Stuttgart, BRN: Brno
IND: Acme Industrial
  APAC: Asia Pacific
    ELX: Electronics        plants SUZ: Suzhou, PEN: Penang
    SEAT: Seating           plants CHE: Chennai, NAG: Nagoya
```

Every plant has the same layout below it. Keys are prefixed with the plant key (`WIN-PS`, `WIN-A10`,
`WIN-L01`):

| subarea (key suffix) | areas | lines |
|---|---|---|
| Press Shop (`-PS`) | Area 10 (`-A10`), Area 20 (`-A20`) | Line 1, 2 in Area 10; Line 3, 4 in Area 20 |
| Machining (`-MC`) | Area 30, Area 40 | Line 5, 6 in Area 30; Line 7, 8 in Area 40 |
| Assembly (`-AS`) | Area 50, Area 60 | Line 9, 10 in Area 50; Line 11, 12 in Area 60 |

Line keys use two digits: Line 3 at Windsor is `WIN-L03`, Line 11 at Graz is `GRZ-L11`.

Subarea, area and line names repeat in every plant, so they only identify something together with a
plant. There is no Detroit plant, no "Line 13", no "Area 70": anything not listed does not exist.

## Conventions

**Filters.** Include exactly the hierarchy levels the person names, as keys. "line 3 at Windsor" ->
`{plant: "WIN", line: "WIN-L03"}`. "the press shop in Graz" -> `{plant: "GRZ", subarea: "GRZ-PS"}`.
"Europe" -> `{region: "EUR"}`. Never add a level the person did not mention. At most 4 levels.

**Query type.**
- *Ranked*: any ranking language (top, bottom, best, worst, highest, lowest, most, least, rank,
  sort by). Needs `orderBy` and a `limit`. `aggregateBy` is the level being ranked ("top lines" ->
  LINE, "which plant" -> PLANT). Ranking individual records ("the 10 highest spindle load readings on
  line 5") has no `aggregateBy`.
- *Aggregate*: "by X", "per X", "for each X", "broken down by X", "grouped by X" -> `aggregateBy: X`,
  no `orderBy`. A value for one named entity ("oee for Windsor", "what was downtime in the Graz press
  shop") is an aggregate at the level of the finest entity named (PLANT, SUBAREA).
  With no entity named ("scrap rate last month"), aggregate at GROUP.
- *Raw records*: records, readings, rows, entries, logs, raw data, "all cnc data for ...", or any
  breakdown by shift or by day (shift and date are not hierarchy levels). No `aggregateBy`, no
  `orderBy`.

**Direction.** top, highest, most, max, largest, biggest, "rank/sort by" -> DESC. bottom, lowest,
least, min, smallest, fewest, "ascending" -> ASC. best -> DESC for higher-is-better metrics, ASC for
lower-is-better. worst -> the opposite of best.

**Limit.** A number in the request is the limit ("top 5", "three worst", "first 20 records").
Ranked with no number: plural ("top lines") -> 10, singular ("which line", "the best plant") -> 1.
Raw and aggregate queries get a limit only when a number is given.

**Time.** No time phrase -> no `timeRange`. Otherwise (today = 2026-11-18):
- today 2026-11-18..2026-11-18; yesterday 2026-11-17..2026-11-17
- this week 2026-11-16..2026-11-18 (weeks start Monday); last week 2026-11-09..2026-11-15
- this month 2026-11-01..2026-11-18; last month 2026-10-01..2026-10-31
- a month name alone ("october", "in march", "the month of june"): the most recent such month that
  has started, as a full month: January..October -> 2026, December -> 2025. November -> this month.
  With a year ("march 2025"), that full month.
- this quarter 2026-10-01..2026-11-18; last quarter 2026-07-01..2026-09-30; Q1..Q3 alone -> 2026
  (Q1 = Jan-Mar, Q2 = Apr-Jun, Q3 = Jul-Sep), Q4 alone -> this quarter; "Q2 2025" -> that quarter
- this year / year to date / YTD 2026-01-01..2026-11-18; last year or "2025" 2025-01-01..2025-12-31
- last/past N days: today and the N-1 days before (last 7 days = 2026-11-12..2026-11-18);
  last/past N weeks: today and the 7N-1 days before
- since DATE: DATE..today; on DATE: DATE..DATE; from/between DATE and/to DATE: inclusive
- a date without a year is the most recent such date not after today ("Nov 3" = 2026-11-03,
  "Dec 5" = 2025-12-05)

**Shift.** shift 1/2/3, first/second/third shift, 1st/2nd/3rd shift; day shift = 1, afternoon
shift = 2, night shift = 3.

**Metric and subgraph.** A request names one metric, or asks for all data of one subgraph ("all press
data", "kpis for Windsor", "cnc data"), in which case every metric of that subgraph is selected.
Requests for two or three specific metrics at once are out of scope. `cycle_time` exists in both machine subgraphs: pick cnc_data
when the request mentions CNC, machining, spindles, milling or a Machining line (5-8), press_data when
it mentions presses, stamping, dies or a Press Shop line (1-4). If nothing decides it, the request
needs clarification.

**Selection set.** Aggregated or ranked: the grouping field then the metric (`{ line oee }`). Raw:
`{ date shift line <metric> }`. All metrics: list every metric field of the subgraph.

## When there is no query

These are labelled `"kind": "clarify"` with a `reason`:
- `needs_plant`: a line, area or subarea with no plant ("oee for line 4")
- `ambiguous_metric`: cycle time with nothing that picks CNC or press
- `missing_metric`: ranking with no metric ("best 5 lines in Chennai")
- `inconsistent_hierarchy`: names that cannot both hold ("Graz in the Americas", "line 9 of the
  Windsor press shop")
- `unknown_entity`: a plant, line, area or metric that does not exist ("Detroit", "line 14",
  "energy consumption")

These are labelled `"kind": "reject"`, `"reason": "out_of_domain"`: anything that is not a request for this
production data (weather, email, jokes, HR questions, writing code, general chat).

## File format

JSON Lines, one example per line, UTF-8:

```json
{"id": "A-0001", "utterance": "...", "kind": "query", "graphql": "query { ... }"}
{"id": "A-0002", "utterance": "...", "kind": "clarify", "reason": "needs_plant"}
{"id": "A-0003", "utterance": "...", "kind": "reject", "reason": "out_of_domain"}
```

`graphql` is a complete query document with literal arguments (no variables, fragments or aliases).
Formatting and argument order do not matter: labels are compared as slots. Every label passes the
checker (`nl2gql check data/examples/*.jsonl`), which proves validity against the schema and the
hierarchy; whether a label means what its request asks is what the conventions above define.
