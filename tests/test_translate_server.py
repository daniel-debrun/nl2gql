from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request

import pytest
from graphql import parse, print_ast
from hypothesis import given, settings
from hypothesis import strategies as st

from nl2gql.checker import check, to_slots
from nl2gql.evaluate import load_jsonl, score
from nl2gql.model import Translator
from nl2gql.translate import translate

from .conftest import DATA, TODAY

RUNNING_EXAMPLE = "give me top lines by oee for the month of october in the region of Americas"


def test_running_example(synthetic_model):
    r = translate(synthetic_model, RUNNING_EXAMPLE, TODAY)
    expected = ('query { kpis(filter: {region: "AMER"}, aggregateBy: LINE, timeRange: {start: "2026-10-01", '
                'end: "2026-10-31"}, orderBy: {metric: OEE, direction: DESC}, limit: 10) { line oee } }')
    assert r.kind == "query"
    assert print_ast(parse(r.graphql)) == print_ast(parse(expected))
    assert r.confidence["joint"] > 0.5
    assert any("limit 10" in a for a in r.assumptions)


def test_worked_examples_all_correct(synthetic_model):
    examples, report = load_jsonl(DATA / "worked_examples.jsonl")
    assert not report.invalid
    s = score(lambda u, d: translate(synthetic_model, u, d, strict=True), examples, TODAY)
    assert s.summary()["decision_accuracy"] == 1.0, s.errors
    assert s.summary()["exact_match"] == 1.0, s.errors


@pytest.mark.parametrize(("text", "kind", "reason"), [
    ("oee for line 4 yesterday", "clarify", "needs_plant"),
    ("scrap rate for Graz in the Americas", "clarify", "inconsistent_hierarchy"),
    ("downtime at the Detroit plant last week", "clarify", "unknown_entity"),
    ("top lines by cycle time last week", "clarify", "ambiguous_metric"),
    ("best 5 lines in Chennai last month", "clarify", "missing_metric"),
    ("tell me a joke", "reject", "out_of_domain"),
])
def test_clarify_and_reject_in_strict_mode(synthetic_model, text, kind, reason):
    r = translate(synthetic_model, text, TODAY, strict=True)
    assert (r.kind, r.reason) == (kind, reason)


@pytest.mark.parametrize(("text", "defaulted", "subgraph", "metric"), [
    ("best 5 lines in Chennai last month", ["subgraph", "metric"], "kpis", "oee"),          # no metric at all
    ("energy usage per line at Windsor", ["subgraph", "metric"], "kpis", "oee"),             # unknown metric
    ("top 3 press lines in Graz", ["metric"], "press_data", "stroke_rate"),                  # source, no metric
])
def test_unrecognised_parameters_get_standard_defaults(synthetic_model, text, defaulted, subgraph, metric):
    r = translate(synthetic_model, text, TODAY)
    assert r.kind == "query" and check(r.graphql) == []
    assert (r.slots.subgraph, r.slots.metric) == (subgraph, metric)
    assert set(defaulted) <= set(r.defaulted)
    assert any("standard" in a for a in r.assumptions)


def test_ambiguous_cycle_time_picks_the_likelier_source_and_says_so(synthetic_model):
    r = translate(synthetic_model, "top lines by cycle time last week", TODAY)
    assert r.kind == "query" and r.slots.metric == "cycle_time"
    assert "subgraph" in r.defaulted and any("CNC or press" in a for a in r.assumptions)


@pytest.mark.parametrize(("text", "reason"), [
    ("oee for line 4 yesterday", "needs_plant"),              # a guessed plant would return the wrong data
    ("scrap rate for Graz in the Americas", "inconsistent_hierarchy"),
    ("tell me a joke", "out_of_domain"),
])
def test_locations_and_off_topic_still_ask_or_refuse_by_default(synthetic_model, text, reason):
    assert translate(synthetic_model, text, TODAY).reason == reason


def test_recognised_parameters_are_not_marked_defaulted(synthetic_model):
    r = translate(synthetic_model, "top 5 lines by oee in Windsor last week", TODAY)
    assert r.defaulted == []


@settings(max_examples=150, deadline=None)
@given(st.text(min_size=1, max_size=120))
def test_property_never_emits_an_invalid_query(synthetic_model, text):
    r = translate(synthetic_model, text, TODAY)
    if r.kind == "query":
        assert check(r.graphql) == []
        assert to_slots(r.graphql) == r.slots


WORDS = ["top", "5", "lines", "by", "oee", "cycle", "time", "in", "Graz", "line", "3", "press", "cnc", "worst",
         "scrap", "last", "week", "shift", "2", "per", "plant", "Americas", "area", "30", "readings", "tonnage"]


@settings(max_examples=200, deadline=None)
@given(st.lists(st.sampled_from(WORDS), min_size=1, max_size=12))
def test_property_domain_word_salad_is_valid_or_refused(synthetic_model, words):
    r = translate(synthetic_model, " ".join(words), TODAY)
    assert r.kind in ("query", "clarify", "reject")
    if r.kind == "query":
        assert check(r.graphql) == []


def test_save_and_load_roundtrip(synthetic_model, tmp_path):
    path = tmp_path / "model.npz"
    synthetic_model.save(path)
    loaded = Translator.load(path)
    for text in (RUNNING_EXAMPLE, "cycle time on line 7 at Brno last week", "tell me a joke"):
        a, b = translate(synthetic_model, text, TODAY), translate(loaded, text, TODAY)
        assert (a.kind, a.graphql, a.reason) == (b.kind, b.graphql, b.reason)


@pytest.fixture()
def server(synthetic_model):
    from http.server import ThreadingHTTPServer

    from nl2gql.server import make_handler

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(synthetic_model))
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


def _post(url, body):
    data = body if isinstance(body, bytes) else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def test_http_translate_check_schema_health(server):
    status, body = _post(server + "/translate", {"text": RUNNING_EXAMPLE, "today": "2026-11-18"})
    assert status == 200 and body["kind"] == "query" and "OEE" in body["graphql"]
    status, body = _post(server + "/translate", {"text": "top lines by cycle time", "strict": True})
    assert status == 200 and body["reason"] == "ambiguous_metric"
    status, body = _post(server + "/check", {"query": "query { kpis(shift: 7) { oee } }"})
    assert status == 200 and body["valid"] is False and body["issues"][0]["code"] == "shift_range"
    with urllib.request.urlopen(server + "/schema", timeout=10) as resp:
        assert "press_data" in resp.read().decode()
    with urllib.request.urlopen(server + "/health", timeout=10) as resp:
        assert json.loads(resp.read()) == {"status": "ok"}


@pytest.mark.parametrize("body", [b"not json", {"text": ""}, {"text": 5}, {"text": "oee", "today": "yesterday"}, [1],
                                  {"text": "oee", "strict": "yes"}])
def test_http_bad_requests(server, body):
    status, _ = _post(server + "/translate", body)
    assert status == 400
