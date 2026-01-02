from __future__ import annotations

import numpy as np
import pytest

from nl2gql.bn import BayesNet, Node
from nl2gql.extract import extract
from nl2gql.model import build_network, observations

from .conftest import TODAY


def _toy() -> BayesNet:
    """Rain -> WetGrass <- Sprinkler, with WetGrass observed."""
    net = BayesNet([
        Node("rain", ("no", "yes")),
        Node("sprinkler", ("off", "on")),
        Node("wet", ("no", "yes"), ("rain", "sprinkler")),
    ], latent=["rain", "sprinkler"])
    net.nodes["rain"].cpt = np.array([0.8, 0.2])
    net.nodes["sprinkler"].cpt = np.array([0.6, 0.4])
    net.nodes["wet"].cpt = np.array([[[1.0, 0.0], [0.1, 0.9]], [[0.2, 0.8], [0.01, 0.99]]])
    return net


def test_exact_inference_matches_hand_computation():
    post = _toy().posterior({"wet": "yes"})
    # P(rain, sprinkler, wet=yes), unnormalised
    joint = {("no", "off"): 0.8 * 0.6 * 0.0, ("no", "on"): 0.8 * 0.4 * 0.9,
             ("yes", "off"): 0.2 * 0.6 * 0.8, ("yes", "on"): 0.2 * 0.4 * 0.99}
    z = sum(joint.values())
    assert post.marginal("rain")["yes"] == pytest.approx((joint[("yes", "off")] + joint[("yes", "on")]) / z)
    assignment, p = post.map()
    assert assignment == {"rain": "no", "sprinkler": "on"}
    assert p == pytest.approx(joint[("no", "on")] / z)


def test_structural_zeros_survive_smoothing():
    mask = np.array([[True, False], [True, True]])
    net = BayesNet([Node("a", ("x", "y")), Node("b", ("p", "q"), ("a",), mask)], latent=["a", "b"])
    net.fit([{"a": "x", "b": "p"}], alpha=5.0)
    assert net.nodes["b"].cpt[0, 1] == 0.0
    assert net.nodes["b"].cpt[1].sum() == pytest.approx(1.0)


def test_schema_constraints_are_hard(synthetic_model):
    net = synthetic_model.net
    m = net.nodes["M"]
    kpis, cnc = net.nodes["S"].index("kpis"), net.nodes["S"].index("cnc_data")
    ranked = net.nodes["Q"].index("ranked")
    assert m.cpt[kpis, ranked, m.index("cnc_data.spindle_load")] == 0.0
    assert m.cpt[cnc, ranked, m.index("none")] == 0.0          # ranked needs a metric
    post = synthetic_model.posterior(extract("top lines by spindle load in Windsor", TODAY))
    assignment, _ = post.map()
    assert assignment["S"] == "cnc_data" and assignment["M"] == "cnc_data.spindle_load"
    # every assignment with a non-zero posterior pairs a metric with its own subgraph
    s_axis, m_axis = net.latent.index("S"), net.latent.index("M")
    for idx in zip(*np.nonzero(post.joint), strict=True):
        metric = m.states[idx[m_axis]]
        assert metric == "none" or metric.split(".")[0] == net.nodes["S"].states[idx[s_axis]]


def test_pgmpy_agrees_with_our_inference(synthetic_model):
    """Build the same network in pgmpy and compare posterior marginals (text evidence off)."""
    pgmpy = pytest.importorskip("pgmpy")
    from pgmpy.factors.discrete import TabularCPD
    from pgmpy.inference import VariableElimination
    try:
        from pgmpy.models import DiscreteBayesianNetwork as Network
    except ImportError:  # older pgmpy
        from pgmpy.models import BayesianNetwork as Network

    net = synthetic_model.net
    assert pgmpy
    edges = [(p, n.name) for n in net.nodes.values() for p in n.parents]
    model = Network(edges)
    for n in net.nodes.values():
        card = len(n.states)
        values = n.cpt.reshape(-1, card).T                     # pgmpy: rows = child states
        model.add_cpds(TabularCPD(n.name, card, values, evidence=list(n.parents) or None,
                                  evidence_card=[len(net.nodes[p].states) for p in n.parents] or None,
                                  state_names={n.name: list(n.states),
                                               **{p: list(net.nodes[p].states) for p in n.parents}}))
    infer = VariableElimination(model)
    for text in ("top lines by oee for october in the Americas", "cycle time on line 7 at Brno last week",
                 "show me spindle load readings for Toledo on night shift"):
        obs = observations(extract(text, TODAY))
        ours = net.posterior(obs)
        for node in ("S", "Q", "M", "A", "D"):
            theirs = infer.query([node], evidence=obs, show_progress=False)
            expected = dict(zip(theirs.state_names[node], theirs.values, strict=True))
            got = ours.marginal(node)
            for state, value in expected.items():
                assert got[state] == pytest.approx(value, abs=1e-9), (text, node, state)


def test_network_structure_is_acyclic_and_small():
    net = build_network()
    sizes = [len(net.nodes[n].states) for n in net.latent]
    assert int(np.prod(sizes)) < 20_000
