import sympy as sym
import pytest

from circuit_quantization import (
    Branch, Circuit, ghz_to_farad, ghz_to_henry, scqubits_yaml, to_scqubits,
)


def test_yaml_units_substitutions_and_parallel_capacitance():
    ec = sym.Symbol("EC", positive=True)
    circuit = Circuit([
        Branch("J", "b", "g", C=ghz_to_farad(ec), EJ=7),
        Branch("L", "a", "b", L=ghz_to_henry(2), C=ghz_to_farad(3)),
    ], ground="g")
    text = scqubits_yaml(circuit, subs={ec: 1}, node_order=["b", "a"])
    assert text == "branches:\n- [JJ, 1, 0, EJ_0 = 7, EC_0 = 1]\n- [L, 2, 1, EL_1 = 2]\n- [C, 2, 1, EC_1 = 3]\n"
    with pytest.raises(ValueError, match="Missing substitutions"):
        scqubits_yaml(circuit)
    with pytest.raises(ValueError, match="node_order"):
        scqubits_yaml(circuit, node_order=["a", "a"])
    with pytest.raises(ValueError, match="finite and positive"):
        scqubits_yaml(circuit, subs={ec: -1})


def test_sources_are_not_silently_dropped():
    circuit = Circuit([Branch("gate", 1, 2, C=1), Branch("V", 2, 0, V=1)], ground=0)
    with pytest.raises(ValueError, match="voltage-source"):
        scqubits_yaml(circuit)


def test_export_matches_transmon_spectrum():
    scq = pytest.importorskip("scqubits")
    import numpy as np

    circuit = Circuit([Branch("J", "island", "g", C=ghz_to_farad(.3), EJ=8)], ground="g")
    exported = to_scqubits(circuit, cutoff=12, trunc_num=5)
    expected = scq.Transmon(EJ=8, EC=.3, ng=0, ncut=12)
    assert np.allclose(exported.eigenvals(4), expected.eigenvals(4))
    assert exported.truncated_dim == 5


def test_closure_mapping_after_split_inductor_and_custom_transform():
    pytest.importorskip("scqubits")
    import numpy as np

    circuit = Circuit([
        Branch("J", "b", "g", C=ghz_to_farad(1), EJ=7),
        Branch("arm", "b", "a", L=ghz_to_henry(2), C=ghz_to_farad(3)),
        Branch("return", "a", "g", L=ghz_to_henry(4)),
    ], ground="g")
    exported = to_scqubits(circuit, symbolic=True, node_order=["b", "a"],
                           transformation_matrix=sym.eye(2), closure_branches=["return"])
    assert exported.closure_branches == [exported.branches[3]]
    assert [node.index for node in exported.branches[0].nodes] == [1, 0]
    assert np.array_equal(exported.transformation_matrix, np.eye(2))
    with pytest.raises(ValueError, match="distinct circuit branch names"):
        to_scqubits(circuit, symbolic=True, closure_branches=["missing"])


def test_floating_topology_and_hierarchy():
    pytest.importorskip("scqubits")
    circuit = Circuit([
        Branch("J1", "a", "b", C=ghz_to_farad(1), EJ=7),
        Branch("J2", "b", "c", C=ghz_to_farad(2), EJ=8),
    ])
    exported = to_scqubits(circuit, cutoff=8, trunc_num=[3, 4])
    assert not exported.is_grounded
    assert exported.system_hierarchy == [[1], [2]]
    assert exported.subsystem_trunc_dims == [3, 4]
