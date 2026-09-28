import pytest
import sympy as sym

from circuit_quantization import Branch, Circuit, Loops, fundamental_cycles, incidence
from circuits import HARMONIUM_CYCLES, coupled, harmonium, zero_pi

C, EJ, L = sym.symbols("C E_J L", positive=True)


@pytest.mark.parametrize("kwargs, message", [
    (dict(EJ=EJ), "needs C > 0"),
    (dict(C=C, L=L, EJ=EJ), "at most one"),
    (dict(C=C, V=1), "must have C = 0"),
    (dict(C=sym.Symbol("x")), "provably positive"),
])
def test_branch_validation(kwargs, message):
    with pytest.raises(ValueError, match=message):
        Branch("b", 1, 0, **kwargs)


def test_source_rules():
    with pytest.raises(ValueError, match="connect to ground"):
        Circuit([Branch("J", 1, 0, C=C, EJ=EJ), Branch("V", 1, 2, V=1)], ground=0)
    with pytest.raises(ValueError, match="Only capacitors"):
        Circuit([Branch("J", 1, 2, C=C, EJ=EJ), Branch("V", 0, 2, V=1)], ground=0)


def test_ordering_and_incidence_match_notes():
    circuit = harmonium()
    assert circuit.nodes == (1, 2, 3, 4, 5)
    assert circuit.branch_names == ("C", "J1", "J2", "J3", "L1", "L2", "L3")
    assert incidence(circuit) == sym.Matrix([  # eq:harm_Omega
        [1, 1, 1, 0, 0, 0, 0],
        [0, -1, 0, 0, 1, 0, 0],
        [0, 0, -1, 0, 0, 1, 0],
        [0, 0, 0, 1, -1, -1, 1],
        [-1, 0, 0, -1, 0, 0, -1],
    ])


def test_branches_sorted_by_type_keep_user_order():
    circuit = coupled()
    assert circuit.branch_names[:8] == ("Ct", "Cc", "Cf", "CV1", "CV2", "CV3", "CV4", "Cg")
    assert circuit.nodes == (1, 2, 3, 4, 5, 6, 7)


def test_default_cycles_of_coupled_example():
    # fig:coupled_circuit_example: 2 superconducting + 7 capacitively broken loops.
    circuit = coupled()
    cycles = fundamental_cycles(circuit)
    assert len(cycles) == 9
    loops = Loops(circuit, cycles)
    assert loops.G * incidence(circuit).T == sym.zeros(9, 7)  # eq:G_OmegaT
    assert set(loops.G) <= {-1, 0, 1}
    kinds = [b.type for b in circuit.branches]
    superconducting = [all(kinds[k] in "JL" for k in range(15) if loops.G[i, k]) for i in range(9)]
    assert superconducting == [True, True] + [False] * 7  # eq:G_blocks
    # One capacitively broken loop closes through a source branch.
    assert any(loops.G[i, circuit.branch_names.index("V1")] for i in range(9))


def test_cycles_from_explicit_tree():
    circuit = harmonium()
    cycles = fundamental_cycles(circuit, tree=["C", "J1", "J2", "J3"])
    assert set(cycles) == {"L1", "L2", "L3"}
    assert cycles["L3"] == {"J3": -1, "L3": 1}


def test_floating_circuit_cycle_count():
    # zero-pi: isolated ground, two components, N_ell = 6 - 3.
    assert len(fundamental_cycles(zero_pi())) == 3


def test_loops_flux_defaults_are_visible():
    circuit = harmonium()
    loops = Loops(circuit, HARMONIUM_CYCLES, {"cb": 0})
    Phi_k, Phi_f = sym.symbols("Phi_k Phi_f", real=True)
    assert loops.Phi_ext == sym.Matrix([Phi_k, Phi_f, 0])


def test_loops_rejects_incomplete_or_invalid_bases():
    circuit = harmonium()
    with pytest.raises(ValueError, match="1 cycle\\(s\\) missing"):
        Loops(circuit, {k: HARMONIUM_CYCLES[k] for k in ("k", "f")})
    with pytest.raises(ValueError, match="not a graph cycle"):
        Loops(circuit, {**HARMONIUM_CYCLES, "k": {"J1": 1}})
    with pytest.raises(ValueError, match="dependent"):
        Loops(circuit, {**HARMONIUM_CYCLES, "cb": HARMONIUM_CYCLES["k"]})


def test_rebase_maps_fluxes_and_rejects_non_unimodular():
    circuit = harmonium()
    loops = Loops(circuit, HARMONIUM_CYCLES, {"cb": 0})
    T = sym.Matrix([[1, 0, 0], [1, 1, 0], [0, 0, 1]])
    rebased = loops.rebase(T * loops.G)
    assert rebased.Phi_ext == T * loops.Phi_ext
    with pytest.raises(ValueError, match="det"):
        loops.rebase(2 * loops.G)


def test_nodes_sort_naturally_across_label_types():
    circuit = Circuit([
        Branch("J1", "q10", "gnd", C=C, EJ=EJ), Branch("J2", 2, "gnd", C=C, EJ=EJ),
        Branch("J3", "q2", "gnd", C=C, EJ=EJ), Branch("J4", 10, "gnd", C=C, EJ=EJ),
    ], ground="gnd")
    assert circuit.internal_nodes == (2, 10, "q2", "q10")


def test_node_labels_must_differ_as_strings():
    with pytest.raises(ValueError, match="same string label"):
        Circuit([Branch("J1", 1, 0, C=C, EJ=EJ), Branch("J2", "1", 0, C=C, EJ=EJ)], ground=0)


@pytest.mark.parametrize("cycle", [
    {"J1": sym.Rational(1, 2), "J2": sym.Rational(-1, 2), "L1": sym.Rational(1, 2), "L2": sym.Rational(-1, 2)},
    {"J1": 2, "J2": -2, "L1": 2, "L2": -2},
])
def test_loops_reject_coefficients_outside_unit_set(cycle):
    # Each row of G is a physical loop, G in {0, +1, -1} (eq:G_OmegaT).
    with pytest.raises(ValueError, match="physical loop"):
        Loops(harmonium(), {**HARMONIUM_CYCLES, "k": cycle})


def test_rebase_requires_one_name_per_row():
    loops = Loops(harmonium(), HARMONIUM_CYCLES, {"cb": 0})
    with pytest.raises(ValueError, match="one name per row"):
        loops.rebase(loops.G, names=("a", "b"))
