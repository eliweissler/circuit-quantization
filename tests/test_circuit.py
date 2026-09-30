import pytest
import sympy as sym

from circuit_quantization import Branch, Circuit, Loops, fundamental_cycles, incidence
from circuits import HARMONIUM_CYCLES, coupled, harmonium, zero_pi

C, EJ, L = sym.symbols("C E_J L", positive=True)


@pytest.mark.parametrize("kwargs, message", [
    (dict(C=0, EJ=EJ), "needs C > 0"),
    (dict(C=C, L=L, EJ=EJ), "at most one"),
    (dict(C=C, V=1), "must have C = 0"),
    (dict(C=sym.Symbol("x")), "provably positive"),
])
def test_branch_validation(kwargs, message):
    with pytest.raises(ValueError, match=message):
        Branch("b", 1, 0, **kwargs)


@pytest.mark.parametrize("name,kind,variables", [
    ("C1", "C", ("C_1",)), ("L1", "L", ("L_1",)),
    ("J1", "J", ("C_J1", "E_J1")), ("V1", "V", ("V_1",)),
    ("C", "C", ("C",)), ("J_1", "J", ("C_J1", "E_J1")),
])
def test_automatic_branch_symbols(name, kind, variables):
    branch = Branch(name, 1, 0)
    assert branch.type == kind
    assert tuple(map(str, branch.variables)) == variables
    assert all(v.is_real for v in branch.variables)
    assert all(v.is_positive for v in branch.variables) if kind != "V" else branch.V.is_positive is None
    if kind == "L":
        assert branch.C == 0


def test_automatic_energy_symbols_and_explicit_values():
    from circuit_quantization import farad_to_ghz, henry_to_ghz
    junction = Branch("J1", 1, 0, energy_symbols=True)
    ec, ej = junction.variables
    assert (str(ec), str(ej)) == ("E_CJ1", "E_J1")
    assert farad_to_ghz(junction.C) == ec
    inductor = Branch("L1", 1, 0, energy_symbols=True)
    assert henry_to_ghz(inductor.L) == inductor.variables[0]
    assert str(inductor.variables[0]) == "E_L1"
    assert Branch("J1", 1, 0, EJ=3).variables == (sym.Symbol("C_J1", positive=True),)
    assert Branch("custom", 1, 0, C=2, EJ=3, energy_symbols=True).variables == ()
    assert Branch("custom", 1, 0, C=2, EJ=3, energy_symbols=True).C == 2
    with pytest.raises(ValueError, match="name starting"):
        Branch("unknown", 1, 0)


def test_variables_preserve_order_shared_symbols_and_loop_rebasing():
    junction = Branch("J1", 1, 0, energy_symbols=True)
    circuit = Circuit([junction, Branch("J2", 1, 0, C=junction.C, EJ=junction.EJ),
                       Branch("C1", 1, 0)], ground=0)
    assert circuit.variables == (sym.Symbol("C_1", positive=True), *junction.variables)
    loops = Loops(circuit, {"k": {"J1": 1, "J2": -1}, "cb": {"C1": 1, "J1": -1}}, {"cb": 0})
    assert loops.variables == (sym.Symbol("Phi_k", real=True),)
    assert loops.rebase(-loops.G).variables == loops.variables
    assert Loops(circuit, dict(zip(loops.names, ({"J1": 1, "J2": -1}, {"C1": 1, "J1": -1}))),
                 {"k": 0, "cb": 0}).variables == ()


def test_automatic_symbols_quantize_and_export():
    from circuit_quantization import quantize, scqubits_yaml
    circuit = Circuit([Branch("J1", 1, 0, energy_symbols=True), Branch("L1", 1, 0, energy_symbols=True)], ground=0)
    loops = Loops(circuit, fundamental_cycles(circuit), {"L1": 0})
    ec, ej, el = circuit.variables
    result = quantize(circuit, loops)
    phase, charge = result.ham.varphi[0], result.ham.n[0]
    assert sym.expand(result.H - (4 * ec * charge**2 + el * phase**2 / 2 - ej * sym.cos(phase))) == 0
    assert "EJ_0 = 8" in scqubits_yaml(circuit, subs={ec: 1, ej: 8, el: 2})


def test_source_rules():
    with pytest.raises(ValueError, match="connect to ground"):
        Circuit([Branch("J", 1, 0, C=C, EJ=EJ), Branch("V", 1, 2, V=1)], ground=0)
    with pytest.raises(ValueError, match="Only capacitors"):
        Circuit([Branch("J", 1, 2, C=C, EJ=EJ), Branch("V", 0, 2, V=1)], ground=0)


@pytest.mark.parametrize("ground", [0, 1])
def test_disconnected_branches_are_rejected(ground):
    with pytest.raises(ValueError, match="connected graph"):
        Circuit([Branch("C1", 1, 2), Branch("C2", 3, 4)], ground=ground)


def test_empty_circuit_is_rejected():
    with pytest.raises(ValueError, match="nonempty connected graph"):
        Circuit([], ground=0)


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


def test_spanning_trees_harmonium_count_and_cycle_constraints():
    import networkx as nx
    from circuit_quantization import spanning_trees
    circuit = harmonium()
    trees = list(spanning_trees(circuit))
    graph = nx.MultiGraph((b.tail, b.head) for b in circuit.branches)
    # Independent count by the matrix-tree theorem (parallel edges count separately).
    laplacian = sym.Matrix(nx.laplacian_matrix(graph).toarray())
    assert len(trees) == laplacian[:-1, :-1].det() == 20
    assert len(set(trees)) == len(trees)
    assert trees == list(spanning_trees(circuit))
    for tree in trees:
        cycles = Loops(circuit, fundamental_cycles(circuit, tree))
        assert cycles.G * incidence(circuit).T == sym.zeros(3, 5)


def test_spanning_forests_preserve_parallel_branch_choices_and_isolated_ground():
    from circuit_quantization import spanning_trees
    circuit = Circuit([
        Branch("a", 1, 2, C=1), Branch("b", 2, 1, C=1),
        Branch("c", 2, 3, C=1), Branch("d", 2, 3, C=1),
    ], ground=0)
    assert list(spanning_trees(circuit)) == [("a", "c"), ("a", "d"), ("b", "c"), ("b", "d")]
    for tree in spanning_trees(circuit):
        assert Loops(circuit, fundamental_cycles(circuit)).G.shape == (2, 4)


def test_different_spanning_trees_can_give_the_same_cycle_matrix():
    from circuit_quantization import spanning_trees
    circuit = Circuit([
        Branch("a", 0, 1, C=1), Branch("b", 1, 2, C=1), Branch("c", 2, 0, C=1),
    ], ground=0)
    trees = list(spanning_trees(circuit))
    assert len(trees) == 3
    for tree in trees:
        cycles = fundamental_cycles(circuit, tree)
        assert Loops(circuit, cycles).G == sym.Matrix([[1, 1, 1]])
        assert set(tree) == set(circuit.branch_names) - set(cycles)


def test_loop_tuples_infer_orientation_and_roundtrip():
    circuit = harmonium()
    labels = {"k": ("J1", "J2", "L1", "L2"), "f": ("J3", "L3"),
              "cb": ("C", "J1", "J3", "L1")}
    loops = Loops(circuit, labels, {"cb": 0})
    expected = Loops(circuit, HARMONIUM_CYCLES, {"cb": 0})
    assert loops.G == expected.G
    assert loops.branch_dict == labels
    assert loops.G * incidence(circuit).T == sym.zeros(3, 5)
    flipped = loops.rebase(-loops.G)
    assert Loops(circuit, flipped.branch_dict).G == flipped.G
    reversed_f = Loops(circuit, {**labels, "f": ("L3", "J3")})
    assert reversed_f.G[1, :] == -loops.G[1, :]


def test_default_loop_constructor_prints_closure_first_tuples(capsys):
    circuit = harmonium()
    loops = Loops.from_circuit(circuit)
    repeated = Loops.from_circuit(circuit)
    assert repeated.branch_dict == loops.branch_dict
    assert repeated.G == loops.G
    assert loops.G == Loops(circuit, fundamental_cycles(circuit)).G
    assert all(labels[0] == name for name, labels in loops.branch_dict.items())
    output = capsys.readouterr().out
    for name in loops.names:
        assert f"{name}: +{name}" in output
        assert f"flux = Phi_{name}" in output
    assert Loops(circuit, loops.branch_dict).G == loops.G


@pytest.mark.parametrize("labels, message", [
    (("J1", "J1"), "distinct"), ((), "nonempty"),
    (("J1", "unknown"), "unknown branches"),
    (("J1", "L1"), "single closed loop"),
    (("J1", "J2", "L1", "L2", "J3"), "single closed loop"),
    (("J1", "J2", "L1", "L2", "J3", "L3"), "single closed loop"),
])
def test_loop_tuples_reject_invalid_support(labels, message):
    with pytest.raises(ValueError, match=message):
        Loops(harmonium(), {"bad": labels})


def test_default_loop_constructor_accepts_tree_circuit(capsys):
    circuit = Circuit([Branch("J1", 1, 0)], ground=0)
    loops = Loops.from_circuit(circuit)
    assert loops.branch_dict == {}
    assert loops.G.shape == (0, 1)
    assert capsys.readouterr().out.strip() == "No loops."


def test_default_loops_zero_capacitive_by_full_loop_support(capsys):
    circuit = harmonium()
    loops = Loops.from_circuit(circuit, zero_capacitive=True)
    for name, labels in loops.branch_dict.items():
        flux = loops.Phi_ext[loops.names.index(name)]
        assert flux == (0 if "C" in labels else sym.Symbol(f"Phi_{name}", real=True))
    assert "[capacitively broken]; flux = 0" in capsys.readouterr().out


def test_zero_capacitive_preserves_explicit_fluxes_and_includes_sources(capsys):
    circuit = coupled()
    baseline = Loops.from_circuit(circuit)
    loops = Loops.from_circuit(circuit, zero_capacitive=True)
    assert sum(flux != 0 for flux in baseline.Phi_ext) == 9
    assert sum(flux != 0 for flux in loops.Phi_ext) == 2
    name = next(name for name, flux in zip(loops.names, loops.Phi_ext) if flux == 0)
    supplied = {name: sym.Symbol("drive", real=True)}
    overridden = Loops(circuit, loops.branch_dict, supplied)
    assert overridden.Phi_ext[overridden.names.index(name)] == supplied[name]
    assert len(supplied) == 1  # caller's dictionary is unchanged


def test_inductor_capacitance_does_not_break_superconducting_loop(capsys):
    circuit = Circuit([Branch("J1", 1, 0), Branch("L1", 1, 0, L=1, C=1)], ground=0)
    loops = Loops.from_circuit(circuit, zero_capacitive=True)
    assert loops.variables == (sym.Symbol("Phi_L1", real=True),)
    assert "[superconducting]" in capsys.readouterr().out
