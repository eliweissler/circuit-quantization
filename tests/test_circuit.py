import pytest
import sympy as sym

from circuit_quantization import Branch, Circuit, Loops, default_tree, incidence, tree_to_G
from circuits import HARMONIUM_CYCLES, coupled, harmonium, zero_pi

C, EJ, L = sym.symbols("C E_J L", positive=True)


@pytest.mark.parametrize("kwargs, message", [
    (dict(C=0, EJ=EJ), "needs C > 0"),
    (dict(C=C, L=L, EJ=EJ), "at most one"),
    (dict(C=C, V=1), "may carry only V"),
    (dict(C=-1), "C must be 0 or positive"),
    (dict(C=C, L=-2), "L must be positive"),
    (dict(C=C, EJ=0), "EJ must be positive"),
])
def test_branch_validation(kwargs, message):
    with pytest.raises(ValueError, match=message):
        Branch("b", 1, 0, **kwargs)


def test_branch_accepts_symbolic_values_of_unknown_sign():
    eps, x = sym.Symbol("epsilon", real=True), sym.Symbol("x")
    junction = Branch("J1", 1, 0, C=C * (1 + eps), EJ=EJ * (1 - eps))
    assert junction.C == C * (1 + eps) and junction.EJ == EJ * (1 - eps)
    assert Branch("b", 1, 0, C=x).C == x
    assert Branch("L1", 1, 0, L=-L).L == -L  # provably negative, but symbolic


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
    loops = Loops.from_branch_dict(circuit, {"k": {"J1": 1, "J2": -1}, "cb": {"C1": 1, "J1": -1}}, {"cb": 0})
    assert loops.variables == (sym.Symbol("Phi_k", real=True),)
    assert loops.rebase(-loops.G).variables == loops.variables
    assert Loops.from_branch_dict(circuit, dict(zip(loops.names, ({"J1": 1, "J2": -1}, {"C1": 1, "J1": -1}))),
                 {"k": 0, "cb": 0}).variables == ()


def test_automatic_symbols_quantize_and_export():
    from circuit_quantization import quantize, scqubits_yaml
    circuit = Circuit([Branch("J1", 1, 0, energy_symbols=True), Branch("L1", 1, 0, energy_symbols=True)], ground=0)
    loops = Loops.from_tree(circuit, default_tree(circuit), {"L1": 0})
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


def test_isolated_ground_is_rejected():
    with pytest.raises(ValueError, match="ground=None"):
        Circuit([Branch("C1", 1, 2)], ground=0)
    assert Circuit([Branch("C1", 1, 2)]).nodes == (1, 2)


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
    loops = Loops.from_tree(circuit, default_tree(circuit))
    assert len(loops.names) == 9
    assert loops.G * incidence(circuit).T == sym.zeros(9, 7)  # eq:G_OmegaT
    assert set(loops.G) <= {-1, 0, 1}
    kinds = [b.type for b in circuit.branches]
    superconducting = [all(kinds[k] in "JL" for k in range(15) if loops.G[i, k]) for i in range(9)]
    assert superconducting == [True, True] + [False] * 7  # eq:G_blocks
    # One capacitively broken loop closes through a source branch.
    assert any(loops.G[i, circuit.branch_names.index("V1")] for i in range(9))


def test_cycles_from_explicit_tree():
    circuit = harmonium()
    loops = Loops.from_tree(circuit, ["C", "J1", "J2", "J3"])
    assert loops.names == loops.closure_branches == ("L1", "L2", "L3")
    assert loops.branch_dict["L3"] == {"L3": 1, "J3": -1}
    G = loops.G
    assert (G[2, circuit.branch_names.index("J3")], G[2, circuit.branch_names.index("L3")]) == (-1, 1)


def test_floating_circuit_cycle_count():
    # zero-pi is floating: N_ell = N_b - (N - 1) = 6 - 3.
    assert tree_to_G(zero_pi(), default_tree(zero_pi()))[1].rows == 3


def test_loops_flux_defaults_are_visible():
    circuit = harmonium()
    loops = Loops.from_branch_dict(circuit, HARMONIUM_CYCLES, {"cb": 0})
    Phi_k, Phi_f = sym.symbols("Phi_k Phi_f", real=True)
    assert loops.Phi_ext == sym.Matrix([Phi_k, Phi_f, 0])


def test_loops_rejects_incomplete_or_invalid_bases():
    circuit = harmonium()
    with pytest.raises(ValueError, match="N_ell = N_b - rank"):
        Loops.from_branch_dict(circuit, {k: HARMONIUM_CYCLES[k] for k in ("k", "f")})
    with pytest.raises(ValueError, match="G Omega\\^T = 0"):
        Loops.from_branch_dict(circuit, {**HARMONIUM_CYCLES, "k": {"J1": 1}})
    with pytest.raises(ValueError, match="independent rows"):
        Loops.from_branch_dict(circuit, {**HARMONIUM_CYCLES, "cb": HARMONIUM_CYCLES["k"]})


def test_rebase_maps_fluxes_and_rejects_non_unimodular():
    circuit = harmonium()
    loops = Loops.from_branch_dict(circuit, HARMONIUM_CYCLES, {"cb": 0})
    T = sym.Matrix([[1, 0, 0], [1, 1, 0], [0, 0, 1]])
    rebased = loops.rebase(T * loops.G)
    assert rebased.Phi_ext == T * loops.Phi_ext
    with pytest.raises(ValueError, match="physical loop"):
        loops.rebase(2 * loops.G)
    # The three squares of K4 form a valid {0, +-1} basis with det T = 2 (eq:G_basis_change).
    from itertools import combinations
    from circuit_quantization import cycle_bases
    K4 = Circuit([Branch(f"C{u}{v}", u, v, C=1) for u, v in combinations(range(4), 2)], ground=0)
    squares = next(b for b in cycle_bases(K4, primitive=False) if b not in list(cycle_bases(K4)))
    with pytest.raises(ValueError, match="det"):
        Loops.from_tree(K4, default_tree(K4)).rebase(Loops.from_branch_dict(K4, squares).G)


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
        Loops.from_branch_dict(harmonium(), {**HARMONIUM_CYCLES, "k": cycle})


def test_rebase_requires_one_name_per_row():
    loops = Loops.from_branch_dict(harmonium(), HARMONIUM_CYCLES, {"cb": 0})
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
        cycles = Loops.from_tree(circuit, tree)
        assert cycles.G * incidence(circuit).T == sym.zeros(3, 5)


def test_spanning_trees_preserve_parallel_branch_choices_in_a_floating_circuit():
    from circuit_quantization import spanning_trees
    circuit = Circuit([
        Branch("a", 1, 2, C=1), Branch("b", 2, 1, C=1),
        Branch("c", 2, 3, C=1), Branch("d", 2, 3, C=1),
    ])
    assert sorted(spanning_trees(circuit)) == [("a", "c"), ("a", "d"), ("b", "c"), ("b", "d")]
    for tree in spanning_trees(circuit):
        assert Loops.from_tree(circuit, tree).G.shape == (2, 4)


def test_cycle_bases_expand_parallel_branches():
    from circuit_quantization import cycle_bases
    circuit = Circuit([
        Branch("a", 1, 2, C=1), Branch("b", 2, 1, C=1),
        Branch("c", 2, 3, C=1), Branch("d", 3, 1, C=1),
    ])
    assert list(cycle_bases(circuit)) == [
        {"loop_1": ("a", "b"), "loop_2": ("a", "c", "d")},
        {"loop_1": ("a", "b"), "loop_2": ("b", "c", "d")},
        {"loop_1": ("a", "c", "d"), "loop_2": ("b", "c", "d")},
    ]


def test_cycle_bases_primitive_excludes_the_three_squares_of_K4():
    from itertools import combinations
    from circuit_quantization import cycle_bases
    circuit = Circuit([Branch(f"C{u}{v}", u, v, C=1) for u, v in combinations(range(4), 2)], ground=0)
    primitive, full = list(cycle_bases(circuit)), list(cycle_bases(circuit, primitive=False))
    # Seven simple loops (four triangles, three squares); 29 of the 35 triples span the cycle space.
    assert (len(primitive), len(full)) == (28, 29)
    squares = [basis for basis in full if basis not in primitive]
    assert [len(loop) for loop in squares[0].values()] == [4, 4, 4]  # the squares sum to twice the cycle space
    for basis in full:
        assert Loops.from_branch_dict(circuit, basis).G.rows == 3


def test_different_spanning_trees_can_give_the_same_cycle_matrix():
    from circuit_quantization import spanning_trees
    circuit = Circuit([
        Branch("a", 0, 1, C=1), Branch("b", 1, 2, C=1), Branch("c", 2, 0, C=1),
    ], ground=0)
    trees = list(spanning_trees(circuit))
    assert len(trees) == 3
    for tree in trees:
        closure, G = tree_to_G(circuit, tree)
        assert G == sym.Matrix([[1, 1, 1]])
        assert set(tree) == set(circuit.branch_names) - set(closure)


def test_loop_tuples_infer_orientation_and_roundtrip():
    circuit = harmonium()
    labels = {"k": ("J1", "J2", "L1", "L2"), "f": ("J3", "L3"),
              "cb": ("C", "J1", "J3", "L1")}
    loops = Loops.from_branch_dict(circuit, labels, {"cb": 0})
    expected = Loops.from_branch_dict(circuit, HARMONIUM_CYCLES, {"cb": 0})
    assert loops.G == expected.G
    assert {k: set(v) for k, v in loops.branch_dict.items()} == {k: set(v) for k, v in labels.items()}  # order lives in G
    assert Loops.from_branch_dict(circuit, loops.branch_dict).G == loops.G
    assert loops.G * incidence(circuit).T == sym.zeros(3, 5)
    flipped = loops.rebase(-loops.G)
    assert Loops.from_branch_dict(circuit, flipped.branch_dict).G == flipped.G
    reversed_f = Loops.from_branch_dict(circuit, {**labels, "f": ("L3", "J3")})
    assert reversed_f.G[1, :] == -loops.G[1, :]


def test_default_loop_constructor_prints_closure_first_tuples(capsys):
    circuit = harmonium()
    loops = Loops.from_circuit(circuit)
    repeated = Loops.from_circuit(circuit)
    assert repeated.branch_dict == loops.branch_dict
    assert repeated.G == loops.G
    assert loops.G == Loops.from_tree(circuit, default_tree(circuit)).G
    assert all(next(iter(labels)) == name for name, labels in loops.branch_dict.items())
    output = capsys.readouterr().out
    for name in loops.names:
        assert f"{name}: +{name}" in output
        assert f"flux = Phi_{name}" in output
    assert Loops.from_branch_dict(circuit, loops.branch_dict).G == loops.G


@pytest.mark.parametrize("labels, message", [
    (("J1", "J1"), "distinct"), ((), "nonempty"),
    (("J1", "unknown"), "unknown branches"),
    (("J1", "L1"), "single closed loop"),
    (("J1", "J2", "L1", "L2", "J3"), "single closed loop"),
    (("J1", "J2", "L1", "L2", "J3", "L3"), "single closed loop"),
])
def test_loop_tuples_reject_invalid_support(labels, message):
    with pytest.raises(ValueError, match=message):
        Loops.from_branch_dict(harmonium(), {"bad": labels})


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
    overridden = Loops.from_branch_dict(circuit, loops.branch_dict, supplied)
    assert overridden.Phi_ext[overridden.names.index(name)] == supplied[name]
    assert len(supplied) == 1  # caller's dictionary is unchanged


def test_inductor_capacitance_does_not_break_superconducting_loop(capsys):
    circuit = Circuit([Branch("J1", 1, 0), Branch("L1", 1, 0, L=1, C=1)], ground=0)
    loops = Loops.from_circuit(circuit, zero_capacitive=True)
    assert loops.variables == (sym.Symbol("Phi_L1", real=True),)
    assert "[superconducting]" in capsys.readouterr().out


def test_closure_branches_follow_tree_and_survive_only_identity_rebases():
    circuit = harmonium()
    loops = Loops.from_tree(circuit, default_tree(circuit))
    assert loops.closure_branches == loops.names
    assert loops.rebase(loops.G, names=("a", "b", "c")).closure_branches == loops.closure_branches
    assert loops.rebase(-loops.G).closure_branches is None
    assert Loops(circuit, loops.G).closure_branches is None
    with pytest.raises(ValueError, match="identity columns"):
        Loops(circuit, loops.G, closure_branches=loops.closure_branches[::-1])


def test_branch_dict_roundtrip_preserves_an_all_negative_loop_orientation():
    # A directed triangle admits either overall sign (eq:G_OmegaT).
    # In particular, a valid loop need not contain a positive branch at all.
    circuit = Circuit([
        Branch("a", 0, 1, C=1), Branch("b", 1, 2, C=1), Branch("c", 2, 0, C=1),
    ], ground=0)
    flux = sym.Symbol("Phi_ring", real=True)
    forward = Loops.from_branch_dict(circuit, {"ring": ("a", "b", "c")}, {"ring": flux})
    reverse = forward.rebase(-forward.G, names=("ring",))
    assert reverse.G == sym.Matrix([[-1, -1, -1]])
    assert reverse.Phi_ext == sym.Matrix([-flux])  # eq:G_basis_change
    rebuilt = Loops.from_branch_dict(circuit, reverse.branch_dict, {"ring": -flux})
    # branch_dict promises this exact roundtrip; flipping G while keeping the
    # supplied flux reverses the physical flux rather than merely relabeling it.
    assert rebuilt.G == reverse.G
    assert rebuilt.Phi_ext == reverse.Phi_ext
