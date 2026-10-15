"""choose_S1 on circuit_quantization coordinates (the port's adapter, beyond sircuitenum's tests)."""

import sympy as sym

from circuit_quantization import Branch, Circuit, decouple, lagrangian, mode_basis, quantize, Loops, transform
from circuit_quantization.optimize import choose_S1


def zero_pi():
    """The symmetric zero-pi of tests/circuits.py (fig:zeropi_kite), floating."""
    CJ, C, L, EJ = sym.symbols("C_J C L E_J", positive=True)
    return Circuit([
        Branch("J1", 1, 2, C=CJ, EJ=EJ), Branch("J2", 3, 4, C=CJ, EJ=EJ),
        Branch("L1", 2, 3, L=L), Branch("L2", 4, 1, L=L),
        Branch("C1", 1, 3, C=C), Branch("C2", 2, 4, C=C),
    ])


ZERO_PI_Z = sym.Matrix([[-1, -1, 1, 1], [1, 1, 1, 1], [-1, 1, -1, 1], [1, -1, -1, 1]]) / 2  # eq:zp_Z


def test_choose_S1_follows_the_coordinates_column_order():
    # The notes' basis lists the extended column first; choose_S1 works internally in
    # (compact, extended, harmonic) order and must hand back S_1 in the caller's order.
    lag = lagrangian(zero_pi())
    noted = decouple(lag, ZERO_PI_Z, ("e", "c", "h", "cy"))
    automatic = decouple(lag, *mode_basis(lag), check=False)
    assert noted.types == ("e", "c", "h") and automatic.types == ("c", "e", "h")
    S_noted, h_noted = choose_S1(lag, noted)
    S_automatic, h_automatic = choose_S1(lag, automatic)
    assert h_noted == h_automatic
    from circuit_quantization.optimize import instance
    for S in S_noted:
        transform(lag, noted, instance(lag, noted, S))  # a valid residual transformation in the noted order


def test_choose_S1_feeds_result_transform():
    circuit = zero_pi()
    result = quantize(circuit, Loops.from_branch_dict(circuit, {
        "ring": {"J1": 1, "J2": 1, "L1": 1, "L2": 1},
        "t1": {"J1": 1, "L1": 1, "C1": -1},
        "t2": {"J2": 1, "L1": 1, "C2": -1},
    }, {"t1": 0, "t2": 0}))
    from circuit_quantization.optimize import choose_instance
    S1s, _ = choose_S1(result.lag, result.coords)
    S1, _ = choose_instance(result.lag, result.coords, S1s[0])
    best = result.transform(S1)
    assert best.coords.Z == result.coords.Z * S1
    assert best.B == result.B


# --- search orders (circuit_quantization.optimize.patterns) ---------------------------------


def test_maximally_compatible_sol_follows_the_cost_order_and_stops_at_the_first_solvable_level():
    from circuit_quantization.optimize.equationset import maximally_compatible_sol
    x, y = sym.symbols("x y")
    terms = [[x - 1], [x - 2], [y]]  # terms 0 and 1 conflict
    # Default: largest first, so both two-term sets that avoid the conflict.
    keys, _ = maximally_compatible_sol(terms, rational_only=True)
    assert keys == [(0, 2), (1, 2)]
    # A cost that prefers term 1 returns only its set, never reaching (0, 2).
    keys, _ = maximally_compatible_sol(terms, rational_only=True, patterns=[(0, (1, 2)), (1, (0, 2))])
    assert keys == [(1, 2)]
    # The unsolvable cheapest level is skipped for the next solvable one.
    keys, _ = maximally_compatible_sol(terms, rational_only=True, patterns=[(0, (0, 1)), (1, (0,)), (1, (2,))])
    assert keys == [(0,), (2,)]


def test_search_order_sees_named_coupling_terms_and_the_default_order_reproduces_sircuitenum():
    from circuit_quantization.optimize import most_zeros_first
    seen = []

    def spy(term_keys):
        seen.append(tuple(term_keys))
        return most_zeros_first(term_keys)

    lag = lagrangian(zero_pi())
    coords = decouple(lag, *mode_basis(lag), check=False)
    default = choose_S1(lag, coords)
    from circuit_quantization.optimize import Stage
    spied = choose_S1(lag, coords, stages=(Stage(square=("J",), order=spy), Stage(square=("L", "Cinv"), order=spy)))
    assert spied[1] == default[1] and [sym.Matrix(S) for S in spied[0]] == [sym.Matrix(S) for S in default[0]]
    pairs = [(0, 1), (0, 2), (1, 2)]  # three retained modes
    assert (("J",) + pairs[0]) in seen[0] and {k[0] for k in seen[0]} == {"J"}
    assert any(set(k[0] for k in keys) == {"L", "Cinv"} for keys in seen[1:])  # the linear stage, L then Cinv
    assert all(keys == tuple((name, *p) for name in dict.fromkeys(k[0] for k in keys) for p in pairs)
               for keys in seen)


def test_by_cost_with_the_default_cost_matches_most_zeros_first():
    from circuit_quantization.optimize import Stage, by_cost
    from sircuitenum_circuits import to_circuit
    order = by_cost(lambda keys: -len(keys))
    stages = (Stage(square=("J",), order=order), Stage(square=("L", "Cinv"), order=order))
    for circuit, edges in [([("J",), ("J",), ("L",), ("L",), ("C",), ("C",)], [(0, 1), (2, 3), (0, 3), (1, 2), (0, 2), (1, 3)]),
                           ([("L1", "C1"), ("L2", "C2"), ("L3",), ("J",)], [(1, 2), (3, 4), (1, 3), (2, 4)])]:
        lag = lagrangian(to_circuit(circuit, edges))
        coords = decouple(lag, *mode_basis(lag), check=False)
        default = choose_S1(lag, coords)
        costed = choose_S1(lag, coords, stages=stages)
        assert costed[1] == default[1]
        assert [sym.Matrix(S) for S in costed[0]] == [sym.Matrix(S) for S in default[0]]


# --- scores ------------------------------------------------------------------------------


def test_a_custom_score_runs_without_the_h_hash_shortcuts_and_agrees_with_it():
    # Wrapping h_hash takes the custom-score path (no support cache, no tiebreaker prefilter),
    # so equal results show those shortcuts never changed the answer.
    from circuit_quantization.optimize import h_hash
    from sircuitenum_circuits import to_circuit
    seen = []

    def wrapped(candidate):
        seen.append(candidate.instance)
        return h_hash(candidate)

    lag = lagrangian(to_circuit([("J",), ("J",), ("L",), ("L",), ("C",), ("C",)],
                                [(0, 1), (2, 3), (0, 3), (1, 2), (0, 2), (1, 3)]))
    coords = decouple(lag, *mode_basis(lag), check=False)
    default = choose_S1(lag, coords)
    custom = choose_S1(lag, coords, score=wrapped)
    assert custom[1] == default[1]
    assert [sym.Matrix(S) for S in custom[0]] == [sym.Matrix(S) for S in default[0]]
    assert seen and not any(seen)  # choose_S1 ranks symbolic candidates; instances come later


def test_a_score_can_break_h_hash_ties():
    # The junction triangle's three compact alignments tie under h_hash.
    from circuit_quantization.optimize import h_hash
    from sircuitenum_circuits import to_circuit
    lag = lagrangian(to_circuit([("J1",), ("J2",), ("J3",)], [(0, 1), (0, 2), (1, 2)]))
    coords = decouple(lag, *mode_basis(lag), check=False)
    tied, _ = choose_S1(lag, coords)
    assert len(tied) == 3
    broken, _ = choose_S1(lag, coords, score=lambda c: (h_hash(c), str(list(c.S))))
    assert len(broken) == 1 and sym.Matrix(broken[0]) in [sym.Matrix(S) for S in tied]


# --- rectangular, entry-wise terms (junction incidence) -----------------------------------


def _row_stage(lag, coords):
    """secondary_decouple on omega_J~^T alone, recording the term keys it offers."""
    from circuit_quantization.optimize import most_zeros_first
    from circuit_quantization.optimize.transforms import secondary_decouple
    order = [j for t in ("c", "e", "h") for j, k in enumerate(coords.types) if k == t]
    P = sym.eye(len(order))[:, order]
    var_types = {name: [i for i, j in enumerate(order) if coords.types[j] == t]
                 for name, t in (("compact", "c"), ("extended", "e"), ("harmonic", "h"))}
    var_types.update(free=[], frozen=[], sigma=[])
    seen = []

    def spy(keys):
        seen.append(list(keys))
        return most_zeros_first(keys)

    wT = sym.Matrix(coords.omega_J_tilde.T) * P
    Zs = secondary_decouple(var_types, [], [], row_mats=[wT], row_names=["wJ"], pattern_order=spy)
    return wT, var_types, Zs, seen


def test_junction_incidence_entries_are_decoupling_terms():
    # Three junctions on two extended modes: the junction rows have rank 2, so at most two of
    # them can each be made to involve a single mode.
    from sircuitenum_circuits import to_circuit
    from test_transforms import add_elem_number
    from circuit_quantization.optimize.transforms import _det_fast
    lag = lagrangian(to_circuit(add_elem_number([("J", "L"), ("C", "J"), ("C", "J", "L")]),
                                [(0, 1), (0, 2), (1, 2)]))
    coords = decouple(lag, *mode_basis(lag), check=False)
    wT, _, Zs, seen = _row_stage(lag, coords)
    assert seen == [[("wJ", k, j) for k in range(3) for j in range(2)]]
    assert len(Zs) == 1
    J = (wT * Zs[0]).applyfunc(sym.cancel)
    assert sum(x == 0 for x in J) == 2
    assert all(sum(x != 0 for x in J.row(k)) >= 1 for k in range(3))
    assert sym.simplify(_det_fast(Zs[0])) != 0


def test_junction_incidence_terms_skip_the_fixed_compact_columns():
    # Kite (c, e, h): the compact column is fixed by the alignment, so it offers no terms;
    # the harmonic column has no junction participation, so its entries start at zero.
    from circuit_quantization import Branch as B, Circuit as Cir
    CS, CJ1, CJ2, L1, L2, EJ1, EJ2 = sym.symbols("C_S C_J1 C_J2 L_1 L_2 E_J1 E_J2", positive=True)
    kite = Cir([B("CS", 1, 4, C=CS), B("J1", 1, 2, C=CJ1, EJ=EJ1), B("J2", 1, 3, C=CJ2, EJ=EJ2),
                B("L1", 2, 4, L=L1), B("L2", 3, 4, L=L2)], ground=4)
    lag = lagrangian(kite)
    coords = decouple(lag, *mode_basis(lag), check=False)
    assert coords.types == ("c", "e", "h")
    wT, var_types, Zs, seen = _row_stage(lag, coords)
    assert seen == [[("wJ", k, j) for k in range(2) for j in (1, 2)]]  # no column 0
    for Z in Zs:
        J = (wT * Z).applyfunc(sym.cancel)
        assert J[:, 0] == wT[:, 0]  # compact participation untouched
        assert J[:, 2] == sym.zeros(2, 1)  # still no junction on the harmonic mode


# --- stages --------------------------------------------------------------------------------


def _sircuitenum_coords(circuit, edges):
    from sircuitenum_circuits import to_circuit
    lag = lagrangian(to_circuit(circuit, edges))
    return lag, decouple(lag, *mode_basis(lag), check=False)


ZERO_PI_SE = ([("J",), ("J",), ("L",), ("L",), ("C",), ("C",)], [(0, 1), (2, 3), (0, 3), (1, 2), (0, 2), (1, 3)])
CHAIN_SE = ([("L1", "C1"), ("L2", "C2"), ("L3",), ("J",)], [(1, 2), (3, 4), (1, 3), (2, 4)])


def test_custom_stage_matrices_reproduce_the_builtin_ones():
    # A custom "flux" matrix transforms like L~*, a custom "charge" matrix like C~^{-1}.
    from circuit_quantization.optimize import Stage
    for circuit, edges in (ZERO_PI_SE, CHAIN_SE):
        lag, coords = _sircuitenum_coords(circuit, edges)
        default = choose_S1(lag, coords)
        custom = choose_S1(lag, coords, stages=(
            Stage(square=("J",)),
            Stage(square=(("L_mine", coords.Lstar_tilde), ("G_mine", coords.C_tilde_inv, "charge"))),
        ))
        assert custom[1] == default[1]
        assert [sym.Matrix(S) for S in custom[0]] == [sym.Matrix(S) for S in default[0]]


def test_a_junction_incidence_stage_localizes_junctions_and_reports_its_zeros():
    from circuit_quantization.optimize import Stage, h_hash
    from test_transforms import add_elem_number
    lag, coords = _sircuitenum_coords(add_elem_number([("J", "L"), ("C", "J"), ("C", "J", "L")]),
                                      [(0, 1), (0, 2), (1, 2)])
    zeros = []

    def score(candidate):
        zeros.append(candidate.zeros)
        return (-sum(key[0] == "wJ" for key in candidate.zeros), h_hash(candidate))

    S1s, _ = choose_S1(lag, coords, stages=(Stage(rows=("wJ",)), Stage(square=("L", "Cinv"))), score=score)
    J = (coords.omega_J_tilde.T * sym.Matrix(S1s[0])).applyfunc(sym.cancel)
    assert sum(x == 0 for x in J) == 2  # two of the rank-2 junction rows each on a single mode
    assert max(sum(key[0] == "wJ" for key in z) for z in zeros) == 2
    assert all(key[0] in {"wJ", "L", "Cinv"} for z in zeros for key in z)


def test_stage_matrices_are_checked():
    import pytest
    from circuit_quantization.optimize import Stage
    lag, coords = _sircuitenum_coords(*ZERO_PI_SE)
    n = len(coords.types)
    for stage, message in [
        (Stage(square=(("bad", sym.eye(n), "phase"),)), "law"),
        (Stage(square=(("bad", sym.Matrix(n, n, lambda i, j: i)),)), "symmetric"),
        (Stage(rows=(("bad", sym.ones(1, n + 1)),)), "one column per retained mode"),
        (Stage(square=("C",)), "Unknown stage matrix 'C'"),
    ]:
        with pytest.raises(ValueError, match=message):
            choose_S1(lag, coords, stages=(stage,))


# --- seeds, blocks, and instances ----------------------------------------------------------


def harmonium_automatic():
    """Harmonium (notes Appendix B.1) in the automatic basis, types (c, e, e, h)."""
    from circuits import harmonium
    lag = lagrangian(harmonium())
    return lag, decouple(lag, *mode_basis(lag), check=False)


def test_with_no_stages_the_seeds_themselves_are_ranked():
    from circuit_quantization.optimize import compact_alignments
    lag, coords = _sircuitenum_coords([("J1",), ("J2",), ("J3",)], [(0, 1), (0, 2), (1, 2)])
    seeds = [sym.Matrix(S) for S in compact_alignments(lag, coords)]
    S1s, _ = choose_S1(lag, coords, stages=())
    assert S1s and all(sym.Matrix(S) in seeds for S in S1s)


def test_seeds_must_be_valid_residual_transformations():
    import pytest
    lag, coords = _sircuitenum_coords(*ZERO_PI_SE)
    c = coords.types.index("c")
    e = coords.types.index("e")
    bad = sym.eye(len(coords.types))
    bad[e, c] = 1  # an extended component in the compact column
    with pytest.raises(ValueError):
        choose_S1(lag, coords, seeds=lambda lag, coords: [bad], stages=())


def test_a_seed_may_change_noncompact_types():
    # The compact charge shear turns harmonium's harmonic column extended (Sec 3.1).
    from circuit_quantization import compact_charge_decoupling, transform
    lag, coords = harmonium_automatic()
    assert coords.types == ("c", "e", "e", "h")
    S1s, _ = choose_S1(lag, coords, seeds=lambda lag, coords: [compact_charge_decoupling(coords)], stages=(),
                       score=lambda candidate: candidate.var_types["harmonic"])
    assert [sym.Matrix(S) for S in S1s] == [sym.Matrix(compact_charge_decoupling(coords))]
    assert transform(lag, coords, S1s[0]).types == ("c", "e", "e", "e")


def test_blocks_are_checked_and_no_blocks_opens_nothing():
    import pytest
    from circuit_quantization.optimize import compact_alignments
    lag, coords = _sircuitenum_coords(*ZERO_PI_SE)
    with pytest.raises(ValueError, match="compact into compact"):
        choose_S1(lag, coords, blocks={("c", "c")})
    seeds = [sym.Matrix(S) for S in compact_alignments(lag, coords)]
    S1s, _ = choose_S1(lag, coords, blocks=set())
    assert all(sym.Matrix(S) in seeds for S in S1s)  # nothing to solve for beyond the seeds


def test_instance_sets_free_entries_deterministically():
    from circuit_quantization import transform
    from circuit_quantization.optimize import instance
    lag, coords = _sircuitenum_coords(*ZERO_PI_SE)
    S1s, _ = choose_S1(lag, coords)
    family = sym.Matrix(S1s[0])
    assert any(str(x).startswith("Z") for x in family.free_symbols)  # free entries remain
    S = instance(lag, coords, family)
    assert not any(str(x).startswith("Z") for x in S.free_symbols)
    assert S == instance(lag, coords, family)  # deterministic
    transform(lag, coords, S)  # valid


def test_choose_instance_passes_concrete_S1_through_and_respects_max_weight():
    from circuit_quantization.optimize import choose_instance
    lag, coords = _sircuitenum_coords(*ZERO_PI_SE)
    S1s, _ = choose_S1(lag, coords)
    S, weight = choose_instance(lag, coords, S1s[0])
    assert choose_instance(lag, coords, S) == (S, weight)  # no free entries: returned as is
    assert choose_instance(lag, coords, S1s[0], max_weight=weight - 1) is None


def test_choose_instance_sets_entries_outside_every_junction_coefficient():
    # asymmetric 0-pi: the harmonic-into-extended entry is in no junction coefficient and not in det S_1
    from circuit_quantization.optimize import choose_instance
    from circuits import zero_pi_asymmetric
    lag = lagrangian(zero_pi_asymmetric())
    coords = decouple(lag, *mode_basis(lag))
    S1s, _ = choose_S1(lag, coords)
    S, weight = choose_instance(lag, coords, S1s[0])
    assert not S.free_symbols - coords.C_tilde.free_symbols - coords.Lstar_tilde.free_symbols
    assert weight == 3
    transform(lag, coords, S)


def test_h_hash_refuses_non_integer_junction_coefficients():
    import pytest
    from circuit_quantization import compact_charge_decoupling
    from circuit_quantization.optimize import candidate, h_hash
    lag, coords = harmonium_automatic()
    sheared = candidate(lag, coords, compact_charge_decoupling(coords), instance=True)
    with pytest.raises(ValueError, match="integer junction coefficients"):
        h_hash(sheared)
