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
    for S in S_noted:
        transform(lag, noted, S)  # a valid residual transformation in the noted column order


def test_choose_S1_feeds_result_transform():
    circuit = zero_pi()
    result = quantize(circuit, Loops.from_branch_dict(circuit, {
        "ring": {"J1": 1, "J2": 1, "L1": 1, "L2": 1},
        "t1": {"J1": 1, "L1": 1, "C1": -1},
        "t2": {"J2": 1, "L1": 1, "C2": -1},
    }, {"t1": 0, "t2": 0}))
    S1s, _ = choose_S1(result.lag, result.coords)
    best = result.transform(S1s[0])
    assert best.coords.Z == result.coords.Z * S1s[0]
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
    spied = choose_S1(lag, coords, pattern_order=spy)
    assert spied[1] == default[1] and [sym.Matrix(S) for S in spied[0]] == [sym.Matrix(S) for S in default[0]]
    pairs = [(0, 1), (0, 2), (1, 2)]  # three retained modes
    assert (("J",) + pairs[0]) in seen[0] and {k[0] for k in seen[0]} == {"J"}
    assert any(set(k[0] for k in keys) == {"L", "C"} for keys in seen[1:])  # the linear stage, L then C
    assert all(keys == tuple((name, *p) for name in dict.fromkeys(k[0] for k in keys) for p in pairs)
               for keys in seen)


def test_by_cost_with_the_default_cost_matches_most_zeros_first():
    from circuit_quantization.optimize import by_cost
    from sircuitenum_circuits import to_circuit
    for circuit, edges in [([("J",), ("J",), ("L",), ("L",), ("C",), ("C",)], [(0, 1), (2, 3), (0, 3), (1, 2), (0, 2), (1, 3)]),
                           ([("L1", "C1"), ("L2", "C2"), ("L3",), ("J",)], [(1, 2), (3, 4), (1, 3), (2, 4)])]:
        lag = lagrangian(to_circuit(circuit, edges))
        coords = decouple(lag, *mode_basis(lag), check=False)
        default = choose_S1(lag, coords, return_instance=False)
        costed = choose_S1(lag, coords, return_instance=False, pattern_order=by_cost(lambda keys: -len(keys)))
        assert costed[1] == default[1]
        assert [sym.Matrix(S) for S in costed[0]] == [sym.Matrix(S) for S in default[0]]


# --- scores ------------------------------------------------------------------------------


def test_a_custom_score_runs_without_the_h_hash_shortcuts_and_agrees_with_it():
    # Wrapping h_hash takes the custom-score path (no support cache, no z3 a-priori bound),
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
    assert False in seen and True in seen  # ranked symbolic candidates, then instances


def test_a_score_can_break_h_hash_ties():
    # The junction triangle's three compact alignments tie under h_hash.
    from circuit_quantization.optimize import h_hash
    from sircuitenum_circuits import to_circuit
    lag = lagrangian(to_circuit([("J1",), ("J2",), ("J3",)], [(0, 1), (0, 2), (1, 2)]))
    coords = decouple(lag, *mode_basis(lag), check=False)
    tied, _ = choose_S1(lag, coords, return_instance=False)
    assert len(tied) == 3
    broken, _ = choose_S1(lag, coords, return_instance=False, score=lambda c: (h_hash(c), str(list(c.S))))
    assert len(broken) == 1 and sym.Matrix(broken[0]) in [sym.Matrix(S) for S in tied]
