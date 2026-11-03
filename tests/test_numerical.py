import numpy as np
import pytest
import sympy as sym

from circuit_quantization import (
    compact_charge_decoupling,
    cost_localization,
    cost_normal_modes,
    cost_targeted,
    normal_mode_decoupling,
    numerical_S1,
    numerify,
    parameterization,
    transform,
)
from circuits import (
    coupled,
    fluxonium_pair,
    harmonium,
    kite,
    random_circuit,
    transmon_fluxonium,
    zero_pi,
    zero_pi_asymmetric,
)
from test_invariants import build


def values_for(circuit, seed=0):
    rng = np.random.default_rng(seed)
    values = rng.integers(50, 300, len(circuit.variables))
    return {symbol: sym.Rational(int(value), 100)
            for symbol, value in zip(circuit.variables, values)}


def off_diagonal_scale(M, indices):
    block = M[np.ix_(indices, indices)]
    if len(indices) < 2:
        return 0.0
    off = block - np.diag(np.diag(block))
    return np.max(np.abs(off)) / np.max(np.abs(block))


def test_parameterization_uses_native_order_and_is_always_invertible():
    types = ("h", "c", "e", "c")
    p = parameterization(types, S_cc=np.array([[1, 2], [0, 1]]), relaxed=True)
    assert p.size == 8
    assert np.array_equal(p.S(p.x0())[np.ix_((1, 3), (1, 3))], [[1, 2], [0, 1]])
    rng = np.random.default_rng(4)
    for _ in range(10):
        assert np.linalg.det(p.S(rng.normal(size=p.size))) != 0


def test_numerical_search_handles_a_fixed_all_compact_parameterization():
    from circuits import rhombus

    circuit = rhombus()
    lag, _, coords, *_ = build(circuit)
    S, info = numerical_S1(
        lag, coords, values_for(circuit), cost_normal_modes, relaxed=True, restarts=0,
    )
    assert S == sym.eye(len(coords.types))
    assert info.result.success


def test_cost_constructors_are_scale_invariant_and_masks_are_respected():
    C = np.array([[2.0, 0.3], [0.3, 1.0]])
    L = np.array([[1.0, 0.2], [0.2, 3.0]])
    wJ = np.array([[1.0, -2.0]])
    types = ("e", "e")
    identity = np.eye(2)
    rescaled = np.diag([3.0, 0.25])

    normal = cost_normal_modes(C, L, types)
    localization = cost_localization(C, L, wJ, types)
    assert normal(identity) == pytest.approx(normal(rescaled))
    assert localization(identity) == pytest.approx(localization(rescaled))

    none = np.zeros((2, 2), dtype=bool)
    only_C = np.array([[False, True], [True, False]])
    targeted = cost_targeted(C, L, types, (only_C, none))
    C_inv = np.linalg.inv(C)
    expected = C_inv[0, 1] ** 2 / (C_inv[0, 0] * C_inv[1, 1])
    assert targeted(identity) == pytest.approx(expected)


BUILDERS = [
    harmonium,
    zero_pi,
    zero_pi_asymmetric,
    kite,
    fluxonium_pair,
    transmon_fluxonium,
    coupled,
    lambda: random_circuit(0),
    lambda: random_circuit(2),
]


@pytest.mark.parametrize("build_circuit", BUILDERS)
def test_numerical_normal_modes_match_analytic(build_circuit):
    circuit = build_circuit()
    lag, _, coords, *_ = build(circuit)
    noncompact = [j for j, kind in enumerate(coords.types) if kind != "c"]
    if len(noncompact) < 2:
        pytest.skip("nothing to diagonalize")
    values = values_for(circuit)

    _, C_ref, L_ref = normal_mode_decoupling(lag, coords, values)
    S_num, info = numerical_S1(
        lag, coords, values, cost_normal_modes, relaxed=True, restarts=3, seed=0,
    )
    numeric = numerify(coords, values)
    C = np.asarray(numeric.C_tilde, dtype=float)
    L = np.asarray(numeric.Lstar_tilde, dtype=float)
    S = np.asarray(S_num, dtype=float)
    Cp, Lp = S.T @ C @ S, S.T @ L @ S
    Cp_inv = np.linalg.solve(Cp, np.eye(Cp.shape[0]))

    assert off_diagonal_scale(Cp_inv, range(len(coords.types))) < 1e-7
    assert off_diagonal_scale(Lp, noncompact) < 1e-7
    assert info.cost < 1e-12

    lam_num = np.sort(np.diag(Lp)[noncompact] / np.diag(Cp)[noncompact])
    C_ref = np.asarray(C_ref, dtype=float)
    L_ref = np.asarray(L_ref, dtype=float)
    lam_ref = np.sort(np.diag(L_ref)[noncompact] / np.diag(C_ref)[noncompact])
    assert np.allclose(lam_num, lam_ref, rtol=1e-6, atol=1e-10)

    # Each numerical column solves the same full generalized eigenproblem.  The
    # compact row is part of this check and therefore also verifies the shear.
    for j in noncompact:
        vector = S[:, j]
        eigenvalue = Lp[j, j] / Cp[j, j]
        residual = L @ vector - eigenvalue * C @ vector
        scale = np.linalg.norm(L @ vector) + abs(eigenvalue) * np.linalg.norm(C @ vector)
        # This residual combines both optimized forms, so it can be a small
        # multiple of either individual off-diagonal tolerance.
        assert np.linalg.norm(residual) / scale < 3e-7

    # The exact compact block and numerical real blocks pass package validation.
    transform(lag, coords, S_num)


def test_numerical_and_analytic_columns_agree_for_distinct_modes():
    circuit = fluxonium_pair()
    lag, _, coords, *_ = build(circuit)
    values = values_for(circuit, seed=2)
    S_ref, _, _ = normal_mode_decoupling(lag, coords, values)
    S_num, _ = numerical_S1(
        lag, coords, values, cost_normal_modes, relaxed=True, restarts=3, seed=2,
    )
    C = np.asarray(numerify(coords, values).C_tilde, dtype=float)
    reference = np.asarray(S_ref, dtype=float)
    numerical = np.asarray(S_num, dtype=float)

    # Both are C-normalized.  Sort by Rayleigh quotient, then compare up to sign.
    L = np.asarray(numerify(coords, values).Lstar_tilde, dtype=float)
    order_ref = np.argsort(np.diag(reference.T @ L @ reference))
    order_num = np.argsort(np.diag(numerical.T @ L @ numerical))
    reference, numerical = reference[:, order_ref], numerical[:, order_num]
    assert np.allclose(reference.T @ C @ reference, np.eye(2), atol=1e-10)
    assert np.allclose(numerical.T @ C @ numerical, np.eye(2), atol=1e-7)
    assert np.allclose(np.abs(reference), np.abs(numerical), atol=2e-6)


def test_compact_shear_is_contained_in_the_numerical_solution():
    circuit = zero_pi_asymmetric()
    lag, _, coords, *_ = build(circuit)
    values = values_for(circuit, seed=5)
    S_num, _ = numerical_S1(
        lag, coords, values, cost_normal_modes, relaxed=True, restarts=3, seed=5,
    )
    S_shear = np.asarray(compact_charge_decoupling(coords).subs(values), dtype=float)
    relative = np.linalg.solve(S_shear, np.asarray(S_num, dtype=float))
    compact = [j for j, kind in enumerate(coords.types) if kind == "c"]
    noncompact = [j for j, kind in enumerate(coords.types) if kind != "c"]
    assert np.max(np.abs(relative[np.ix_(compact, noncompact)])) < 1e-7
