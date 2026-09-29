"""Cases adapted from sircuitenum/tests/test_quantize.py (GPL-3.0).

Source checkout: eb67c34. Ported test_var_trans_basis, the singular-Z case,
test_decouple_column, and the automatic/explicit test_symbolic_hamiltonian case.
No runtime dependency on sircuitenum. An isolated reference ground retains its
floating-node convention; sigma maps to cy. Assert physical invariants rather
than a particular basis ordering, Hamiltonian hash, or LaTeX representation.
"""

from collections import Counter

import pytest
import sympy as sym

import circuit_quantization as cq
from circuit_quantization.modes import _decouple_column


# Element strings denote parallel elements on an edge; J carries intrinsic C_J.
CASES = [
    ("extended_triangle", ["J", "JL", "CJL"], [(0, 1), (0, 2), (1, 2)], "e e cy"),
    ("formerly_singular", ["J", "J", "CL", "L"], [(0, 2), (0, 3), (1, 3), (2, 3)], None),
    ("compact_harmonic_frozen", ["CL", "L", "J", "L"], [(0, 2), (0, 3), (1, 3), (2, 3)], "c h fz cy"),
    ("junction_star", ["J"] * 3, [(0, 3), (1, 3), (2, 3)], "c c c cy"),
    ("junction_triangle", ["J"] * 3, [(0, 1), (0, 2), (1, 2)], "c c cy"),
    ("bifluxon", ["J", "J", "L"], [(1, 2), (1, 3), (2, 3)], "c e cy"),
    ("bifluxon_with_oscillator", ["J", "J", "L", "CL"], [(1, 2), (1, 3), (2, 3), (1, 4)], "c e h cy"),
    ("two_compact_one_harmonic", ["C", "L", "J", "J"], [(1, 2), (1, 3), (2, 3), (1, 4)], "c c h cy"),
    ("transmon_molecule", ["J", "J", "C", "C"], [(1, 2), (3, 4), (1, 3), (2, 4)], "c c fr cy"),
    ("zero_pi", ["J", "J", "L", "L", "C", "C"], [(1, 2), (3, 4), (1, 4), (2, 3), (1, 3), (2, 4)], "c e h cy"),
    ("fully_linear", ["L", "L", "C", "C"], [(1, 2), (3, 4), (1, 3), (2, 4)], "h fr fz cy"),
]


def circuit_from_edges(elements, edges, equal):
    branches = []
    for i, (kinds, (tail, head)) in enumerate(zip(elements, edges)):
        def value(kind):
            return sym.Symbol(kind if equal else f"{kind}_{i}", positive=True)
        for kind in kinds:
            params = {"C": {"C": value("C")},
                      "J": {"C": value("CJ"), "EJ": value("EJ")},
                      "L": {"L": value("L")}}[kind]
            branches.append(cq.Branch(f"{kind}{i}", tail, head, **params))
    return cq.Circuit(branches, ground="reference")


def zero(matrix):
    return matrix.applyfunc(sym.cancel).is_zero_matrix


@pytest.mark.parametrize("equal", [False, True], ids=["distinct", "equal"])
@pytest.mark.parametrize("name,elements,edges,expected", CASES, ids=[case[0] for case in CASES])
def test_mode_basis_and_decoupling(name, elements, edges, expected, equal):
    lag = cq.lagrangian(circuit_from_edges(elements, edges, equal))
    Z, types = cq.mode_basis(lag)
    if expected is not None:
        assert Counter(types) == Counter(expected.split())
    assert Z.det() != 0
    assert not Z.has(sym.nan, sym.zoo, sym.oo, -sym.oo)
    cq.check_basis(lag, Z, types)
    coords = cq.decouple(lag, Z, types)
    cq.check_coordinates(lag, coords)
    assert zero(coords.C_tilde * coords.C_tilde_inv - sym.eye(coords.Z.cols))
    free = Z[:, [i for i, t in enumerate(types) if t in ("fr", "cy")]]
    frozen = Z[:, [i for i, t in enumerate(types) if t in ("fz", "cy")]]
    assert zero(coords.Z.T * lag.C_int * free)
    assert zero(coords.Z.T * lag.Lstar_int * frozen)
    assert zero(lag.omega_J_int.T * free)


@pytest.mark.parametrize("equal", [False, True], ids=["distinct", "equal"])
def test_transmon_molecule_decouple_column(equal):
    circuit = circuit_from_edges(["J", "J", "C", "C"], [(1, 2), (3, 4), (1, 3), (2, 4)], equal)
    lag = cq.lagrangian(circuit)
    half = sym.Rational(1, 2)
    Z = sym.Matrix([[half, 0, 1, 1], [-half, 0, 1, 1], [0, half, 0, 1], [0, -half, 0, 1]])
    discarded = Z[:, 2:]
    assert (discarded.T * lag.C_int * discarded).det() == 0
    retained = sym.Matrix.hstack(*(_decouple_column(Z[:, i], discarded, lag.C_int) for i in range(2)))
    assert zero(retained.T * lag.C_int * discarded)
    assert zero(lag.omega_J_int.T * (retained - Z[:, :2]))


def test_automatic_and_explicit_hamiltonian_agree():
    circuit = circuit_from_edges(["CL", "JL"], [(0, 2), (1, 2)], equal=False)
    loops = cq.Loops(circuit, cq.fundamental_cycles(circuit))
    automatic = cq.quantize(circuit, loops)
    Z, types = cq.mode_basis(automatic.lag)
    explicit = cq.quantize(circuit, loops, Z, types)
    assert sym.simplify(automatic.H - explicit.H) == 0
