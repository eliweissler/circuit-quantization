import pytest
import sympy as sym

from circuit_quantization import check_basis, lagrangian, mode_basis
from circuit_quantization.modes import subspaces
from circuits import HARMONIUM_Z, coupled, harmonium, kite


def zero(M):
    return sym.Matrix(M).applyfunc(sym.simplify).is_zero_matrix


def same_span(A, B):
    return sym.Matrix(A).rank() == sym.Matrix(B).rank() == sym.Matrix.hstack(A, B).rank()


def test_coupled_example_subspaces():
    sub = subspaces(lagrangian(coupled()))
    assert same_span(sub["J"], sym.Matrix([1, 0, 1]))  # eq:fig1_nullspaces
    assert same_span(sub["L"], sym.Matrix([[1, 0], [0, 0], [0, 1]]))
    assert sub["C"].cols == 0
    assert [sub[t].cols for t in ("c", "e", "h", "fr", "fz", "cy")] == [1, 1, 0, 1, 0, 0]


def test_automatic_kite_basis_separates_the_harmonic_direction():
    lag = lagrangian(kite())
    Z, types = mode_basis(lag)
    assert types == ("c", "e", "h")
    check_basis(lag, Z, types)
    omega = lag.omega_J_int.T * Z
    assert omega[:, 0] == sym.Matrix([1, 1])  # compact period Phi0 (eq:Scc_selection)
    assert omega[:, 2] == sym.zeros(2, 1)  # harmonic column sees no junction


@pytest.mark.parametrize("types, message", [
    (("e", "e", "e", "e"), "1 'c' modes"),
    (("h", "e", "e", "e"), "not a valid 'h'"),
])
def test_check_basis_rejects_wrong_types(types, message):
    with pytest.raises(ValueError, match=message):
        check_basis(lagrangian(harmonium()), HARMONIUM_Z, types)


def test_check_basis_rejects_non_primitive_compact_lattice():
    Z = HARMONIUM_Z.copy()
    Z[:, 0] = Z[:, 0] / 2
    with pytest.raises(ValueError, match="integer|primitive"):
        check_basis(lagrangian(harmonium()), Z, ("c", "e", "e", "e"))
