import pytest
import sympy as sym

from circuit_quantization import (
    Branch, Circuit, check_basis, decouple, lagrangian, mode_basis, residual_freedom, transform,
)
from circuit_quantization.modes import subspaces
from circuits import (
    COUPLED_ZERO_PI_TYPES, COUPLED_ZERO_PI_Z, HARMONIUM_Z, RHOMBUS_Z, ZERO_PI_Z,
    coupled, coupled_zero_pis, harmonium, kite, pos, rhombus, zero_pi,
)


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
    (("e", "e", "e", "e"), "not a valid 'e'"),  # the compact column lies in N(L*)
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


def test_coupled_example_given_Z_reproduces_schur_complement():
    Ct, Cc, Cf, Cg, CV1, CV2, CV3, CV4, CJ1, CJ2, CJ3 = pos(
        "C_t C_c C_f C_g C_V1 C_V2 C_V3 C_V4 C_J1 C_J2 C_J3")
    CT, C1, C2 = Ct + CJ1 + CJ2, Cc + CV1 + CV2, Cc + Cf + CJ3 + CV3 + CV4
    lag = lagrangian(coupled())
    Z = sym.Matrix([[1, 0, 1], [0, 1, 0], [0, 0, 1]])  # eq:fig1_Z
    check_basis(lag, Z, ("c", "e", "fr"))
    coords = decouple(lag, Z, ("c", "e", "fr"))
    expected = sym.Matrix([  # eq:fig1_schur
        [CT + C1 * Cg / (C1 + Cg), -Cc * Cg / (C1 + Cg)],
        [-Cc * Cg / (C1 + Cg), C2 - Cc**2 / (C1 + Cg)],
    ])
    assert zero(coords.C_tilde - expected)


def test_automatic_zero_pi_truncates_cyclic_mode():
    lag = lagrangian(zero_pi())
    Z, types = mode_basis(lag)
    assert sorted(types) == sorted(("c", "e", "h", "cy"))
    coords = decouple(lag, Z, types)
    assert coords.types == ("c", "e", "h")
    assert coords.Z.shape == (4, 3)


def test_zero_pi_given_Z_matches_notes():
    CJ, C, L = pos("C_J C L")
    lag = lagrangian(zero_pi())
    types = ("e", "c", "h", "cy")
    check_basis(lag, ZERO_PI_Z, types)
    coords = decouple(lag, ZERO_PI_Z, types)
    assert coords.C_tilde == sym.diag(2 * CJ, 2 * (C + CJ), 2 * C)  # eq:zp_transformed
    assert coords.Lstar_tilde == sym.diag(2 / L, 0, 2 / L)
    assert coords.omega_J_tilde.T == sym.Matrix([[-1, -1, 0], [-1, 1, 0]])


def test_harmonium_given_Z_promotes_harmonic_to_extended():
    lag = lagrangian(harmonium())
    check_basis(lag, HARMONIUM_Z, ("c", "e", "e", "e"))
    coords = decouple(lag, HARMONIUM_Z, ("c", "e", "e", "e"))
    assert coords.omega_J_tilde.T == sym.Matrix([[1, -1, -1, -1], [1, -1, 1, -1], [0, 1, 0, 0]])  # eq:harm_Ltilde
    assert coords.omega_L_tilde.T == sym.Matrix([[0, 0, 1, 1], [0, 0, -1, 1], [0, 1, 0, 0]])


def test_rhombus_given_Z_free_mode_already_decoupled():
    CJ, CB, Cg, alpha = pos("C_J C_B C_g alpha")
    lag = lagrangian(rhombus())
    types = ("c", "c", "c", "fr")
    check_basis(lag, RHOMBUS_Z, types)
    coords = decouple(lag, RHOMBUS_Z, types)
    assert coords.Z == RHOMBUS_Z[:, :3]  # M_C = 0 for the weighted centroid
    expected = sym.Matrix([  # eq:rh_transformed
        [CB + (1 + alpha) * CJ + 3 * Cg / 4, CB + alpha * CJ + Cg / 2, alpha * CJ + Cg / 4],
        [CB + alpha * CJ + Cg / 2, CB + (1 + alpha) * CJ + Cg, alpha * CJ + Cg / 2],
        [alpha * CJ + Cg / 4, alpha * CJ + Cg / 2, (1 + alpha) * CJ + 3 * Cg / 4],
    ])
    assert zero(coords.C_tilde - expected)
    assert coords.omega_J_tilde.T == sym.Matrix([[-1, 0, 0], [0, -1, 0], [0, 0, -1], [1, 1, 1]])


def test_mode_by_mode_decoupling_equals_block_formula():
    # Two floating transmons coupled by C_x: two free modes coupled to each other in C.
    Ca, Cb, Cx, C1, C2, C3, C4 = pos("C_a C_b C_x C_1 C_2 C_3 C_4")
    circuit = Circuit([
        Branch("Ja", 1, 2, C=Ca, EJ=1), Branch("Jb", 3, 4, C=Cb, EJ=1), Branch("Cx", 2, 3, C=Cx),
        Branch("C1", 1, 0, C=C1), Branch("C2", 2, 0, C=C2), Branch("C3", 3, 0, C=C3), Branch("C4", 4, 0, C=C4),
    ], ground=0)
    lag = lagrangian(circuit)
    Z, types = mode_basis(lag)
    assert types.count("fr") == 2
    coords = decouple(lag, Z, types)
    dyn = [j for j, t in enumerate(types) if t in ("c", "e", "h")]
    free = [j for j, t in enumerate(types) if t == "fr"]
    Z1, Z2 = Z[:, dyn], Z[:, free]
    M = -(Z2.T * lag.C_int * Z2).inv() * (Z2.T * lag.C_int * Z1)  # eq:decouple_final
    assert zero(coords.Z - (Z1 + Z2 * M))  # eq:ZY_blocks


def test_frozen_node_gives_series_inductance():
    CJ, La, Lb = pos("C_J L_a L_b")
    circuit = Circuit([Branch("J", 1, 0, C=CJ, EJ=1), Branch("La", 1, 2, L=La), Branch("Lb", 2, 0, L=Lb)], ground=0)
    lag = lagrangian(circuit)
    Z, types = mode_basis(lag)
    assert types == ("e", "fz")
    coords = decouple(lag, Z, types)
    assert zero(coords.Lstar_tilde - sym.Matrix([[1 / (La + Lb)]]))


def test_rhombus_decoupling_produces_weighted_centroid():
    # Appendix B.4: for unequal C_g the free-mode decoupling gives weights C_g_i / sum C_g.
    CJ, CB, EJ, alpha = pos("C_J C_B E_J alpha")
    Cg = pos("C_g1:5")
    circuit = Circuit([
        Branch("J1", 1, 2, C=CJ, EJ=EJ), Branch("J2", 2, 3, C=CJ, EJ=EJ), Branch("J3", 3, 4, C=CJ, EJ=EJ),
        Branch("J4", 4, 1, C=alpha * CJ, EJ=alpha * EJ), Branch("CB", 1, 3, C=CB),
        *(Branch(f"Cg{i}", i, 5, C=Cg[i - 1]) for i in range(1, 5)),
    ], ground=5)
    lag = lagrangian(circuit)
    phases = [[-1, 1, 0, 0], [0, -1, 1, 0], [0, 0, -1, 1]]
    weighted = sym.Matrix(phases + [[g / sum(Cg) for g in Cg]]).inv()  # eq:rh_Z
    plain = sym.Matrix(phases + [[sym.Rational(1, 4)] * 4]).inv()
    types = ("c", "c", "c", "fr")
    check_basis(lag, plain, types)
    assert zero(decouple(lag, plain, types).Z - weighted[:, :3])


def test_coupled_zero_pis_give_pure_theta_theta_coupling():
    # Zero-pi two-qubit gate paper, circuit (a): after removing Sigma_1, Sigma_2, varphi_+-,
    # the retained charging matrix couples only theta_1 and theta_2.
    CJ, C, C0, Cc = pos("C_J C C_0 C_c")
    lag = lagrangian(coupled_zero_pis())
    Z, types = mode_basis(lag)
    assert sorted(types) == sorted(COUPLED_ZERO_PI_TYPES)
    check_basis(lag, COUPLED_ZERO_PI_Z, COUPLED_ZERO_PI_TYPES)
    coords = decouple(lag, COUPLED_ZERO_PI_Z, COUPLED_ZERO_PI_TYPES)
    assert coords.types == ("c", "e", "h", "c", "e", "h")
    Ct = 2 * C + 2 * CJ + C0 + Cc  # tilde C_theta
    Cp = 2 * CJ + C0 + Cc  # tilde C_phi
    theta = sym.Matrix([[2 * Ct - Cc, Cc], [Cc, 2 * Ct - Cc]]) / (2 * Ct * (Ct - Cc))
    expected = sym.zeros(6, 6)
    expected[0, 0], expected[0, 3], expected[3, 0], expected[3, 3] = theta
    expected[1, 1] = expected[4, 4] = 1 / Cp
    expected[2, 2] = expected[5, 5] = 1 / (2 * C + C0 + Cc)
    assert zero(coords.C_tilde_inv - expected)


def test_residual_freedom_patterns():
    lag = lagrangian(kite())
    Z, types = mode_basis(lag)
    coords = decouple(lag, Z, types)  # types (c, e, h)
    a = residual_freedom(coords)
    b = residual_freedom(coords, keep_harmonic=False)
    nonzero = lambda S: [[int(x != 0) for x in S.row(i)] for i in range(3)]  # noqa: E731
    assert nonzero(a) == [[1, 1, 0], [0, 1, 0], [0, 1, 1]]  # S_{1,a}, eq:S1_structure
    assert nonzero(b) == [[1, 1, 0], [0, 1, 1], [0, 1, 1]]  # S_{1,b}


def test_transform_is_a_congruence_and_updates_types():
    lag = lagrangian(kite())
    Z, types = mode_basis(lag)
    coords = decouple(lag, Z, types)
    S = sym.Matrix([[1, 1, 0], [0, 1, 1], [0, 0, 1]])
    new = transform(coords, S)
    assert new.types == ("c", "e", "e")  # S_eh != 0 promotes harmonic to extended
    assert zero(new.C_tilde - S.T * coords.C_tilde * S)
    assert new.Z == coords.Z * S


@pytest.mark.parametrize("S, message", [
    (sym.Matrix([[1, 0, 0], [0, 1, 0], [1, 0, 1]]), "mixes 'h' into 'c'"),
    (sym.Matrix([[2, 0, 0], [0, 1, 0], [0, 0, 1]]), "GL"),
])
def test_transform_rejects_disallowed_mixing(S, message):
    lag = lagrangian(kite())
    coords = decouple(lag, *mode_basis(lag))
    with pytest.raises(ValueError, match=message):
        transform(coords, S)


def test_check_basis_rejects_harmonic_direction_labelled_extended():
    lag = lagrangian(kite())
    Z, _ = mode_basis(lag)  # third column is the harmonic direction
    with pytest.raises(ValueError, match="not a valid 'e'"):
        check_basis(lag, Z, ("c", "e", "e"))
    lc = Circuit([Branch("C", 1, 0, C=1), Branch("L", 1, 0, L=1)], ground=0)
    with pytest.raises(ValueError, match="not a valid 'e'"):
        check_basis(lagrangian(lc), sym.eye(1), ("e",))
