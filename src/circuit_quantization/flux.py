"""Irrotational external-flux assignment (Sec V)."""

import sympy as sym


def irrotational_B(lag, coords, loops):
    """B = (I - Omega_int^T Z C~^{-1} Z^T Omega_int C_b) G^+, eq:B_irrot_direct_closed with W = 0.

    Evaluated as B = G^+ - Omega_int^T Z X with X = C~^{-1} Z^T Omega_int C_b G^+,
    so the only rational-function matrix built is the small N_d x N_ell block X.
    """
    G = sym.Matrix(loops.G)
    N_b = lag.Omega.cols
    if G.rows == 0:
        return sym.ImmutableMatrix(sym.zeros(N_b, 0))
    G_plus = G.T * (G * G.T).inv()  # eq:G_right_inverse
    Z, Omega_int = sym.Matrix(coords.Z), sym.Matrix(lag.Omega_int)
    R = Z.T * Omega_int * lag.C_b * G_plus  # EMF drive of G^+ on the retained modes
    X = (coords.C_tilde_inv * R).applyfunc(sym.cancel)
    B = (G_plus - Omega_int.T * Z * X).applyfunc(sym.cancel)
    # G B = G G^+ - (G Omega_int^T) Z X = I because G Omega^T = 0 (eq:G_OmegaT).
    if G * Omega_int.T != sym.zeros(G.rows, Omega_int.rows) or G * G_plus != sym.eye(G.rows):
        raise AssertionError("B violates G B = I (eq:GB_constraint)")
    # Z^T Omega_int C_b B = R - C~ X because Z^T Omega_int C_b Omega_int^T Z = C~.
    if (coords.C_tilde * X - R).applyfunc(sym.cancel) != sym.zeros(Z.cols, G.rows):
        raise AssertionError("B is not irrotational (eq:irrotational_expanded)")
    return sym.ImmutableMatrix(B)
