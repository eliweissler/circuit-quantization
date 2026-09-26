"""Irrotational external-flux assignment (Sec V)."""

import sympy as sym


def irrotational_B(lag, coords, loops):
    """B = (I - Omega_int^T Z C~^{-1} Z^T Omega_int C_b) G^+, eq:B_irrot_direct_closed with W = 0."""
    G = sym.Matrix(loops.G)
    N_b = lag.Omega.cols
    if G.rows == 0:
        return sym.ImmutableMatrix(sym.zeros(N_b, 0))
    G_plus = G.T * (G * G.T).inv()  # eq:G_right_inverse
    Z, Omega_int, C_b = coords.Z, lag.Omega_int, lag.C_b
    P = sym.eye(N_b) - Omega_int.T * Z * coords.C_tilde_inv * Z.T * Omega_int * C_b
    B = (P * G_plus).applyfunc(sym.cancel)
    if (G * B - sym.eye(G.rows)).applyfunc(sym.cancel) != sym.zeros(G.rows, G.rows):
        raise AssertionError("B violates G B = I (eq:GB_constraint)")
    if (Z.T * Omega_int * C_b * B).applyfunc(sym.cancel) != sym.zeros(Z.cols, G.rows):
        raise AssertionError("B is not irrotational (eq:irrotational_expanded)")
    return sym.ImmutableMatrix(B)
