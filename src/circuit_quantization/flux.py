"""Irrotational external-flux assignment (Sec V)."""

import sympy as sym


def irrotational_B(lag, coords, loops):
    """B = (I - Omega_int^T Z C~^{-1} Z^T Omega_int C_b) G^+, eq:B_irrot_direct_closed with W = 0.

    Evaluated as B = G^+ - Omega_int^T Z X with X = C~^{-1} Z^T Omega_int C_b G^+,
    so the only rational-function matrix built is the small N_d x N_ell block X.
    The omitted homogeneous term is Omega_int^T W with coords.R * W = 0,
    equivalently Z^T C_int W = 0 (eq:R_capacitance). Z^+ W = 0 is not the
    correct condition after truncation. Choosing W = 0 leaves this formula unchanged.

    :param lag: Node Lagrangian matrices.
    :param coords: Decoupled retained coordinates.
    :param loops: Complete cycle basis and branch ordering for the circuit.
    :returns: Branch-by-loop flux assignment matrix B, satisfying G * B = I.
    """
    G = sym.Matrix(loops.G)
    N_b = lag.Omega.cols
    if G.rows == 0:
        return sym.ImmutableMatrix(sym.zeros(N_b, 0))
    G_plus = G.T * (G * G.T).inv()  # eq:G_right_inverse
    Z, Omega_int = sym.Matrix(coords.Z), sym.Matrix(lag.Omega_int)
    drive = (Z.T * Omega_int * lag.C_b * G_plus).applyfunc(sym.cancel)
    X = (coords.C_tilde_inv * drive).applyfunc(sym.cancel)
    B = (G_plus - Omega_int.T * Z * X).applyfunc(sym.cancel)
    return sym.ImmutableMatrix(B)
