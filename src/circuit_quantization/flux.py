"""Irrotational external-flux assignment (Sec V)."""

from .modes import _cancel
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
    G, Z, Omega_int = loops.G, coords.Z, lag.Omega_int
    G_plus = G.T * (G * G.T).inv()  # eq:G_right_inverse
    drive = _cancel(Z.T * Omega_int * lag.C_b * G_plus)
    X = _cancel(coords.C_tilde_inv * drive)
    return _cancel(G_plus - Omega_int.T * Z * X)


def B_from_shares(circuit, loops, shares):
    """Flux assignment B (branches, loops) from ``shares[loop][branch]``.

    :raises ValueError: unless ``loops.G * B`` is the identity.
    """
    branches = list(circuit.branch_names)
    B = sym.zeros(len(branches), len(loops.names))
    for j, loop in enumerate(loops.names):
        for branch, value in shares.get(loop, {}).items():
            B[branches.index(branch), j] = sym.sympify(value)
    if (sym.simplify(sym.Matrix(loops.G) * B - sym.eye(len(loops.names)))) != sym.zeros(*[len(loops.names)] * 2):
        raise ValueError("shares do not satisfy G * B = I")
    return sym.ImmutableMatrix(B)