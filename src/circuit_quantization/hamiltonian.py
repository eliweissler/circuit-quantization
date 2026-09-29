"""Frequency-unit Hamiltonian (Sec IV, Sec VI)."""

from dataclasses import dataclass

import sympy as sym

from .units import e, farad_to_ghz, henry_to_ghz, weber_to_phase


@dataclass(frozen=True)
class Hamiltonian:
    H: sym.Expr  # H/h in GHz, eq:full_H_dimless
    varphi: sym.ImmutableMatrix
    n: sym.ImmutableMatrix
    n_g: sym.ImmutableMatrix
    varphi_ext: sym.ImmutableMatrix
    E_C: sym.ImmutableMatrix
    E_L: sym.ImmutableMatrix
    E_Lb: sym.ImmutableMatrix


def simplify_hamiltonian(H):
    """Combine trigonometric terms, then factor each term's numerator and denominator.

    Keep the Hamiltonian as a sum rather than collecting it over one denominator.
    This optional symbolic step can be expensive for large circuits.
    """
    H = sym.sympify(H)
    algebraic, trig = H.as_independent(*H.atoms(sym.sin, sym.cos), as_Add=True)
    H = algebraic + sym.trigsimp(trig, method="fu")
    return sym.Add(*(sym.factor(term) for term in sym.Add.make_args(H)))


def hamiltonian(lag, coords, loops, B, t, *, simplify=False):
    N_d = coords.Z.cols
    varphi = sym.Matrix(sym.symbols(f"varphi_1:{N_d + 1}", real=True))
    n = sym.Matrix(sym.symbols(f"n_1:{N_d + 1}", real=True))
    Z, B = sym.Matrix(coords.Z), sym.Matrix(B)
    Phi_ext = sym.Matrix(loops.Phi_ext)
    varphi_ext = weber_to_phase(Phi_ext)  # eq:dimless_defs

    # eq:EC_EL_matrix: scale inverse-element matrices by the unit-element energies.
    E_C = (farad_to_ghz(1) * coords.C_tilde_inv).applyfunc(sym.cancel)
    E_L = (henry_to_ghz(1) * coords.Lstar_tilde).applyfunc(sym.cancel)
    E_Lb = (henry_to_ghz(1) * lag.Lb_inv).applyfunc(sym.cancel)

    # eq:Q_g_tilde_expanded; the flux term vanishes for irrotational B.
    Q_g = Z.T * (-lag.C_int_src * lag.V + lag.Omega_int * lag.C_b * B * Phi_ext.diff(t))
    # Each compact mode also carries a static sector label (Sec IV.B.1).
    sector = [sym.Symbol(f"n_g{j + 1}", real=True) if typ == "c" else 0 for j, typ in enumerate(coords.types)]
    n_g = (Q_g / (2 * e)).applyfunc(sym.cancel) + sym.Matrix(sector)

    B_L = B[list(lag.L), :]
    B_J = B[list(lag.J), :]
    q = n - n_g
    H = 4 * (q.T * E_C * q)[0]  # charging
    H += (varphi.T * E_L * varphi)[0] / 2  # inductive
    H += (varphi.T * coords.omega_L_tilde * E_Lb * B_L * varphi_ext)[0]  # inductive-flux cross term
    junction_phase = coords.omega_J_tilde.T * varphi + B_J * varphi_ext
    H -= sum(EJ * sym.cos(phase) for EJ, phase in zip(lag.EJ, junction_phase))  # Josephson

    freeze = sym.ImmutableMatrix
    return Hamiltonian(
        H=simplify_hamiltonian(H) if simplify else H, varphi=freeze(varphi), n=freeze(n), n_g=freeze(n_g),
        varphi_ext=freeze(varphi_ext), E_C=freeze(E_C), E_L=freeze(E_L), E_Lb=freeze(E_Lb),
    )
