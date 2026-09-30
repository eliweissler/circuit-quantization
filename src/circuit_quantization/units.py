"""Physical constants and unit conversions (Sec VI).

Circuit inputs are SI (F, H, V, Wb, s).
Energies and the Hamiltonian are ordinary frequencies E/h in GHz.
Constants remain symbolic until subs_constants() is called.
"""

import sympy as sym

# Keep SI h distinct from the GHz conversion symbol displayed as h.
e, h, hbar, Phi0 = sym.symbols("e h_SI hbar Phi_0", positive=True)
phi0 = Phi0 / (2 * sym.pi)  # reduced flux quantum [Wb], eq:dimless_defs
e_num = sym.Rational("1.602176634e-19")  # elementary charge [C]
h_num = sym.Rational("6.62607015e-34")  # Planck constant [J s]
hbar_num = h_num / (2 * sym.pi)
Phi0_num = h_num / (2 * e_num)  # flux quantum [Wb]
phi0_num = Phi0_num / (2 * sym.pi)
GHz = sym.Integer(10) ** 9
h_GHz = sym.Symbol("h", positive=True)  # h in GHz-based formulas: joules per GHz, 10^9 h_SI
h_GHz_num = h_num * GHz


def subs_constants(expr):
    """Substitute exact constants in a SymPy expression or matrix; use .evalf() for decimals.

    :param expr: SymPy expression or matrix containing symbolic physical constants.
    :returns: Expression or matrix with exact numerical constants substituted.
    """
    return sym.sympify(expr).subs({e: e_num, h: h_num, hbar: hbar_num, Phi0: Phi0_num, h_GHz: h_GHz_num}, simultaneous=True)


# eq:EC_EL_scalar: h E_C = e^2 / 2C
def farad_to_ghz(C):
    """Convert capacitance in farads.

    :param C: Capacitance in farads; scalar or SymPy expression.
    :returns: Charging energy E_C/h in GHz, with physical constants kept symbolic.
    """
    return e**2 / (2 * h_GHz * C)


def ghz_to_farad(E_C):
    """Convert charging energy in GHz to capacitance in farads.

    :param E_C: Charging energy E_C/h in GHz; scalar or SymPy expression.
    :returns: Capacitance in farads, with physical constants kept symbolic.
    """
    return e**2 / (2 * h_GHz * E_C)


# eq:EC_EL_scalar: h E_L = phi0^2 / L
def henry_to_ghz(L):
    """Convert inductance in henries.

    :param L: Inductance in henries; scalar or SymPy expression.
    :returns: Inductive energy E_L/h in GHz, with physical constants kept symbolic.
    """
    return phi0**2 / (h_GHz * L)


def ghz_to_henry(E_L):
    """Convert inductive energy in GHz to inductance in henries.

    :param E_L: Inductive energy E_L/h in GHz; scalar or SymPy expression.
    :returns: Inductance in henries, with physical constants kept symbolic.
    """
    return phi0**2 / (h_GHz * E_L)


# eq:dimless_defs: varphi = Phi / phi0
def weber_to_phase(Phi):
    """Convert flux in webers.

    :param Phi: Flux in webers; scalar or SymPy expression.
    :returns: Dimensionless phase 2*pi*Phi/Phi0, with physical constants kept symbolic.
    """
    return Phi / phi0


def phase_to_weber(varphi):
    """Convert dimensionless phase.

    :param varphi: Dimensionless phase; scalar or SymPy expression.
    :returns: Flux in webers, with physical constants kept symbolic.
    """
    return varphi * phi0
