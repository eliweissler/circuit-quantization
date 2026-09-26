"""Physical constants and unit conversions (Sec VI).

Circuit inputs are SI (F, H, V, Wb, s).
Energies and the Hamiltonian are ordinary frequencies E/h in GHz.
"""

import sympy as sym

e = sym.Rational("1.602176634e-19")  # elementary charge [C]
h = sym.Rational("6.62607015e-34")  # Planck constant [J s]
hbar = h / (2 * sym.pi)
Phi0 = h / (2 * e)  # flux quantum [Wb]
phi0 = hbar / (2 * e)  # reduced flux quantum [Wb], eq:dimless_defs
GHz = sym.Integer(10) ** 9


# eq:EC_EL_scalar: h E_C = e^2 / 2C
def farad_to_ghz(C):
    return e**2 / (2 * h * C * GHz)


def ghz_to_farad(E_C):
    return e**2 / (2 * h * E_C * GHz)


# eq:EC_EL_scalar: h E_L = phi0^2 / L
def henry_to_ghz(L):
    return phi0**2 / (h * L * GHz)


def ghz_to_henry(E_L):
    return phi0**2 / (h * E_L * GHz)


# eq:dimless_defs: varphi = Phi / phi0
def weber_to_phase(Phi):
    return Phi / phi0


def phase_to_weber(varphi):
    return varphi * phi0
