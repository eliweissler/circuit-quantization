"""Canonical physical constants and unit conversions.

Circuit inputs use farads, henrys, volts, webers, and seconds.  Element
energies and Hamiltonians use ordinary frequency units (GHz), i.e. E/h rather
than angular frequency E/hbar.
"""

from __future__ import annotations

from sympy import Integer, Rational, pi

ELEMENTARY_CHARGE = Rational("1.602176634e-19")
PLANCK_CONSTANT = Rational("6.62607015e-34")
HBAR = PLANCK_CONSTANT / (2 * pi)
REDUCED_FLUX_QUANTUM = HBAR / (2 * ELEMENTARY_CHARGE)
FLUX_QUANTUM = PLANCK_CONSTANT / (2 * ELEMENTARY_CHARGE)
GHZ = Integer(10) ** 9


def farad_to_ghz(capacitance):
    """Convert capacitance in farads to charging energy E_C/h in GHz."""
    return ELEMENTARY_CHARGE**2 / (2 * PLANCK_CONSTANT * capacitance * GHZ)


def ghz_to_farad(charging_energy):
    """Convert charging energy E_C/h in GHz to capacitance in farads."""
    return ELEMENTARY_CHARGE**2 / (2 * PLANCK_CONSTANT * charging_energy * GHZ)


def henry_to_ghz(inductance):
    """Convert inductance in henrys to inductive energy E_L/h in GHz."""
    return REDUCED_FLUX_QUANTUM**2 / (PLANCK_CONSTANT * inductance * GHZ)


def ghz_to_henry(inductive_energy):
    """Convert inductive energy E_L/h in GHz to inductance in henrys."""
    return REDUCED_FLUX_QUANTUM**2 / (
        PLANCK_CONSTANT * inductive_energy * GHZ
    )


def joule_to_ghz(energy):
    return energy / (PLANCK_CONSTANT * GHZ)


def ghz_to_joule(frequency):
    return frequency * PLANCK_CONSTANT * GHZ


def weber_to_phase(flux):
    return flux / REDUCED_FLUX_QUANTUM


def phase_to_weber(phase):
    return phase * REDUCED_FLUX_QUANTUM


def coulomb_to_cooper_pairs(charge):
    return charge / (2 * ELEMENTARY_CHARGE)


def cooper_pairs_to_coulomb(number):
    return number * (2 * ELEMENTARY_CHARGE)
