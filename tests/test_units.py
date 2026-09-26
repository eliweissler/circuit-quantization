import math

import sympy as sp

import circuit_quantization as cq


def test_exact_si_constants():
    assert cq.ELEMENTARY_CHARGE == sp.Rational("1.602176634e-19")
    assert cq.PLANCK_CONSTANT == sp.Rational("6.62607015e-34")
    assert sp.simplify(cq.HBAR - cq.PLANCK_CONSTANT / (2 * sp.pi)) == 0


def test_element_conversion_round_trips():
    for capacitance in (1e-15, 25e-15, 1e-12):
        assert math.isclose(
            float(cq.ghz_to_farad(cq.farad_to_ghz(capacitance))),
            capacitance,
            rel_tol=1e-14,
        )
    for inductance in (1e-9, 50e-9, 1e-6):
        assert math.isclose(
            float(cq.ghz_to_henry(cq.henry_to_ghz(inductance))),
            inductance,
            rel_tol=1e-14,
        )


def test_symbolic_converters_follow_appendix():
    c, l = sp.symbols("C L", positive=True)
    h = cq.PLANCK_CONSTANT
    e = cq.ELEMENTARY_CHARGE
    phi0 = cq.REDUCED_FLUX_QUANTUM
    ghz = sp.Integer(10) ** 9
    assert sp.simplify(cq.farad_to_ghz(c) - e**2 / (2 * h * c * ghz)) == 0
    assert sp.simplify(cq.henry_to_ghz(l) - phi0**2 / (h * l * ghz)) == 0


def test_coordinate_unit_round_trips():
    phi, charge = sp.symbols("Phi Q")
    assert sp.simplify(cq.phase_to_weber(cq.weber_to_phase(phi)) - phi) == 0
    assert sp.simplify(
        cq.cooper_pairs_to_coulomb(cq.coulomb_to_cooper_pairs(charge))
        - charge
    ) == 0
