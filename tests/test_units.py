import sympy as sym

import circuit_quantization as cq
from circuit_quantization import units


def test_constants_are_exact_si():
    assert units.e == sym.Rational("1.602176634e-19")
    assert units.h == sym.Rational("6.62607015e-34")
    assert units.phi0 == units.h / (4 * sym.pi * units.e)


def test_converters_round_trip():
    E = sym.Symbol("E", positive=True)
    assert sym.simplify(cq.farad_to_ghz(cq.ghz_to_farad(E)) - E) == 0
    assert sym.simplify(cq.henry_to_ghz(cq.ghz_to_henry(E)) - E) == 0
    assert sym.simplify(cq.weber_to_phase(cq.phase_to_weber(E)) - E) == 0


def test_transmon_scale():
    # 20 GHz^-1 ... sanity: E_C = 0.2 GHz is about 97 fF.
    C = float(cq.ghz_to_farad(sym.Rational(1, 5)))
    assert 96e-15 < C < 98e-15
