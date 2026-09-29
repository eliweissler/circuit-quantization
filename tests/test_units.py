import sympy as sym

import circuit_quantization as cq
from circuit_quantization import units


def test_constants_are_symbolic_with_exact_si_values():
    for constant in (units.e, units.h, units.hbar, units.Phi0):
        assert isinstance(constant, sym.Symbol) and constant.is_positive
    assert units.e_num == sym.Rational("1.602176634e-19")
    assert units.h_num == sym.Rational("6.62607015e-34")
    assert units.phi0 == units.Phi0 / (2 * sym.pi)
    assert cq.subs_constants(units.hbar - units.h / (2 * sym.pi)) == 0
    assert cq.subs_constants(units.Phi0 - units.h / (2 * units.e)) == 0
    assert cq.subs_constants(units.phi0) == units.phi0_num


def test_substitution_preserves_circuit_symbols_and_matrices():
    C = sym.Symbol("C", positive=True)
    expr = cq.farad_to_ghz(C)
    expected = units.e_num**2 / (2 * units.h_num * C * units.GHz)
    assert cq.subs_constants(expr) == expected
    matrix = sym.ImmutableMatrix([[expr, units.hbar], [units.Phi0, units.phi0]])
    result = cq.subs_constants(matrix)
    assert result == sym.ImmutableMatrix([[expected, units.hbar_num], [units.Phi0_num, units.phi0_num]])
    assert result.free_symbols == {C}
    assert cq.subs_constants(result) == result


def test_converters_round_trip():
    E = sym.Symbol("E", positive=True)
    assert sym.simplify(cq.farad_to_ghz(cq.ghz_to_farad(E)) - E) == 0
    assert sym.simplify(cq.henry_to_ghz(cq.ghz_to_henry(E)) - E) == 0
    assert sym.simplify(cq.weber_to_phase(cq.phase_to_weber(E)) - E) == 0


def test_transmon_scale():
    # 20 GHz^-1 ... sanity: E_C = 0.2 GHz is about 97 fF.
    C = float(cq.subs_constants(cq.ghz_to_farad(sym.Rational(1, 5))))
    assert 96e-15 < C < 98e-15



def test_ghz_conversion_factor_keeps_symbolic_coefficients_clean():
    C, L = sym.symbols("C L", positive=True)
    assert cq.farad_to_ghz(C) == units.e**2 / (2 * units.h_GHz * C)
    assert cq.henry_to_ghz(L) == units.phi0**2 / (units.h_GHz * L)
    assert cq.subs_constants(units.h_GHz - units.GHz * units.h) == 0
    assert cq.subs_constants(cq.henry_to_ghz(L)) == units.phi0_num**2 / (units.h_num * units.GHz * L)
