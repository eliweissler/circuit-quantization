import sympy as sym

from circuit_quantization import Loops, decouple, irrotational_B, lagrangian, mode_basis, numerify, quantize
from circuits import KITE_CYCLES, kite, zero_pi_asymmetric


def element_values(lag):
    """Distinct exact values for every element symbol of lag."""
    symbols = (lag.C_b.free_symbols | lag.Lb_inv.free_symbols | set().union(*(sym.sympify(E).free_symbols for E in lag.EJ)))
    return {s: sym.Rational(10 + 3 * k, 10) for k, s in enumerate(sorted(symbols, key=str))}


def test_numerify_result_substitutes_every_field_but_circuit_and_loops():
    circuit = kite()
    result = quantize(circuit, Loops.from_branch_dict(circuit, KITE_CYCLES))
    values = element_values(result.lag)
    num = numerify(result, values)
    assert num.circuit is result.circuit and num.loops is result.loops
    assert num.lag.J == result.lag.J and num.lag.L == result.lag.L
    assert num.lag.EJ == tuple(sym.sympify(E).subs(values) for E in result.lag.EJ)
    assert num.coords.types == result.coords.types
    for name in ("C_tilde", "C_tilde_inv", "Lstar_tilde", "R", "Z"):
        assert getattr(num.coords, name) == getattr(result.coords, name).subs(values)
    assert num.B == result.B.subs(values)
    assert num.ham.E_C == result.ham.E_C.subs(values)
    assert not num.H.free_symbols & set(values)


def test_irrotational_B_commutes_with_numerify():
    circuit = zero_pi_asymmetric()
    loops = Loops.from_circuit(circuit)
    lag = lagrangian(circuit)
    coords = decouple(lag, *mode_basis(lag))
    values = element_values(lag)
    B_num = irrotational_B(numerify(lag, values), numerify(coords, values), loops)
    assert B_num == numerify(irrotational_B(lag, coords, loops), values)


def test_exact_values_keep_mode_classification():
    lag = lagrangian(zero_pi_asymmetric())
    num = numerify(lag, element_values(lag))
    assert decouple(num, *mode_basis(num)).types == decouple(lag, *mode_basis(lag)).types
