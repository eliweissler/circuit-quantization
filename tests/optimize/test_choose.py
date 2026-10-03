"""choose_S1 on circuit_quantization coordinates (the port's adapter, beyond sircuitenum's tests)."""

import sympy as sym

from circuit_quantization import Branch, Circuit, decouple, lagrangian, mode_basis, quantize, Loops, transform
from circuit_quantization.optimize import choose_S1


def zero_pi():
    """The symmetric zero-pi of tests/circuits.py (fig:zeropi_kite), floating."""
    CJ, C, L, EJ = sym.symbols("C_J C L E_J", positive=True)
    return Circuit([
        Branch("J1", 1, 2, C=CJ, EJ=EJ), Branch("J2", 3, 4, C=CJ, EJ=EJ),
        Branch("L1", 2, 3, L=L), Branch("L2", 4, 1, L=L),
        Branch("C1", 1, 3, C=C), Branch("C2", 2, 4, C=C),
    ])


ZERO_PI_Z = sym.Matrix([[-1, -1, 1, 1], [1, 1, 1, 1], [-1, 1, -1, 1], [1, -1, -1, 1]]) / 2  # eq:zp_Z


def test_choose_S1_follows_the_coordinates_column_order():
    # The notes' basis lists the extended column first; choose_S1 works internally in
    # (compact, extended, harmonic) order and must hand back S_1 in the caller's order.
    lag = lagrangian(zero_pi())
    noted = decouple(lag, ZERO_PI_Z, ("e", "c", "h", "cy"))
    automatic = decouple(lag, *mode_basis(lag), check=False)
    assert noted.types == ("e", "c", "h") and automatic.types == ("c", "e", "h")
    S_noted, h_noted = choose_S1(lag, noted)
    S_automatic, h_automatic = choose_S1(lag, automatic)
    assert h_noted == h_automatic
    for S in S_noted:
        transform(lag, noted, S)  # a valid residual transformation in the noted column order


def test_choose_S1_feeds_result_transform():
    circuit = zero_pi()
    result = quantize(circuit, Loops.from_branch_dict(circuit, {
        "ring": {"J1": 1, "J2": 1, "L1": 1, "L2": 1},
        "t1": {"J1": 1, "L1": 1, "C1": -1},
        "t2": {"J2": 1, "L1": 1, "C2": -1},
    }, {"t1": 0, "t2": 0}))
    S1s, _ = choose_S1(result.lag, result.coords)
    best = result.transform(S1s[0])
    assert best.coords.Z == result.coords.Z * S1s[0]
    assert best.B == result.B
