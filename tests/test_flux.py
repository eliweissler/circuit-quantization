import sympy as sym

from circuit_quantization import (
    Branch, Circuit, Loops, decouple, default_tree, irrotational_B, lagrangian, mode_basis,
)
from circuits import (
    HARMONIUM_CYCLES, HARMONIUM_Z, KITE_CYCLES, KITE_Z, RHOMBUS_CYCLES, RHOMBUS_Z,
    ZERO_PI_CYCLES, ZERO_PI_Z, coupled, harmonium, kite, parallel, pos, rhombus, zero_pi,
)


def assign(circuit, loops, Z=None, types=None):
    """Irrotational B for the given (or automatic) coordinates."""
    lag = lagrangian(circuit)
    if Z is None:
        Z, types = mode_basis(lag)
    return irrotational_B(lag, decouple(lag, Z, types), loops)


def column(circuit, loops, loop, Z=None, types=None):
    """One column of B, as {branch name: coefficient}."""
    B = assign(circuit, loops, Z, types)
    j = loops.names.index(loop)
    return {name: sym.simplify(B[k, j]) for k, name in enumerate(circuit.branch_names)}


def test_isolated_squid_divider():
    C1, C2 = pos("C_1 C_2")
    circuit = parallel(Branch("J1", 1, 0, C=C1, EJ=1), Branch("J2", 1, 0, C=C2, EJ=1))
    B = column(circuit, Loops.from_branch_dict(circuit, {"sq": {"J1": 1, "J2": -1}}), "sq")
    assert sym.simplify(B["J1"] - C2 / (C1 + C2)) == 0
    assert sym.simplify(B["J2"] + C1 / (C1 + C2)) == 0


def test_fluxonium_divider_with_and_without_inductor_capacitance():
    CJ, CL, L = pos("C_J C_L L")
    for C_L, expected in ((0, {"J": 0, "L": -1}), (CL, {"J": CL / (CJ + CL), "L": -CJ / (CJ + CL)})):
        circuit = parallel(Branch("J", 1, 0, C=CJ, EJ=1), Branch("L", 1, 0, C=C_L, L=L))
        B = column(circuit, Loops.from_branch_dict(circuit, {"f": {"J": 1, "L": -1}}), "f")
        assert all(sym.simplify(B[k] - v) == 0 for k, v in expected.items())


def test_shared_inductor():
    C1, C2, L = pos("C_1 C_2 L")
    circuit = parallel(Branch("J1", 1, 0, C=C1, EJ=1), Branch("J2", 1, 0, C=C2, EJ=1), Branch("L", 1, 0, L=L))
    B = assign(circuit, Loops.from_branch_dict(circuit, {"1": {"J1": 1, "L": -1}, "2": {"J2": 1, "L": -1}}))
    expected = sym.Matrix([[C2, -C2], [-C1, C1], [-C1, -C2]]) / (C1 + C2)  # fig:shared_inductor
    assert (B - expected).applyfunc(sym.simplify).is_zero_matrix


def test_shunt_placement_is_set_by_the_capacitively_broken_flux():
    C1, C2, C, x = pos("C_1 C_2 C x")
    Phi = sym.Symbol("Phi", real=True)
    circuit = parallel(Branch("J1", 1, 0, C=C1, EJ=1), Branch("J2", 1, 0, C=C2, EJ=1), Branch("C", 1, 0, C=C))
    loops = Loops.from_branch_dict(circuit, {"sq": {"J1": 1, "J2": -1}, "cb": {"J1": 1, "C": -1}}, {"sq": Phi, "cb": x * Phi})
    offsets = assign(circuit, loops) * loops.Phi_ext
    S = C1 + C2 + C
    J1, J2 = circuit.branch_names.index("J1"), circuit.branch_names.index("J2")
    assert sym.simplify(offsets[J1] - (C * x + C2) / S * Phi) == 0  # eq:shunt_placement
    assert sym.simplify(offsets[J2] + (C * (1 - x) + C1) / S * Phi) == 0


def test_capacitive_coupler_loop():
    CJ1, CJ2, Cc = pos("C_J1 C_J2 C_c")
    circuit = parallel(Branch("J1", 1, 0, C=CJ1, EJ=1), Branch("J2", 2, 0, C=CJ2, EJ=1), Branch("Cc", 1, 2, C=Cc))
    B = column(circuit, Loops.from_branch_dict(circuit, {"c": {"J1": 1, "J2": -1, "Cc": -1}}), "c")
    D = Cc * (CJ1 + CJ2) + CJ1 * CJ2
    assert sym.simplify(B["J1"] - Cc * CJ2 / D) == 0  # eq:coupler_loop
    assert sym.simplify(B["J2"] + Cc * CJ1 / D) == 0


def test_harmonium_B():
    circuit = harmonium()
    B = assign(circuit, Loops.from_branch_dict(circuit, HARMONIUM_CYCLES, {"cb": 0}), HARMONIUM_Z, ("c", "e", "e", "e"))
    assert B[:4, :] == sym.zeros(4, 3)  # B_C = B_J = 0
    assert B[4:, :] == sym.Matrix([[0, 0, -1], [-1, 0, -1], [0, -1, 0]])  # eq:harm_B


def test_zero_pi_B():
    circuit = zero_pi()
    loops = Loops.from_branch_dict(circuit, ZERO_PI_CYCLES, {"t1": 0, "t2": 0})
    B = column(circuit, loops, "ring", ZERO_PI_Z, ("e", "c", "h", "cy"))
    assert (B["J1"], B["J2"], B["L1"], B["L2"]) == (0, 0, 0, 1)  # eq:zp_B


def test_kite_B():
    circuit = kite()
    B = column(circuit, Loops.from_branch_dict(circuit, KITE_CYCLES, {"cb": 0}), "kite", KITE_Z, ("c", "e", "e"))
    assert (B["CS"], B["J1"], B["J2"], B["L1"], B["L2"]) == (0, 0, 0, 0, -1)


def test_rhombus_B_isolated_limit():
    CJ, CB, Cg, alpha = pos("C_J C_B C_g alpha")
    circuit = rhombus()
    ring = column(circuit, Loops.from_branch_dict(circuit, RHOMBUS_CYCLES), "ring", RHOMBUS_Z, ("c", "c", "c", "fr"))
    B = {k: sym.simplify(v.subs(Cg, 0)) for k, v in ring.items()}
    D = 2 * CB * (1 + alpha) + CJ * (1 + 3 * alpha)
    expected = {"J1": alpha * CJ / D, "J2": alpha * CJ / D, "J3": alpha * (2 * CB + CJ) / D, "J4": (2 * CB + CJ) / D}
    assert all(sym.simplify(B[k] - v) == 0 for k, v in expected.items())  # eq:rh_B


def test_branch_offsets_invariant_under_rebase():
    circuit = harmonium()
    loops = Loops.from_branch_dict(circuit, HARMONIUM_CYCLES, {"cb": 0})
    T = sym.Matrix([[1, 0, 0], [1, 1, 0], [1, 0, 1]])  # rows k, k + f, cb + k: all physical loops
    rebased = loops.rebase(T * loops.G)
    before = assign(circuit, loops, HARMONIUM_Z, ("c", "e", "e", "e")) * loops.Phi_ext
    after = assign(circuit, rebased, HARMONIUM_Z, ("c", "e", "e", "e")) * rebased.Phi_ext
    assert (before - after).applyfunc(sym.simplify).is_zero_matrix


def test_coupled_example_assignment_is_fast():
    # fig:coupled_circuit_example: 15 branches, 9 loops, all-symbolic capacitances.
    import signal

    def timeout(*_):
        raise TimeoutError("irrotational_B took longer than 60 s")

    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(60)
    try:
        circuit = coupled()
        B = assign(circuit, Loops.from_tree(circuit, default_tree(circuit)))
    finally:
        signal.alarm(0)
    assert B.shape == (15, 9)


def test_irrotational_assignment_satisfies_both_constraints():
    CJ, C1, C2 = pos("C_J C_1 C_2")
    circuit = Circuit([
        Branch("J", 1, 2, C=CJ, EJ=1),
        Branch("C1", 1, 0, C=C1), Branch("C2", 2, 0, C=C2),
    ], ground=0)  # Includes a discarded free mode and a capacitive loop.
    lag = lagrangian(circuit)
    coords = decouple(lag, *mode_basis(lag))
    loops = Loops.from_tree(circuit, default_tree(circuit))
    B = irrotational_B(lag, coords, loops)
    assert (loops.G * B - sym.eye(loops.G.rows)).applyfunc(sym.cancel).is_zero_matrix
    assert (coords.Z.T * lag.Omega_int * lag.C_b * B).applyfunc(sym.cancel).is_zero_matrix
