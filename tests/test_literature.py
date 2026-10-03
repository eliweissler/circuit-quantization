"""Closed forms from the literature, beyond the notes' own worked examples."""

import sympy as sym

import circuit_quantization as cq
from circuit_quantization import (
    Branch, Circuit, Loops, decouple, default_tree, hamiltonian,
    lagrangian, quantize, subs_constants,
)
from circuit_quantization.units import e, h_GHz
from circuits import COUPLED_ZERO_PI_Z, ZERO_PI_Z, coupled_zero_pis, pos


def zero(M):
    return sym.Matrix(M).applyfunc(sym.simplify).is_zero_matrix


def coordinates(circuit, Z, types):
    lag = lagrangian(circuit)
    coords = decouple(lag, Z, types)
    return lag, coords


def test_double_squid_with_shared_junction():
    # You, Sauls, Koch, PRB 99, 174512 (arXiv:1902.04734), Sec V.B: meshes R = ((1, -1, 0), (0, 1, -1)).
    C1, C2, C3 = pos("C_1 C_2 C_3")
    circuit = Circuit([Branch(f"J{k}", 1, 0, C=C, EJ=1) for k, C in ((1, C1), (2, C2), (3, C3))], ground=0)
    loops = Loops.from_branch_dict(circuit, {"e1": {"J1": 1, "J2": -1}, "e2": {"J2": 1, "J3": -1}})
    S = C1 + C2 + C3
    expected = sym.Matrix([[C2 + C3, C3], [-C1, C3], [-C1, -(C1 + C2)]]) / S  # last columns of M_+^{-1}
    assert zero(quantize(circuit, loops).B - expected)


def test_ring_of_junctions_divides_flux_by_inverse_capacitance():
    # You, Sauls, Koch, Sec V.A: in a loop of junctions each junction takes C_k^{-1} / sum_j C_j^{-1}.
    Cs = pos("C_1:6")
    circuit = Circuit([Branch(f"J{k}", k, k % 5 + 1, C=C, EJ=1) for k, C in enumerate(Cs, start=1)])
    loops = Loops.from_branch_dict(circuit, {"ring": {f"J{k}": 1 for k in range(1, 6)}})
    B = quantize(circuit, loops).B
    assert zero(B - sym.Matrix([1 / C for C in Cs]) / sum(1 / C for C in Cs))


def test_junction_allocation_is_the_irrotational_hamiltonian_in_a_moving_frame():
    # Bryon et al. (arXiv:2208.03738), Eqs. (2)-(4): putting Phi(t) on the fluxonium junction leaves a
    # dPhi/dt drive in n_g.  The shift varphi -> varphi - varphi_e with generator 2e n dPhi/dt / h
    # (eq:gauge_H_expanded) must recover the irrotational H, which puts Phi(t) on the inductor.
    CJ, L, EJ = pos("C_J L E_J")
    circuit = Circuit([Branch("J", 1, 0, C=CJ, EJ=EJ), Branch("L", 1, 0, L=L)], ground=0)
    Phi = sym.Function("Phi")(circuit.t)
    loops = Loops.from_branch_dict(circuit, {"f": {"J": 1, "L": -1}}, {"f": Phi})
    irrotational = quantize(circuit, loops)
    assert irrotational.B == sym.Matrix([[0], [-1]])  # inductor allocation, B_J = 0
    lag, coords = irrotational.lag, irrotational.coords
    junction = hamiltonian(lag, coords, loops, sym.Matrix([[1], [0]]), circuit.t)
    assert junction.n_g.has(sym.Derivative)
    varphi, n = junction.varphi[0], junction.n[0]
    moved = junction.H.subs(varphi, varphi - junction.varphi_ext[0]) + 2 * e * n * Phi.diff(circuit.t) / h_GHz
    difference = sym.expand(moved - irrotational.H.subs({irrotational.ham.varphi[0]: varphi, irrotational.ham.n[0]: n}))
    assert not difference.free_symbols & {varphi, n}


def test_disordered_zero_pi_capacitance_and_susceptance():
    # Notes eq:zp_disorder (after Groszkowski et al. and arXiv:2503.14634), order (phi, theta, zeta).
    CJ1, CJ2, C1, C2, L1, L2 = pos("C_J1 C_J2 C_1 C_2 L_1 L_2")
    circuit = Circuit([
        Branch("J1", 1, 2, C=CJ1, EJ=1), Branch("J2", 3, 4, C=CJ2, EJ=1),
        Branch("L1", 2, 3, L=L1), Branch("L2", 4, 1, L=L2),
        Branch("C1", 1, 3, C=C1), Branch("C2", 2, 4, C=C2),
    ])
    _, coords = coordinates(circuit, ZERO_PI_Z, ("e", "c", "h", "cy"))
    CJ, dCJ = (CJ1 + CJ2) / 2, (CJ1 - CJ2) / ((CJ1 + CJ2) / 2)
    C, dC = (C1 + C2) / 2, (C1 - C2) / ((C1 + C2) / 2)
    EL, dEL = (1 / L1 + 1 / L2) / 2, (1 / L1 - 1 / L2) / ((1 / L1 + 1 / L2) / 2)  # phi_0^2 set to 1
    assert zero(coords.C_tilde - sym.Matrix([
        [2 * CJ, CJ * dCJ, 0], [CJ * dCJ, 2 * (C + CJ), -C * dC], [0, -C * dC, 2 * C]]))
    assert zero(coords.Lstar_tilde - sym.Matrix([[2 * EL, 0, EL * dEL], [0, 0, 0], [EL * dEL, 0, 2 * EL]]))


def test_coupled_zero_pis_mixed_coupler():
    # Zero-pi two-qubit gate paper, circuit (b): theta_1 theta_2, phi_1 phi_2, and zeta_1 zeta_2
    # couplings with E_C^mu = e^2 C_mu / 2(C_mu^2 - C_c^2) and g_mu = 4 e^2 C_c / (C_mu^2 - C_c^2).
    CJ, C, C0, Cc = pos("C_J C C_0 C_c")
    Z = COUPLED_ZERO_PI_Z[:8, :8]
    _, coords = coordinates(coupled_zero_pis("b"), Z, ("c", "e", "h", "fr") * 2)
    assert coords.types == ("c", "e", "h", "c", "e", "h")
    expected = sym.zeros(6, 6)
    for k, (C_mu, sign) in enumerate((
        (2 * C + 2 * CJ + Cc + C0, 1), (2 * CJ + Cc + C0, -1), (2 * C + Cc + C0, -1),
    )):  # theta, phi, zeta
        expected[k, k] = expected[k + 3, k + 3] = C_mu / (C_mu**2 - Cc**2)
        expected[k, k + 3] = expected[k + 3, k] = sign * Cc / (C_mu**2 - Cc**2)
    assert zero(coords.C_tilde_inv - expected)


def test_ding_free_mode_removal_example():
    # Ding et al., PRB 103, 174501 (arXiv:2011.10564), Sec III example: a driven Cooper-pair box
    # coupled to a resonator.  Their matrices are 8 E_C in GHz (H = n^T M n / 2, Q = 2e n),
    # quoted to three figures, with coordinates (Phi_1g, Phi_q = Phi_2 - Phi_1, Phi_rg).
    fF, nH = sym.Rational(1, 10**15), sym.Rational(1, 10**9)
    circuit = Circuit([
        Branch("C1g", 1, "g", C=fF), Branch("C2g", 2, "g", C=2 * fF), Branch("Crg", "r", "g", C=30 * fF),
        Branch("J", 1, 2, C=40 * fF, EJ=3), Branch("C1r", 1, "r", C=5 * fF), Branch("C2r", 2, "r", C=6 * fF),
        Branch("C1v", 1, "v", C=7 * fF / 1000), Branch("V", "g", "v", V=sym.Symbol("V")),
        Branch("L", "r", "g", L=800 * nH),
    ], ground="g")
    Z = sym.Matrix([[1, 0, 0], [1, 1, 0], [0, 0, 1]])  # columns 1g (free), q, r over nodes (1, 2, r)
    lag, coords = coordinates(circuit, Z, ("fr", "c", "h"))

    def M(inverse_capacitance):
        return subs_constants(8 * cq.farad_to_ghz(1) * inverse_capacitance)

    def close(A, B):
        return all(abs(float(a) - b) <= 0.006 * abs(b) + 1e-3 for a, b in zip(A, B))

    assert close(M((Z.T * lag.C_int * Z).inv()), [15.2, -2.06, 3.78, -2.06, 3.57, -0.0312, 3.78, -0.0312, 4.79])
    # Removing the free mode preserves the nonfree block of C^{-1} (their C'^{-1}).
    assert close(M(coords.C_tilde_inv), [3.57, -0.0312, -0.0312, 4.79])
    assert close(subs_constants(-lag.C_int_src / (2 * e)), [-21.8, 0, 0])  # their C_V, per volt
    # Their M_0 entry reads 0.0204 GHz, but phi_0^2 / (h 800 nH) is 0.204 GHz.
    assert abs(float(subs_constants(cq.henry_to_ghz(800 * nH))) - 0.204) < 1e-3


def test_series_inductors_are_the_small_capacitance_limit():
    # Rymarz and DiVincenzo, PRX 13, 021017 (arXiv:2208.11767), Sec II.A: a node joining two linear
    # inductors is frozen.  Giving it a capacitance eps and letting eps -> 0 must reproduce the frozen
    # reduction: the fast node's Schur complement of E_L and the slow E_C both converge.
    CJ, La, Lb, eps = pos("C_J L_a L_b epsilon")

    def fluxonium(C_middle):
        return Circuit([Branch("J", 1, 0, C=CJ, EJ=1), Branch("La", 1, 2, L=La),
                        Branch("Lb", 2, 0, C=C_middle, L=Lb)], ground=0)

    frozen = quantize(fluxonium(0), Loops.from_tree(fluxonium(0), default_tree(fluxonium(0))))
    assert frozen.coords.types == ("e",)
    circuit = fluxonium(eps)
    lag, coords = coordinates(circuit, sym.eye(2), ("e", "h"))
    E_C = cq.farad_to_ghz(1) * coords.C_tilde_inv
    E_L = cq.henry_to_ghz(1) * coords.Lstar_tilde
    schur = E_L[0, 0] - E_L[0, 1] ** 2 / E_L[1, 1]
    assert sym.simplify(schur - frozen.ham.E_L[0, 0]) == 0
    assert sym.simplify(sym.limit(E_C[0, 0], eps, 0) - frozen.ham.E_C[0, 0]) == 0
