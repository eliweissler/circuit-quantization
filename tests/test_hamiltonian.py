import pytest
import sympy as sym

import circuit_quantization as cq
from circuit_quantization import Branch, Circuit, Loops, quantize
from circuits import HARMONIUM_CYCLES, HARMONIUM_Z, KITE_CYCLES, KITE_Z, kite, pos


def harmonium_in_ghz():
    """fig:harmonium with element energies in GHz, converted to SI at input."""
    EC, EC1, EC2, EC3, EL1, EL2, EL3, EJ1, EJ2, EJ3 = pos(
        "E_C E_CJ1 E_CJ2 E_CJ3 E_L1 E_L2 E_L3 E_J1 E_J2 E_J3")
    F, H = cq.ghz_to_farad, cq.ghz_to_henry
    return Circuit([
        Branch("C", 1, 5, C=F(EC)),
        Branch("J1", 1, 2, C=F(EC1), EJ=EJ1),
        Branch("J2", 1, 3, C=F(EC2), EJ=EJ2),
        Branch("J3", 4, 5, C=F(EC3), EJ=EJ3),
        Branch("L1", 2, 4, L=H(EL1)),
        Branch("L2", 3, 4, L=H(EL2)),
        Branch("L3", 4, 5, L=H(EL3)),
    ], ground=5)


def test_harmonium_dimensionless_hamiltonian():
    EC, EC1, EC2, EC3, EL1, EL2, EL3, EJ1, EJ2, EJ3 = pos(
        "E_C E_CJ1 E_CJ2 E_CJ3 E_L1 E_L2 E_L3 E_J1 E_J2 E_J3")
    vk, vf = sym.symbols("varphi_k varphi_f", real=True)
    circuit = harmonium_in_ghz()
    fluxes = {"k": cq.phase_to_weber(vk), "f": cq.phase_to_weber(vf), "cb": 0}
    res = quantize(circuit, Loops(circuit, HARMONIUM_CYCLES, fluxes), HARMONIUM_Z, ("c", "e", "e", "e"))
    p1, p2, p3, p4 = res.varphi
    n1, n2, n3, n4 = res.n
    ng = sym.Symbol("n_g1", real=True)
    EbarC, dEC = (EC1 + EC2) / 2, (EC1 - EC2) / 2
    expected = (  # eq:harm_H_dimless
        4 * EC * (n1 + n4 - ng) ** 2 + 4 * EC3 * (n2 - n4) ** 2
        + 2 * EbarC * (n3**2 + n4**2) + 4 * dEC * n3 * n4
        + EL1 / 2 * (p3 + p4) ** 2 + EL2 / 2 * (p4 - p3 - vk) ** 2 + EL3 / 2 * (p2 - vf) ** 2
        - EJ1 * sym.cos(p1 - p2 - p3 - p4) - EJ2 * sym.cos(p1 - p2 + p3 - p4) - EJ3 * sym.cos(p2)
    )
    difference = sym.expand(res.H - expected)
    # Equal up to the flux-only constant dropped in eq:U_full.
    assert not (difference.free_symbols & (set(res.varphi) | set(res.n)))


def test_kite_charging_matrix():
    ECS, EC1, EC2 = pos("E_CS E_CJ1 E_CJ2")
    circuit = kite()
    circuit_ghz = Circuit([
        b if b.name not in ("CS", "J1", "J2") else Branch(
            b.name, b.tail, b.head, C=cq.ghz_to_farad({"CS": ECS, "J1": EC1, "J2": EC2}[b.name]), EJ=b.EJ)
        for b in circuit.branches
    ], ground=4)
    res = quantize(circuit_ghz, Loops(circuit_ghz, KITE_CYCLES, {"cb": 0}), KITE_Z, ("c", "e", "e"))
    s, d = (EC1 + EC2) / 4, (EC1 - EC2) / 4
    expected = sym.Matrix([[ECS, 0, ECS], [0, s, d], [ECS, d, ECS + s]])  # eq:kite_Cinv in GHz
    assert (res.E_C - expected).applyfunc(sym.simplify).is_zero_matrix


def test_voltage_drive_offset_charge_sign():
    # Q_g = -C_int,src V (eq:Qg): a positive gate voltage gives n_g = -C_g V / 2e.
    CJ, Cg, V = pos("C_J C_g V")
    circuit = Circuit([Branch("J", 1, 0, C=CJ, EJ=1), Branch("Cg", 1, 2, C=Cg), Branch("V", 0, 2, V=V)], ground=0)
    cycles = cq.fundamental_cycles(circuit)  # one loop J -> V -> Cg, no flux
    res = quantize(circuit, Loops(circuit, cycles, {name: 0 for name in cycles}))
    sector = sym.Symbol("n_g1", real=True)
    assert sym.simplify(res.n_g[0] - (sector - Cg * V / (2 * cq.units.e))) == 0


def test_time_dependent_flux_leaves_no_emf_drive():
    C1, C2 = pos("C_1 C_2")
    circuit = Circuit([Branch("J1", 1, 0, C=C1, EJ=1), Branch("J2", 1, 0, C=C2, EJ=1)], ground=0)
    Phi = sym.Function("Phi")(circuit.t)
    res = quantize(circuit, Loops(circuit, {"sq": {"J1": 1, "J2": -1}}, {"sq": Phi}))
    assert not res.n_g.has(sym.Derivative)
    assert res.H.has(Phi)


def test_result_exposes_named_matrices_and_latex():
    circuit = kite()
    res = quantize(circuit, Loops(circuit, KITE_CYCLES, {"cb": 0}), KITE_Z, ("c", "e", "e"))
    assert res.Z == KITE_Z
    assert res.C_int.shape == (3, 3)
    assert "H/h" in res._repr_latex_()


def test_transmon_without_loops_and_with_float_parameters():
    circuit = Circuit([Branch("J", "island", "gnd", C=97e-15, EJ=15.0)], ground="gnd")
    res = quantize(circuit, Loops(circuit, {}))
    assert res.coords.types == ("c",)
    assert res.B.shape == (1, 0)
    assert abs(float(res.E_C[0, 0]) - float(cq.farad_to_ghz(97e-15))) < 1e-12


def test_lc_oscillator_is_harmonic():
    circuit = Circuit([Branch("C", 1, 0, C=cq.ghz_to_farad(0.2)), Branch("L", 1, 0, L=cq.ghz_to_henry(0.5))], ground=0)
    cycles = cq.fundamental_cycles(circuit)
    res = quantize(circuit, Loops(circuit, cycles, {name: 0 for name in cycles}))
    assert res.coords.types == ("h",)
    assert sym.nsimplify(res.E_L[0, 0]) == sym.Rational(1, 2)


def test_float_parameters_in_circuits_with_loops_and_free_modes():
    # Floats are rationalized at input, so exact-zero checks hold (Review Focus 1).
    squid = Circuit([Branch("J1", 1, 0, C=41.7e-15, EJ=12.3), Branch("J2", 1, 0, C=13.1e-15, EJ=7.7),
                     Branch("Cs", 1, 0, C=73.3e-15)], ground=0)
    res = quantize(squid, Loops(squid, cq.fundamental_cycles(squid)))
    assert abs(float(res.E_C[0, 0]) - float(cq.farad_to_ghz(128.1e-15))) < 1e-9
    floating = Circuit([Branch("J", 1, 2, C=5e-15, EJ=10.0), Branch("C1", 1, 0, C=30e-15),
                        Branch("C2", 2, 0, C=40e-15)], ground=0)
    cycles = cq.fundamental_cycles(floating)
    res = quantize(floating, Loops(floating, cycles, {name: 0 for name in cycles}))
    assert res.coords.types == ("c",)


def test_quantize_infers_types_for_a_given_Z():
    circuit = kite()
    res = quantize(circuit, Loops(circuit, KITE_CYCLES, {"cb": 0}), KITE_Z)
    assert res.coords.types == ("c", "e", "e")
    with pytest.raises(ValueError, match="types requires Z_full"):
        quantize(circuit, Loops(circuit, KITE_CYCLES, {"cb": 0}), types=("c", "e", "e"))
