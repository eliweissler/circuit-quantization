import pytest
import sympy as sym

import circuit_quantization as cq
from circuit_quantization import Branch, Circuit, Loops, quantize
from circuits import HARMONIUM_CYCLES, HARMONIUM_Z, KITE_CYCLES, KITE_Z, ZERO_PI_CYCLES, kite, pos, zero_pi


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
    res = quantize(circuit, Loops.from_branch_dict(circuit, HARMONIUM_CYCLES, fluxes), HARMONIUM_Z, ("c", "e", "e", "e"))
    p1, p2, p3, p4 = res.ham.varphi
    n1, n2, n3, n4 = res.ham.n
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
    assert not (difference.free_symbols & (set(res.ham.varphi) | set(res.ham.n)))


def test_kite_charging_matrix():
    ECS, EC1, EC2 = pos("E_CS E_CJ1 E_CJ2")
    circuit = kite()
    circuit_ghz = Circuit([
        b if b.name not in ("CS", "J1", "J2") else Branch(
            b.name, b.tail, b.head, C=cq.ghz_to_farad({"CS": ECS, "J1": EC1, "J2": EC2}[b.name]), EJ=b.EJ)
        for b in circuit.branches
    ], ground=4)
    res = quantize(circuit_ghz, Loops.from_branch_dict(circuit_ghz, KITE_CYCLES, {"cb": 0}), KITE_Z, ("c", "e", "e"))
    s, d = (EC1 + EC2) / 4, (EC1 - EC2) / 4
    expected = sym.Matrix([[ECS, 0, ECS], [0, s, d], [ECS, d, ECS + s]])  # eq:kite_Cinv in GHz
    assert (res.ham.E_C - expected).applyfunc(sym.simplify).is_zero_matrix


def test_voltage_drive_offset_charge_sign():
    # Q_g = -C_int,src V (eq:Qg): a positive gate voltage gives n_g = -C_g V / 2e.
    CJ, Cg, V = pos("C_J C_g V")
    circuit = Circuit([Branch("J", 1, 0, C=CJ, EJ=1), Branch("Cg", 1, 2, C=Cg), Branch("V", 0, 2, V=V)], ground=0)
    closure, G = cq.tree_to_G(circuit, cq.default_tree(circuit))  # one loop J -> V -> Cg, no flux
    res = quantize(circuit, Loops(circuit, G, closure, closure, dict.fromkeys(closure, 0)))
    sector = sym.Symbol("n_g1", real=True)
    assert sym.simplify(res.ham.n_g[0] - (sector - Cg * V / (2 * cq.units.e))) == 0


def test_time_dependent_flux_leaves_no_emf_drive():
    C1, C2 = pos("C_1 C_2")
    circuit = Circuit([Branch("J1", 1, 0, C=C1, EJ=1), Branch("J2", 1, 0, C=C2, EJ=1)], ground=0)
    Phi = sym.Function("Phi")(circuit.t)
    res = quantize(circuit, Loops.from_branch_dict(circuit, {"sq": {"J1": 1, "J2": -1}}, {"sq": Phi}))
    assert not res.ham.n_g.has(sym.Derivative)
    assert res.H.has(Phi)


def test_result_exposes_named_matrices_and_latex():
    circuit = kite()
    res = quantize(circuit, Loops.from_branch_dict(circuit, KITE_CYCLES, {"cb": 0}), KITE_Z, ("c", "e", "e"))
    assert res.coords.Z == KITE_Z
    assert res.lag.C_int.shape == (3, 3)
    assert r"H_{\mathrm{GHz}}" in res._repr_latex_()


def test_transmon_without_loops_and_with_float_parameters():
    circuit = Circuit([Branch("J", "island", "gnd", C=97e-15, EJ=15.0)], ground="gnd")
    res = quantize(circuit, Loops.from_branch_dict(circuit, {}))
    assert res.coords.types == ("c",)
    assert res.B.shape == (1, 0)
    assert abs(float(cq.subs_constants(res.ham.E_C[0, 0])) - float(cq.subs_constants(cq.farad_to_ghz(97e-15)))) < 1e-12


def test_lc_oscillator_is_harmonic():
    circuit = Circuit([Branch("C", 1, 0, C=cq.ghz_to_farad(0.2)), Branch("L", 1, 0, L=cq.ghz_to_henry(0.5))], ground=0)
    closure, G = cq.tree_to_G(circuit, cq.default_tree(circuit))
    res = quantize(circuit, Loops(circuit, G, closure, closure, dict.fromkeys(closure, 0)))
    assert res.coords.types == ("h",)
    assert sym.nsimplify(res.ham.E_L[0, 0]) == sym.Rational(1, 2)


def test_float_parameters_in_circuits_with_loops_and_free_modes():
    # Floats are rationalized at input, so exact-zero checks hold (Review Focus 1).
    squid = Circuit([Branch("J1", 1, 0, C=41.7e-15, EJ=12.3), Branch("J2", 1, 0, C=13.1e-15, EJ=7.7),
                     Branch("Cs", 1, 0, C=73.3e-15)], ground=0)
    res = quantize(squid, Loops.from_tree(squid, cq.default_tree(squid)))
    assert abs(float(cq.subs_constants(res.ham.E_C[0, 0])) - float(cq.subs_constants(cq.farad_to_ghz(128.1e-15)))) < 1e-9
    floating = Circuit([Branch("J", 1, 2, C=5e-15, EJ=10.0), Branch("C1", 1, 0, C=30e-15),
                        Branch("C2", 2, 0, C=40e-15)], ground=0)
    closure, G = cq.tree_to_G(floating, cq.default_tree(floating))
    res = quantize(floating, Loops(floating, G, closure, closure, dict.fromkeys(closure, 0)))
    assert res.coords.types == ("c",)


def test_quantize_requires_basis_and_types_together():
    circuit = kite()
    loops = Loops.from_branch_dict(circuit, KITE_CYCLES, {"cb": 0})
    for kwargs in ({"Z_full": KITE_Z}, {"types": ("c", "e", "e")}):
        with pytest.raises(ValueError, match="supplied together"):
            quantize(circuit, loops, **kwargs)


def test_quantize_rebuilds_charging_energy_for_changed_circuit():
    def transmon(C):
        return Circuit([Branch("J", 1, 0, C=C, EJ=1)], ground=0)

    a, b = transmon(1), transmon(2)
    first = quantize(a, Loops.from_branch_dict(a, {}))
    second = quantize(b, Loops.from_branch_dict(b, {}))
    assert second.ham.E_C == first.ham.E_C / 2
    with pytest.raises(TypeError, match="coords"):
        quantize(b, Loops.from_branch_dict(b, {}), coords=first.coords)
    with pytest.raises(ValueError, match="loops must belong"):
        quantize(b, Loops.from_branch_dict(a, {}))


def test_result_transform_reclassifies_residual_mixing_and_preserves_hamiltonian():
    circuit = kite()
    loops = Loops.from_branch_dict(circuit, KITE_CYCLES, {"cb": 0})
    original = quantize(circuit, loops)
    assert original.residual_freedom(keep_harmonic=False)[1, 2] != 0  # extended may mix into harmonic
    S = sym.Matrix([[1, 0, 0], [0, 1, 1], [0, 0, 1]])
    mixed = original.transform(S)
    assert mixed.coords.types == ("c", "e", "e")
    assert mixed.circuit is circuit and mixed.loops is loops
    substitution = dict(zip(original.ham.varphi, S * mixed.ham.varphi))
    substitution.update(zip(original.ham.n, S.inv().T * mixed.ham.n))
    assert sym.simplify(original.H.subs(substitution, simultaneous=True) - mixed.H) == 0
    restored = mixed.transform(S.inv())
    assert restored.coords.types == ("c", "e", "h")
    assert sym.simplify(restored.H - original.H) == 0



def test_simplify_hamiltonian_factors_coefficients_and_combines_cosines():
    a, b, x, y, n, EJ = sym.symbols("a b x y n E_J", real=True)
    raw = (a**2 - 1) * n**2 / (b**2 + 2*b + 1) - EJ*(sym.cos(x-y) + sym.cos(x+y))
    expected = (a-1)*(a+1)*n**2/(b+1)**2 - 2*EJ*sym.cos(x)*sym.cos(y)
    assert cq.simplify_hamiltonian(raw) == expected


def test_quantize_optional_simplification_preserves_zero_pi_hamiltonian():
    from circuits import zero_pi, ZERO_PI_CYCLES, ZERO_PI_Z
    circuit = zero_pi()
    loops = Loops.from_branch_dict(circuit, ZERO_PI_CYCLES, {"t1": 0, "t2": 0})
    args = (circuit, loops, ZERO_PI_Z, ("e", "c", "h", "cy"))
    raw = quantize(*args)
    simplified = quantize(*args, simplify=True)
    assert sym.cancel(sym.trigsimp(sym.expand(simplified.H - raw.H))) == 0
    assert simplified.H == cq.simplify_hamiltonian(raw.H)
    assert len(simplified.H.atoms(sym.cos)) == 2
    assert all(len(c.args[0].free_symbols) == 1 for c in simplified.H.atoms(sym.cos))
    assert simplified.ham.E_C == raw.ham.E_C


@pytest.mark.parametrize("time_dependent", [False, True], ids=["static", "driven"])
def test_fluxonium_custom_B_puts_flux_on_junction(time_dependent):
    from circuit_quantization.units import e

    EC, EL, EJ = pos("E_C E_L E_J")
    CJ = cq.ghz_to_farad(EC)
    circuit = Circuit([
        Branch("J", 1, 0, C=CJ, EJ=EJ),
        Branch("L", 1, 0, L=cq.ghz_to_henry(EL)),
    ], ground=0)
    phase_ext = (sym.Function("varphi_ext", real=True)(circuit.t) if time_dependent
                 else sym.Symbol("varphi_ext", real=True))
    flux = cq.phase_to_weber(phase_ext)
    loops = Loops.from_branch_dict(circuit, {"f": ("J", "L")}, {"f": flux})
    assert circuit.branch_names == ("J", "L")
    B_junction = sym.ImmutableMatrix([1, 0])  # G = [1, -1]
    result = quantize(circuit, loops, B=B_junction)
    default = quantize(circuit, loops)
    assert result.B == B_junction
    assert default.B == sym.Matrix([0, -1])
    assert loops.G * result.B == sym.eye(1)

    phase, charge = result.ham.varphi[0], result.ham.n[0]
    ng = CJ * sym.diff(flux, circuit.t) / (2 * e)
    expected = (4 * EC * (charge - ng)**2 + EL * phase**2 / 2
                - EJ * sym.cos(phase + phase_ext))
    assert sym.simplify(result.ham.n_g[0] - ng) == 0
    assert default.ham.n_g == sym.zeros(1, 1)
    assert sym.simplify(sym.expand(result.H - expected)) == 0
    # Default gauge puts the flux on L, with its flux-only scalar omitted.
    expected_default = (4 * EC * charge**2 + EL * phase**2 / 2
                        - EL * phase * phase_ext - EJ * sym.cos(phase))
    assert sym.simplify(sym.expand(default.H - expected_default)) == 0


def test_custom_assignment_must_satisfy_the_loop_constraint():
    circuit = Circuit([
        Branch("J", 1, 0, C=1, EJ=2), Branch("L", 1, 0, L=3),
    ], ground=0)
    loops = Loops.from_branch_dict(circuit, {"f": ("J", "L")})
    bad_B = sym.zeros(2, 1)
    assert loops.G * bad_B != sym.eye(1)  # eq:GB_constraint
    # The shape is correct, but accepting this assignment silently erases the
    # specified loop flux. quantize is the only entry point that takes a B.
    with pytest.raises(ValueError):
        quantize(circuit, loops, B=bad_B)


@pytest.mark.parametrize("scale", [1, 2], ids=["identity", "rescale"])
def test_result_transform_preserves_an_existing_driven_assignment(scale):
    circuit = Circuit([
        Branch("J", 1, 0, C=cq.ghz_to_farad(2), EJ=3),
        Branch("L", 1, 0, L=cq.ghz_to_henry(5)),
    ], ground=0)
    drive = sym.Function("flux_drive", real=True)(circuit.t)
    loops = Loops.from_branch_dict(circuit, {"f": ("J", "L")}, {"f": drive})
    original = quantize(circuit, loops, B=sym.Matrix([1, 0]))
    S = sym.Matrix([[scale]])
    # Sec III.C is a constant canonical relabeling, with the branch offsets held fixed.
    transformed = original.transform(S)
    assert transformed.B == original.B
    substitution = dict(zip(original.ham.varphi, S * transformed.ham.varphi))
    substitution.update(zip(original.ham.n, S.inv().T * transformed.ham.n))
    expected = original.H.subs(substitution, simultaneous=True)
    assert sym.simplify(transformed.H - expected) == 0


def test_result_transform_accepts_an_irrotational_B_equal_to_its_loop_constraint_only_after_cancel():
    # Zero-pi's irrotational B carries capacitance ratios: G B = I holds only after cancellation.
    circuit = zero_pi()
    result = quantize(circuit, Loops.from_branch_dict(circuit, ZERO_PI_CYCLES, {"t1": 0, "t2": 0}))
    assert result.loops.G * result.B != sym.eye(3)
    assert result.transform(sym.eye(3)).H == result.H



def split_fluxonium(offset):
    """Fluxonium at static half flux plus `offset`, loop f = J - L (G = [1, -1])."""
    EC, EL, EJ = pos("E_C E_L E_J")
    circuit = Circuit([
        Branch("J", 1, 0, C=cq.ghz_to_farad(EC), EJ=EJ),
        Branch("L", 1, 0, L=cq.ghz_to_henry(EL)),
    ], ground=0)
    loops = Loops.from_branch_dict(circuit, {"f": ("J", "L")}, {"f": cq.phase_to_weber(sym.pi + offset)})
    return circuit, loops, (EC, EL, EJ)


def test_static_split_puts_bias_on_junction_and_offset_on_inductor():
    # eq:particular_static_split: static pi on J, offset on the capacitance-free L.
    d = sym.Function("d", real=True)(sym.Symbol("t", real=True))  # Circuit's default time symbol
    circuit, loops, (EC, EL, EJ) = split_fluxonium(d)
    assert d.has(circuit.t)
    B_J, B_L = sym.ImmutableMatrix([1, 0]), sym.ImmutableMatrix([0, -1])
    result = quantize(circuit, loops, B=B_L, B_static=B_J, static={"f": cq.phase_to_weber(sym.pi)})

    assert result.ham.varphi_b == sym.Matrix([sym.pi, -d])
    assert not result.ham.n_g.has(sym.Derivative)  # only B sets the EMF drive
    assert quantize(circuit, loops, B=B_J).ham.n_g.has(sym.Derivative)  # whole flux on J drives it

    phase, charge, sector = result.ham.varphi[0], result.ham.n[0], result.ham.n_g[0]
    expected = 4 * EC * (charge - sector) ** 2 + EL * phase ** 2 / 2 - EL * d * phase + EJ * sym.cos(phase)
    assert sym.simplify(result.H - expected) == 0


def test_static_split_with_equal_assignments_matches_single_assignment():
    circuit, loops, _ = split_fluxonium(sym.Symbol("delta", real=True))
    B = sym.ImmutableMatrix([0, -1])
    split = quantize(circuit, loops, B=B, B_static=B, static={"f": cq.phase_to_weber(sym.pi)})
    assert sym.simplify(split.H - quantize(circuit, loops, B=B).H) == 0


def test_static_split_survives_transform():
    circuit, loops, _ = split_fluxonium(sym.Symbol("delta", real=True))
    static = {"f": cq.phase_to_weber(sym.pi)}
    result = quantize(circuit, loops, B=sym.ImmutableMatrix([0, -1]), B_static=sym.ImmutableMatrix([1, 0]),
                      static=static)
    moved = result.transform(sym.eye(1))
    assert (moved.B_static, moved.static) == (result.B_static, result.static)
    assert moved.H == result.H


def test_static_split_rejects_incomplete_or_invalid_input():
    circuit, loops, _ = split_fluxonium(sym.Symbol("delta", real=True))
    B, static = sym.ImmutableMatrix([0, -1]), {"f": cq.phase_to_weber(sym.pi)}
    with pytest.raises(ValueError, match="together"):
        quantize(circuit, loops, B=B, B_static=B)
    with pytest.raises(ValueError, match="together"):
        quantize(circuit, loops, B=B, static=static)
    with pytest.raises(ValueError, match="unknown loops"):
        quantize(circuit, loops, B=B, B_static=B, static={"g": 0})
    with pytest.raises(ValueError, match="B_static"):
        quantize(circuit, loops, B=B, B_static=sym.ImmutableMatrix([2, 0]), static=static)
