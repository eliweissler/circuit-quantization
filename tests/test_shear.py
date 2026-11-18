import numpy as np
import pytest

import circuit_quantization as cq
from circuit_quantization import Branch, Circuit, Loops, NumericHamiltonian, numeric_hamiltonian, quantize
from circuit_quantization.shear import shear_S1


def junction_frame(EC, EL, EJ, theta, ECf, ELf, kappa=0.25):
    """One junction mode x and one harmonic frame mode f coupled by g x f (docs/shear-rule.md, the toy model)."""
    g = kappa * np.sqrt(EL * ELf)
    return NumericHamiltonian(types=("e", "h"), E_C=np.diag([EC, ECf]), E_L=np.array([[EL, g], [g, ELf]]),
                              flux_linear=np.zeros(2), n_g=np.zeros(2), cosines=((EJ, np.array([1.0, 0.0]), theta),))


FLUXONIUM = dict(EC=0.5534, EL=1.6346, EJ=8.98, theta=np.pi)
COUPLER = dict(EC=0.7748, EL=12.113, EJ=3.5, theta=2 * np.pi * 0.3)


def test_harmonic_junction_gives_the_exact_fraction():
    # with no junction energy the leaf is an oscillator, and the rule is exact: s = Omega / (Omega + omega), both in
    # the sheared frame, self-consistently
    ham = junction_frame(EC=0.5, EL=2.0, EJ=1e-12, theta=0.0, ECf=0.4, ELf=30.0)
    S, info = shear_S1(ham, [(1, 0)])
    sheared = ham.transformed(S)
    Omega = np.sqrt(8 * sheared.E_C[1, 1] * sheared.E_L[1, 1])
    omega = np.sqrt(8 * sheared.E_C[0, 0] * sheared.E_L[0, 0])
    row = info.pairs[0]
    assert row.s == pytest.approx(Omega / (Omega + omega), rel=1e-6)
    assert row.omega1 == pytest.approx(omega, rel=1e-6) and row.omega2 == pytest.approx(omega, rel=1e-6)
    assert row.a_star == pytest.approx(-ham.E_L[0, 1] / ham.E_L[1, 1])
    assert S[1, 0] == pytest.approx(row.s * row.a_star) and np.count_nonzero(S - np.eye(2)) == 1
    assert info.converged


@pytest.mark.parametrize("junction, ELf, expected", [
    (FLUXONIUM, 68.4, 0.971),   # a stiff frame mode against a fluxonium at half flux: almost the full shear
    (COUPLER, 3.27, 0.277),     # a soft frame mode against a single-well coupler: about a quarter
])
def test_matches_the_toy_model(junction, ELf, expected):
    # docs/shear-rule.md, section "Toy model", column s* (3), from an independent solve (sinc DVR)
    ham = junction_frame(**junction, ECf=0.3874, ELf=ELf)
    _, info = shear_S1(ham, [(1, 0)])
    assert info.pairs[0].s == pytest.approx(expected, abs=2e-3)
    assert info.junctions[0].edge_weight < 1e-8


def test_fixed_iterations_and_report():
    ham = junction_frame(**FLUXONIUM, ECf=0.3874, ELf=68.4)
    _, one = shear_S1(ham, [(1, 0)], maxiter=1, tol=0.0)
    _, full = shear_S1(ham, [(1, 0)])
    assert one.iterations == 1 and not one.converged
    assert full.converged and abs(one.pairs[0].s - full.pairs[0].s) > 1e-6
    text = str(full)
    assert "omega1" in text and "s*" in text


def test_two_frame_modes_on_one_junction():
    g1, g2 = 0.25 * np.sqrt(1.6346 * 68.4), 0.25 * np.sqrt(1.6346 * 20.0)
    ham = NumericHamiltonian(types=("e", "h", "h"), E_C=np.diag([0.5534, 0.3874, 0.3874]),
                             E_L=np.array([[1.6346, g1, g2], [g1, 68.4, 0.0], [g2, 0.0, 20.0]]),
                             flux_linear=np.zeros(3), n_g=np.zeros(3), cosines=((8.98, np.array([1.0, 0, 0]), np.pi),))
    S, info = shear_S1(ham, [(1, 0), (2, 0)])
    assert len(info.pairs) == 2 and S[1, 0] != 0 and S[2, 0] != 0 and S[1, 2] == 0
    assert 0 < info.pairs[1].s < info.pairs[0].s < 1                                 # the softer mode shears less


@pytest.mark.parametrize("pairs, match", [
    ([(0, 1)], "harmonic"),                    # f must be harmonic
    ([(1, 1)], "distinct"),
    ([(1, 0), (1, 0)], "twice"),
])
def test_rejects_invalid_pairs(pairs, match):
    ham = junction_frame(**FLUXONIUM, ECf=0.3874, ELf=68.4)
    with pytest.raises(ValueError, match=match):
        shear_S1(ham, pairs)


def test_rejects_a_harmonic_junction_slot():
    ham = NumericHamiltonian(("h", "h"), np.diag([0.5, 0.4]), np.array([[2.0, 1.0], [1.0, 30.0]]), np.zeros(2),
                             np.zeros(2), ())
    with pytest.raises(ValueError, match="junction"):
        shear_S1(ham, [(1, 0)])


def test_rejects_what_the_rule_does_not_cover():
    base = junction_frame(**FLUXONIUM, ECf=0.3874, ELf=68.4)
    uncoupled = NumericHamiltonian(base.types, base.E_C, np.diag(np.diag(base.E_L)), base.flux_linear, base.n_g, base.cosines)
    with pytest.raises(ValueError, match="nothing to shear"):
        shear_S1(uncoupled, [(1, 0)])
    E_C = base.E_C.copy(); E_C[0, 1] = E_C[1, 0] = 0.01
    charge_coupled = NumericHamiltonian(base.types, E_C, base.E_L, base.flux_linear, base.n_g, base.cosines)
    with pytest.raises(ValueError, match="E_C"):
        shear_S1(charge_coupled, [(1, 0)])
    two_mode = NumericHamiltonian(("e", "e", "h"), np.diag([0.55, 0.55, 0.39]),
                                  np.array([[1.6, 0, 3.0], [0, 1.6, 0], [3.0, 0, 68.4]]), np.zeros(3), np.zeros(3),
                                  ((8.98, np.array([1.0, 0, 0]), np.pi), (2.0, np.array([1.0, -1.0, 0]), 0.0)))
    with pytest.raises(ValueError, match="single-mode"):
        shear_S1(two_mode, [(2, 0)])


def test_result_transform_agrees_with_the_numeric_frame():
    # a grounded fluxonium (node 1) inductively coupled to an LC mode (node 2), quantized by circuit-quantization;
    # the shear applied to the Result gives the same Hamiltonian as applied to its numeric form
    circuit = Circuit([Branch("J", 1, 0, C=cq.ghz_to_farad(0.5534), EJ=8.98),
                       Branch("L1", 1, 0, L=cq.ghz_to_henry(1.0)),
                       Branch("Lc", 1, 2, L=cq.ghz_to_henry(2.0)),
                       Branch("C2", 2, 0, C=cq.ghz_to_farad(0.3874)),
                       Branch("L2", 2, 0, L=cq.ghz_to_henry(40.0))], ground=0)
    loops = Loops.from_tree(circuit, cq.default_tree(circuit))
    loops = Loops(circuit, loops.G, loops.names, loops.closure_branches, dict.fromkeys(loops.names, 0))
    res = quantize(circuit, loops, np.eye(2), ("e", "h"))
    ham = numeric_hamiltonian(res)
    assert ham.types == ("e", "h") and ham.E_L[0, 1] != 0 and ham.E_C[0, 1] == 0
    S, info = shear_S1(ham, [(1, 0)])
    expected = numeric_hamiltonian(res.transform(S))
    actual = ham.transformed(S)
    np.testing.assert_allclose(actual.E_C, expected.E_C, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(actual.E_L, expected.E_L, rtol=1e-10, atol=1e-12)
    assert 0 < info.pairs[0].s < 1
