import numpy as np
import pytest
import sympy as sym
from scipy.linalg import sqrtm

import circuit_quantization as cq
from circuit_quantization import Branch, Circuit, Loops, load_spec, quantize, to_spec
from circuit_quantization.product_state import (
    NumericHamiltonian,
    cost_product_state,
    default_bases,
    exact_ground_state,
    infidelity,
    normal_mode_transformation,
    numeric_hamiltonian,
    product_state,
    product_state_S1,
)


def capacitance_E_C(E_C0, coupling):
    """E_C matrix of two equal shunts, coupling capacitor ``coupling`` times a shunt."""
    C = np.array([[1 + coupling, -coupling], [-coupling, 1 + coupling]]) / E_C0
    return np.linalg.inv(C)


def harmonic_pair():
    E_C = capacitance_E_C(1.0, 0.2)
    E_L = np.array([[1.0, 0.3], [0.3, 2.0]])
    return NumericHamiltonian(("h", "h"), E_C, E_L, [0, 0], [0, 0], ())


def fluxonium_pair_half_flux():
    E_C = capacitance_E_C(1.0, 0.2)
    E_L = np.diag([0.5, 0.5])
    cosines = ((4.0, [1, 0], np.pi), (4.0, [0, 1], np.pi))
    return NumericHamiltonian(("e", "e"), E_C, E_L, [0, 0], [0, 0], cosines)


def transmon_fluxonium():
    E_C = capacitance_E_C(1.0, 0.2)
    E_L = np.diag([0.0, 0.5])
    cosines = ((10.0, [1, 0], 0.0), (4.0, [0, 1], np.pi))
    return NumericHamiltonian(("c", "e"), E_C, E_L, [0, 0], [0.25, 0], cosines)


def test_harmonic_pair_product_energy_at_normal_modes_is_exact():
    ham = harmonic_pair()
    omega = np.sqrt(np.linalg.eigvals(8 * ham.E_C @ ham.E_L).real)
    exact = 0.5 * omega.sum()
    S = normal_mode_transformation(ham)
    at_normal_modes = product_state(ham.transformed(S))
    assert at_normal_modes.converged
    assert at_normal_modes.energy == pytest.approx(exact, abs=1e-6)
    assert product_state(ham).energy > exact + 1e-4


def gaussian_width(ham):
    """``Omega`` of the harmonic ground state ``exp(-phi^T Omega phi / 2)``."""
    M = np.linalg.inv(8 * ham.E_C)
    M_half, M_minus_half = sqrtm(M), sqrtm(np.linalg.inv(M))
    return M_half @ sqrtm(M_minus_half @ ham.E_L @ M_minus_half) @ M_half


def test_harmonic_pair_optimizer_factorizes_the_ground_state():
    """The product energy alone fixes only the frame of the Gaussian: a flat valley."""
    ham = harmonic_pair()
    omega = np.sqrt(np.linalg.eigvals(8 * ham.E_C @ ham.E_L).real)
    S, info = product_state_S1(ham, relaxed=True, x0=[0.2, 0.1, 0.0, 0.1])
    assert info.energy == pytest.approx(0.5 * omega.sum(), abs=1e-5)
    width = S.T @ gaussian_width(ham) @ S
    assert abs(width[0, 1]) < 1e-2 * np.sqrt(width[0, 0] * width[1, 1])


def test_harmonic_pair_tie_break_selects_normal_modes():
    ham = harmonic_pair()
    omega = np.sqrt(np.linalg.eigvals(8 * ham.E_C @ ham.E_L).real)
    S, info = product_state_S1(ham, relaxed=True, regularization=1e-3)
    assert info.energy == pytest.approx(0.5 * omega.sum(), abs=1e-5)
    transformed = ham.transformed(S)
    assert abs(transformed.E_C[0, 1]) < 1e-2 * abs(transformed.E_C[0, 0])
    assert abs(transformed.E_L[0, 1]) < 1e-2 * abs(transformed.E_L[0, 0])


def test_cost_is_invariant_under_column_scaling_and_signs():
    ham = fluxonium_pair_half_flux()
    cost = cost_product_state(ham)
    S = np.array([[1.0, 0.3], [-0.2, 1.0]])
    assert cost(S @ np.diag([2.0, -0.5])) == pytest.approx(cost(S), rel=1e-5)


def test_fluxonium_pair_prefers_node_coordinates_to_normal_modes():
    ham = fluxonium_pair_half_flux()
    S_nm = normal_mode_transformation(ham)
    at_nodes = product_state(ham)
    at_normal_modes = product_state(ham.transformed(S_nm))
    assert at_nodes.converged and at_normal_modes.converged
    assert at_nodes.energy < at_normal_modes.energy - 0.05

    bases = default_bases(ham, max_points=120)
    E_exact, psi_exact = exact_ground_state(ham, bases)
    at_nodes = product_state(ham, bases=bases, adapt=False)
    assert at_nodes.energy >= E_exact - 1e-6
    assert infidelity(at_nodes, psi_exact) < 1e-2

    S, info = product_state_S1(ham, relaxed=True, x0=None)
    assert info.energy <= at_nodes.energy + 1e-6
    # The optimum is the node coordinates up to scale, sign, and permutation.
    off = np.abs(S) / np.abs(S).max(axis=0)
    assert np.sort(off.ravel())[:2].max() < 0.05


def test_exact_energy_is_coordinate_independent():
    ham = fluxonium_pair_half_flux()
    S = np.array([[1.0, 0.4], [0.1, 0.8]])
    E_ref, _ = exact_ground_state(ham, default_bases(ham, max_points=120))
    E_rot, _ = exact_ground_state(ham.transformed(S), default_bases(ham.transformed(S), max_points=120))
    assert E_rot == pytest.approx(E_ref, abs=2e-4)


def test_transmon_fluxonium_shear_keeps_compact_multiplier_integer():
    ham = transmon_fluxonium()
    S, info = product_state_S1(ham, relaxed=True)
    assert info.state.converged
    assert info.state.bases[0].compact
    transformed = ham.transformed(S)
    assert transformed.cosines[0][1][0] == pytest.approx(1.0)
    assert np.all(np.abs(transformed.E_L[0]) < 1e-9)
    bases = default_bases(transformed, charge_cutoff=12, max_points=100)
    E_exact, psi_exact = exact_ground_state(transformed, bases)
    state = product_state(transformed, bases=bases, adapt=False)
    assert state.energy >= E_exact - 1e-6
    assert infidelity(state, psi_exact) < 5e-2


def test_numeric_hamiltonian_from_quantized_circuit():
    EC1, EC2, ECc, EL1, EL2, EJ1, EJ2 = sym.symbols("E_C1 E_C2 E_Cc E_L1 E_L2 E_J1 E_J2", positive=True)
    ext = sym.Symbol("varphi_ext", real=True)
    circuit = Circuit([
        Branch("J1", 1, 0, C=cq.ghz_to_farad(EC1), EJ=EJ1),
        Branch("L1", 1, 0, L=cq.ghz_to_henry(EL1)),
        Branch("J2", 2, 0, C=cq.ghz_to_farad(EC2), EJ=EJ2),
        Branch("L2", 2, 0, L=cq.ghz_to_henry(EL2)),
        Branch("Cc", 1, 2, C=cq.ghz_to_farad(ECc)),
    ], ground=0)
    loops = Loops.from_tree(circuit, cq.default_tree(circuit),
                            {"L1": cq.phase_to_weber(ext), "L2": cq.phase_to_weber(ext)})
    res = quantize(circuit, loops)
    spec = load_spec(to_spec(res))
    ham = numeric_hamiltonian(spec, {"E_C1": 1.0, "E_C2": 1.0, "E_Cc": 5.0, "E_L1": 0.5,
                                     "E_L2": 0.5, "E_J1": 4.0, "E_J2": 4.0, "varphi_ext": np.pi,
                                     "Phi_Cc": 0.0})
    assert ham.types == ("e", "e")
    assert len(ham.cosines) == 2
    state = product_state(ham)
    assert state.converged
    S, info = product_state_S1(ham, relaxed=True)
    assert info.energy <= state.energy + 1e-6
