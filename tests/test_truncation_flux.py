"""L-C-L-C counterexample to using Z^+ after truncation (eq:Z_truncated)."""

import pytest
import sympy as sym

from circuit_quantization import Branch, Circuit, Loops, mode_basis, quantize


def box():
    circuit = Circuit([
        Branch("L1", 1, 2, L=2), Branch("C1", 2, 3, C=3),
        Branch("L2", 3, 4, L=5), Branch("C2", 4, 1, C=7),
    ], ground=0)
    flux = sym.Function("Phi_ext", real=True)(circuit.t)
    loops = Loops(circuit, {"box": ("L1", "C1", "L2", "C2")}, {"box": flux})
    result = quantize(circuit, loops)
    Z_full, types = mode_basis(result.lag)
    assert types == ("h", "fr", "fz", "cy")
    return result, Z_full[:, 1:]


def test_box_coordinate_extraction_ignores_discarded_flux():
    result, discarded = box()
    Z, R = result.coords.Z, result.coords.R
    # Independent definition: retained rows of the inverse full decoupled basis.
    assert R == Z.row_join(discarded).inv()[:1, :]
    assert R == sym.Matrix([[-1, 1, -1, 1]]) / 4
    assert R * Z == sym.eye(1)
    assert R * discarded == sym.zeros(1, 3)
    # Previous theoretical prescription spuriously extracts a retained flux.
    frozen = discarded[:, 1]
    flux = result.loops.Phi_ext[0]
    assert (Z.pinv() * frozen * flux)[0] == sym.Rational(525, 1646) * flux
    assert R * frozen * flux == sym.zeros(1, 1)


@pytest.mark.parametrize("sector", [0, 1, 2])
def test_box_gauge_shift_cancels_emf_including_discarded_directions(sector):
    result, discarded = box()
    lag, coords, loops = result.lag, result.coords, result.loops
    G_plus = loops.G.T * (loops.G * loops.G.T).inv()
    drive = coords.C_tilde_inv * coords.Z.T * lag.Omega_int * lag.C_b * G_plus
    # eq:Lambda_general: arbitrary discarded shifts must remain homogeneous.
    Lambda = -coords.Z * drive + discarded[:, sector]
    assert coords.R * discarded[:, sector] == sym.zeros(1, 1)
    assert (drive + coords.R * Lambda) * loops.Phi_ext.diff(result.circuit.t) == sym.zeros(1, 1)
    B_shifted = result.B + lag.Omega_int.T * discarded[:, sector]
    assert loops.G * B_shifted == sym.eye(1)
    assert coords.Z.T * lag.Omega_int * lag.C_b * B_shifted == sym.zeros(1, 1)
    assert coords.Z.T * lag.Lstar_int * discarded[:, sector] == sym.zeros(1, 1)


def test_box_frozen_equilibrium_changes_only_discarded_energy():
    result, discarded = box()
    lag, coords = result.lag, result.coords
    x, eta = sym.symbols("x eta", real=True)
    flux = result.loops.Phi_ext[0]
    branch_flux = (lag.omega_L_int.T * (coords.Z * x + discarded[:, 1] * eta)
                   + result.B[list(lag.L), :] * flux)
    energy = (branch_flux.T * lag.Lb_inv * branch_flux)[0] / 2
    # A driven frozen coordinate need not be zero. Its elimination changes
    # only a scalar offset, since the dynamical and frozen blocks decouple.
    equilibrium = sym.solve(sym.diff(energy, eta), eta)[0]
    assert equilibrium == -3 * flux / 56
    correction = sym.expand(energy.subs(eta, equilibrium) - energy.subs(eta, 0))
    assert not correction.has(x)
    assert sym.diff(energy, eta).subs(eta, equilibrium) == 0


def test_R_transforms_contragrediently():
    result, _ = box()
    transformed = result.transform(sym.Matrix([[2]]))
    assert transformed.coords.R == result.coords.R / 2
    assert transformed.B == result.B


def test_R_selects_retained_rows_when_modes_are_interleaved():
    from circuit_quantization import decouple

    result, _ = box()
    original, types = mode_basis(result.lag)
    order = [1, 3, 0, 2]  # free, cyclic, retained harmonic, frozen
    coords = decouple(result.lag, original[:, order], tuple(types[i] for i in order))
    assert coords.Z == result.coords.Z
    assert coords.R == result.coords.R
    full = original[:, order].as_mutable()
    full[:, 2] = coords.Z
    assert coords.R == full.inv()[[2], :]
    assert coords.R == coords.C_tilde_inv * coords.Z.T * result.lag.C_int


def test_coordinates_retain_and_classify_the_full_decoupled_basis():
    from circuit_quantization import coordinates, mode_types

    result, discarded = box()
    full = result.coords.Z.row_join(discarded)
    order = [1, 3, 0, 2]
    full = full[:, order]
    coords = coordinates(result.lag, full)
    assert coords.Z_full == full
    assert coords.types_full == mode_types(result.lag, full) == ("fr", "cy", "h", "fz")
    assert coords.types == mode_types(result.lag, coords.Z) == ("h",)
    assert coords.Z == result.coords.Z
    assert coords.R == full.inv()[[2], :]
    assert coords.R * full[:, [0, 1, 3]] == sym.zeros(1, 3)
    with pytest.raises(ValueError, match="including non-dynamical"):
        coordinates(result.lag, coords.Z)


def test_residual_transform_preserves_discarded_columns():
    result, discarded = box()
    transformed = result.transform(sym.Matrix([[2]]))
    assert transformed.coords.Z_full[:, 1:] == discarded
    assert transformed.coords.Z_full[:, :1] == 2 * result.coords.Z
    assert transformed.coords.types_full == result.coords.types_full
    assert transformed.coords.R == transformed.coords.Z_full.inv()[:1, :]
