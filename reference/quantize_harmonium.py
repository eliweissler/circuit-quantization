"""Quantize the optimized non-degenerate noise-resilient qubit (harmonium).

Circuit and parameters: Hays, Kim, and Oliver, PRX Quantum 6, 040321
(2025), arXiv:2502.15459v1. The kite junction difference divided by their
mean is 1%, matching the optimization convention in the paper.
"""

from __future__ import annotations

import circuit_quantization as cq
import sympy as sp


JUNCTION_ASYMMETRY = sp.Rational(1, 100)
KITE_EJ_GHZ = sp.Integer(35)
FLUXONIUM_EJ_GHZ = sp.Integer(50)
KITE_EL_GHZ = sp.Integer(11)
FLUXONIUM_EL_GHZ = sp.Integer(14)
SHUNT_CAPACITANCE_F = sp.Rational(1500, 10**15)
CRITICAL_CURRENT_DENSITY_A_PER_M2 = sp.Integer(4_000_000)
SPECIFIC_CAPACITANCE_F_PER_M2 = sp.Rational(5, 100)
KITE_REDUCED_FLUX, FLUXONIUM_REDUCED_FLUX = sp.symbols(
    "varphi_ext_k varphi_ext_f", real=True
)
KITE_FLUX_WB = cq.phase_to_weber(KITE_REDUCED_FLUX)
FLUXONIUM_FLUX_WB = cq.phase_to_weber(FLUXONIUM_REDUCED_FLUX)

# Appendix coordinates phi_D = (varphi, varphi_f, varphi_Sigma_k,
# varphi_Delta_k). Rows follow this example's internal node order:
# (top, kite_left, kite_right, fluxonium).
PAPER_COORDINATES = sp.ImmutableMatrix(
    [
        [1, 0, 0, 0],
        [0, 1, 1, 1],
        [0, 1, -1, 1],
        [0, 1, 0, 0],
    ]
)
PAPER_MODE_TYPES = (
    cq.ModeType.COMPACT,
    cq.ModeType.EXTENDED,
    cq.ModeType.EXTENDED,
    cq.ModeType.EXTENDED,
)


def junction_capacitance(ej_ghz: sp.Expr) -> sp.Expr:
    """Paper's fixed EJ/CJ fabrication relation, returned in farads."""
    ej_joule = cq.ghz_to_joule(ej_ghz)
    energy_per_capacitance = (
        cq.REDUCED_FLUX_QUANTUM
        * CRITICAL_CURRENT_DENSITY_A_PER_M2
        / SPECIFIC_CAPACITANCE_F_PER_M2
    )
    return ej_joule / energy_per_capacitance


def harmonium(
    kite_flux_wb=KITE_FLUX_WB,
    fluxonium_flux_wb=FLUXONIUM_FLUX_WB,
) -> cq.Circuit:
    """Return harmonium with external loop fluxes in webers."""
    top, kite_left, kite_right, fluxonium, ground = range(5)
    ej_left = KITE_EJ_GHZ * (1 + JUNCTION_ASYMMETRY / 2)
    ej_right = KITE_EJ_GHZ * (1 - JUNCTION_ASYMMETRY / 2)
    branches = (
        cq.Branch("C_sh", top, ground, capacitance=SHUNT_CAPACITANCE_F),
        cq.Branch(
            "J_k_left", top, kite_left,
            capacitance=junction_capacitance(ej_left),
            josephson_energy=ej_left,
        ),
        cq.Branch(
            "L_k_left", kite_left, fluxonium,
            inductance=cq.ghz_to_henry(KITE_EL_GHZ),
        ),
        cq.Branch(
            "J_k_right", kite_right, top,
            capacitance=junction_capacitance(ej_right),
            josephson_energy=ej_right,
        ),
        cq.Branch(
            "L_k_right", fluxonium, kite_right,
            inductance=cq.ghz_to_henry(KITE_EL_GHZ),
        ),
        cq.Branch(
            "J_f", ground, fluxonium,
            capacitance=junction_capacitance(FLUXONIUM_EJ_GHZ),
            josephson_energy=FLUXONIUM_EJ_GHZ,
        ),
        cq.Branch(
            "L_f", fluxonium, ground,
            inductance=cq.ghz_to_henry(FLUXONIUM_EL_GHZ),
        ),
    )
    loops = (
        cq.LoopFlux(
            "kite_flux",
            {
                "J_k_left": 1,
                "L_k_left": 1,
                "L_k_right": 1,
                "J_k_right": 1,
            },
            kite_flux_wb,
        ),
        cq.LoopFlux(
            "fluxonium_flux", {"J_f": 1, "L_f": 1}, fluxonium_flux_wb
        ),
        # Complete the graph-cycle basis across the shunt capacitor at zero flux.
        cq.LoopFlux(
            "shunt_cycle",
            {"C_sh": 1, "J_f": 1, "L_k_left": -1, "J_k_left": -1},
            0,
        ),
    )
    return cq.Circuit(branches, ground=ground, loop_fluxes=loops)


def at_half_flux(hamiltonian):
    """Specialize a symbolic harmonium Hamiltonian to its double sweet spot."""
    return hamiltonian.subs(
        {
            KITE_REDUCED_FLUX: sp.pi,
            FLUXONIUM_REDUCED_FLUX: sp.pi,
        }
    )


if __name__ == "__main__":
    result = cq.quantize(
        harmonium(),
        # transformation="structural",
        # mode_types=PAPER_MODE_TYPES,
    )
    print("Paper coordinates (varphi, varphi_f, varphi_Sigma_k, varphi_Delta_k):")
    print("Node-flux transformation (phi_node = Z @ phi_paper):")
    print(result.coordinates.transformation)
    print("\nHamiltonian H/h [GHz]:")
    print(sp.N(result.hamiltonian, 6))
