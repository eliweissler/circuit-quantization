"""End-to-end symbolic circuit quantization."""

from __future__ import annotations

from dataclasses import dataclass

from .coordinates import (
    CoordinateSystem,
    choose_coordinates,
    coordinates_from_transformation,
)
from .flux import FluxAssignment, assign_flux
from .hamiltonian import HamiltonianData, build_hamiltonian
from .matrices import CircuitMatrices, assemble
from .model import Circuit


@dataclass(frozen=True)
class QuantizationResult:
    matrices: CircuitMatrices
    coordinates: CoordinateSystem
    flux_assignment: FluxAssignment
    hamiltonian_data: HamiltonianData

    @property
    def hamiltonian(self):
        """The symbolic Hamiltonian H/h in GHz."""
        return self.hamiltonian_data.hamiltonian


def quantize(
    circuit: Circuit,
    coordinate_strategy: str = "structural",
    gauge: str = "irrotational",
    *,
    transformation=None,
    mode_types=None,
) -> QuantizationResult:
    matrices = assemble(circuit)
    if transformation is None:
        if mode_types is not None:
            raise ValueError("mode_types requires an explicit transformation")
        coordinates = choose_coordinates(matrices, strategy=coordinate_strategy)
    else:
        if coordinate_strategy != "structural":
            raise ValueError(
                "coordinate_strategy cannot be combined with an explicit transformation"
            )
        if mode_types is None:
            raise ValueError("Explicit transformations require mode_types")
        coordinates = coordinates_from_transformation(
            matrices, transformation, mode_types
        )
    flux_assignment = assign_flux(matrices, coordinates, gauge=gauge)
    hamiltonian = build_hamiltonian(matrices, coordinates, flux_assignment)
    return QuantizationResult(
        matrices=matrices,
        coordinates=coordinates,
        flux_assignment=flux_assignment,
        hamiltonian_data=hamiltonian,
    )
