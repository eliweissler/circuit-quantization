"""Branch-centric circuit data model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Hashable, Mapping

import sympy as sp


def _expr(value):
    return sp.sympify(value)


def _known_negative(value) -> bool:
    return value.is_negative is True


def _known_nonpositive(value) -> bool:
    return value.is_nonpositive is True


@dataclass(frozen=True)
class Branch:
    """One oriented circuit branch.

    Capacitance is in F, inductance in H, Josephson energy is E_J/h in
    GHz, and voltage is in V.  A branch may carry capacitance plus at most
    one flux-dependent element.
    """

    name: str
    tail: Hashable
    head: Hashable
    capacitance: object = 0
    inductance: object | None = None
    josephson_energy: object | None = None
    voltage: object | None = None

    def __post_init__(self):
        object.__setattr__(self, "capacitance", _expr(self.capacitance))
        for attr in ("inductance", "josephson_energy", "voltage"):
            value = getattr(self, attr)
            if value is not None:
                object.__setattr__(self, attr, _expr(value))

        if not self.name:
            raise ValueError("Branch names must be nonempty")
        if self.tail == self.head:
            raise ValueError(f"Branch {self.name!r} is a self-loop")
        if _known_negative(self.capacitance):
            raise ValueError(f"Branch {self.name!r} has negative capacitance")

        kinds = sum(
            value is not None
            for value in (self.inductance, self.josephson_energy, self.voltage)
        )
        if kinds > 1:
            raise ValueError(
                f"Branch {self.name!r} has more than one non-capacitive element"
            )
        if self.inductance is not None and _known_nonpositive(self.inductance):
            raise ValueError(f"Branch {self.name!r} has nonpositive inductance")
        if (
            self.josephson_energy is not None
            and _known_nonpositive(self.josephson_energy)
        ):
            raise ValueError(
                f"Branch {self.name!r} has nonpositive Josephson energy"
            )
        if self.josephson_energy is not None and self.capacitance.is_positive is not True:
            raise ValueError(
                f"Junction branch {self.name!r} requires a provably positive capacitance"
            )
        if self.voltage is not None and self.capacitance != 0:
            raise ValueError(f"Voltage-source branch {self.name!r} must have C=0")
        if kinds == 0 and self.capacitance.is_positive is not True:
            raise ValueError(
                f"Pure-capacitor branch {self.name!r} requires a provably positive capacitance"
            )

    @property
    def kind(self) -> str:
        if self.josephson_energy is not None:
            return "junction"
        if self.inductance is not None:
            return "inductor"
        if self.voltage is not None:
            return "source"
        return "capacitor"


@dataclass(frozen=True)
class LoopFlux:
    """An oriented physical loop and the flux threading it in webers."""

    name: str
    cycle: Mapping[str, int]
    value: object = 0

    def __post_init__(self):
        cycle = dict(self.cycle)
        if not cycle:
            raise ValueError("Loop cycles must contain at least one branch")
        if any(value not in (-1, 1) for value in cycle.values()):
            raise ValueError("Loop orientations must be +1 or -1")
        object.__setattr__(self, "cycle", cycle)
        object.__setattr__(self, "value", _expr(self.value))


@dataclass(frozen=True)
class Circuit:
    branches: tuple[Branch, ...]
    ground: Hashable
    loop_fluxes: tuple[LoopFlux, ...] = field(default_factory=tuple)
    time: sp.Symbol = field(default_factory=lambda: sp.Symbol("t", real=True))

    def __post_init__(self):
        object.__setattr__(self, "branches", tuple(self.branches))
        object.__setattr__(self, "loop_fluxes", tuple(self.loop_fluxes))
        object.__setattr__(self, "time", sp.sympify(self.time))
        if not self.branches:
            raise ValueError("A circuit requires at least one branch")
        names = [branch.name for branch in self.branches]
        if len(names) != len(set(names)):
            raise ValueError("Branch names must be unique")

        source_nodes = []
        for branch in self.branches:
            if branch.kind != "source":
                continue
            if self.ground not in (branch.tail, branch.head):
                raise ValueError(
                    f"Voltage source {branch.name!r} must connect to ground"
                )
            source_nodes.append(
                branch.head if branch.tail == self.ground else branch.tail
            )
        if len(source_nodes) != len(set(source_nodes)):
            raise ValueError("Each source node must have exactly one voltage source")

        source_set = set(source_nodes)
        for branch in self.branches:
            if branch.kind == "source":
                continue
            if source_set.intersection((branch.tail, branch.head)) and branch.kind != "capacitor":
                raise ValueError(
                    f"Only coupling capacitors may terminate on a source node: {branch.name!r}"
                )

