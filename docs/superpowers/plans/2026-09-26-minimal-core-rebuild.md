# Minimal Core Rebuild Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the Codex-built package with a minimal symbolic implementation whose modules follow the sections of `full_matrix_quantization.tex` one-to-one.

**Architecture:** Plain functions over frozen dataclasses, one module per section of the notes: `units` (Sec VI), `circuit` (Sec I, App A), `lagrangian` (Sec II), `modes` (Sec III), `flux` (Sec V), `hamiltonian` (Sec IV/VI), and a thin `quantize` driver.
Every intermediate matrix keeps its notes name.
Tests pin the closed forms printed in the notes (Appendix B and the body examples).

**Tech Stack:** Python 3.11, sympy 1.14, networkx 3.6, pytest; scqubits and nbformat/nbconvert (already in the `cqed` conda env) for one cross-check and the example notebooks.

**Spec:** `docs/superpowers/specs/2026-09-26-minimal-core-rebuild-design.md`

## Global Constraints

- Inputs SI: `C` [F], `L` [H], source `V` [V] (source-node potential relative to ground), loop flux [Wb], time [s]; `EJ` and `H` in GHz (`E/h`).
- Node order internal | ground | source; branch order C | J | L | V, user order kept within each group (eq:Omega_columns).
- `Omega` column: `+1` at tail, `-1` at head.
- Every cycle's flux is a visible symbol `Phi_<name>` unless set explicitly; `Loops` requires a complete cycle basis.
- Comments cite notes `\label` names (e.g. `eq:B_irrot_direct_closed`), never printed equation numbers.
- Dependencies: `sympy`, `networkx` only. Test extras: `pytest`, optional `scqubits`.
- No vendored optimizer, no global `simplify` of `H`.
- Run everything in the `cqed` conda env: `conda run -n cqed ...`.

## Review Focus

These are inputs the spec implies but does not name; each has a pinning test in the task that owns the code.

1. Numeric (float) element values typed straight into a notebook, e.g. `C=97e-15`: must quantize like symbols — `test_transmon_without_loops_and_with_float_parameters` (Task 9).
2. Circuits with no loops, no junctions, or no inductors (single transmon, LC oscillator): empty matrices must flow through — same test and `test_lc_oscillator_is_harmonic` (Task 9).
3. Floating circuits whose ground carries no branch (0-pi): cycle count uses `N_b - rank(Omega)` and one root per component — `test_floating_circuit_cycle_count` (Task 3), `test_zero_pi_given_Z_matches_notes` (Task 6).
4. User-supplied `Z` with symbolic entries (rhombus with unequal `C_g`): integrality checks must simplify first — `test_rhombus_decoupling_produces_weighted_centroid` (Task 6).
5. A loop that closes through a voltage source (gate-driven transmon): `Loops` must demand it and the drive must survive — `test_voltage_drive_offset_charge_sign` (Task 9), `test_default_cycles_of_coupled_example` (Task 3).

## Notes for the implementer

- All code and tests below were run in the `cqed` env before this plan was written: replaying Tasks 1-9 in order, each task's new tests fail before its implementation step and the whole suite passes after it (3, 10, 16, 18, 23, 32, 36, 46, 54 tests).
  Copy code exactly; if a test fails, stop and report rather than editing the expected values, which come from the notes.
- Mode types are the notes' superscripts as strings: `"c"`, `"e"`, `"h"` (dynamical) and `"fr"`, `"fz"`, `"cy"` (non-dynamical).
- `decouple` realizes eq:ZY_blocks one column and one non-dynamical mode at a time: it first Gram-Schmidt-orthogonalizes the free block in `C_int` and the frozen block in `L*_int`, so every division is by a positive scalar. `test_mode_by_mode_decoupling_equals_block_formula` checks it against `Z_1 + Z_2 M`.
- Each compact mode gets a static sector symbol `n_g<j>` in `n_g` (Sec IV.B.1, as in eq:harm_H_dimless).
- The old implementation is parked under `reference/` in Task 1 so it stays readable while the rebuild proceeds; Task 12 deletes it.

---

### Task 1: Branch, snapshot, scaffold, and units
**Files:**
- Modify: `.gitignore` (append `.DS_Store`)
- Move: `src/circuit_quantization/` -> `reference/old_package/`, `tests/` -> `reference/old_tests/`, `examples/quantize_harmonium.py` -> `reference/`
- Move: `examples/*.nb` -> `examples/mathematica/`
- Modify: `pyproject.toml` (full replacement)
- Create: `src/circuit_quantization/__init__.py`, `src/circuit_quantization/units.py`
- Test: `tests/test_units.py`

**Interfaces:**
- Produces: `circuit_quantization.units` with exact SI rationals `e`, `h`, `hbar`, `Phi0`, `phi0`, `GHz`, and converters `farad_to_ghz(C)`, `ghz_to_farad(E_C)`, `henry_to_ghz(L)`, `ghz_to_henry(E_L)`, `weber_to_phase(Phi)`, `phase_to_weber(varphi)`, all re-exported at package top level.

- [ ] **Step 1: Create the branch and snapshot the current state**

```bash
cd /Users/eweissler/src/circuit-quantization
git checkout -b minimal-core
printf '\n# macOS\n.DS_Store\n' >> .gitignore
git add -A
git commit -m "Snapshot Codex implementation, spec, and plan before rebuild"
```

- [ ] **Step 2: Move the old implementation out of the import path**

```bash
git mv src/circuit_quantization reference/old_package
git mv tests reference/old_tests
git mv examples/quantize_harmonium.py reference/quantize_harmonium.py
mkdir -p examples/mathematica
git mv examples/*.nb examples/mathematica/
mkdir -p src/circuit_quantization tests
```

- [ ] **Step 3: Replace `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=69"]
build-backend = "setuptools.build_meta"

[project]
name = "circuit-quantization"
version = "0.2.0"
description = "Symbolic quantization of superconducting circuits, following the matrix-formulation notes"
readme = "README.md"
requires-python = ">=3.11"
license = "GPL-3.0-only"
authors = [{name = "Eli Weissler"}]
dependencies = ["networkx>=3.4", "sympy>=1.13"]

[project.optional-dependencies]
test = ["pytest>=8.3", "scqubits>=4.2"]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra"
```

- [ ] **Step 4: Write the failing tests**

Create `tests/test_units.py`:

```python
import sympy as sym

import circuit_quantization as cq
from circuit_quantization import units


def test_constants_are_exact_si():
    assert units.e == sym.Rational("1.602176634e-19")
    assert units.h == sym.Rational("6.62607015e-34")
    assert units.phi0 == units.h / (4 * sym.pi * units.e)


def test_converters_round_trip():
    E = sym.Symbol("E", positive=True)
    assert sym.simplify(cq.farad_to_ghz(cq.ghz_to_farad(E)) - E) == 0
    assert sym.simplify(cq.henry_to_ghz(cq.ghz_to_henry(E)) - E) == 0
    assert sym.simplify(cq.weber_to_phase(cq.phase_to_weber(E)) - E) == 0


def test_transmon_scale():
    # 20 GHz^-1 ... sanity: E_C = 0.2 GHz is about 97 fF.
    C = float(cq.ghz_to_farad(sym.Rational(1, 5)))
    assert 96e-15 < C < 98e-15
```

- [ ] **Step 5: Run the new tests to verify they fail**

Run: `MPLCONFIGDIR=/tmp/cq-mpl conda run -n cqed python -m pytest tests/test_units.py -q -p no:cacheprovider`

Expected: FAIL during collection, e.g. `ModuleNotFoundError: No module named 'circuit_quantization'`.

- [ ] **Step 6: Implement**

Create `src/circuit_quantization/units.py`:

```python
"""Physical constants and unit conversions (Sec VI).

Circuit inputs are SI (F, H, V, Wb, s).
Energies and the Hamiltonian are ordinary frequencies E/h in GHz.
"""

import sympy as sym

e = sym.Rational("1.602176634e-19")  # elementary charge [C]
h = sym.Rational("6.62607015e-34")  # Planck constant [J s]
hbar = h / (2 * sym.pi)
Phi0 = h / (2 * e)  # flux quantum [Wb]
phi0 = hbar / (2 * e)  # reduced flux quantum [Wb], eq:dimless_defs
GHz = sym.Integer(10) ** 9


# eq:EC_EL_scalar: h E_C = e^2 / 2C
def farad_to_ghz(C):
    return e**2 / (2 * h * C * GHz)


def ghz_to_farad(E_C):
    return e**2 / (2 * h * E_C * GHz)


# eq:EC_EL_scalar: h E_L = phi0^2 / L
def henry_to_ghz(L):
    return phi0**2 / (h * L * GHz)


def ghz_to_henry(E_L):
    return phi0**2 / (h * E_L * GHz)


# eq:dimless_defs: varphi = Phi / phi0
def weber_to_phase(Phi):
    return Phi / phi0


def phase_to_weber(varphi):
    return varphi * phi0
```

Create `src/circuit_quantization/__init__.py`:

```python
"""Symbolic quantization of superconducting circuits (rebuild in progress)."""

from .units import (
    farad_to_ghz,
    ghz_to_farad,
    ghz_to_henry,
    henry_to_ghz,
    phase_to_weber,
    weber_to_phase,
)
```

- [ ] **Step 7: Install the package in editable mode**

```bash
conda run -n cqed python -m pip install -e . -q
```

- [ ] **Step 8: Run the whole suite to verify it passes**

Run: `MPLCONFIGDIR=/tmp/cq-mpl conda run -n cqed python -m pytest tests -q -p no:cacheprovider`

Expected: PASS, 3 passed.

- [ ] **Step 9: Commit**

```bash
git add -A
git commit -m "Rebuild: scaffold package and unit conversions (Sec VI)" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```


---

### Task 2: Branch, Circuit, and the incidence matrix (Sec I.A-B)
**Files:**
- Create: `src/circuit_quantization/circuit.py`
- Modify: `src/circuit_quantization/__init__.py`
- Create: `tests/circuits.py` (notes circuits shared by all tests)
- Test: `tests/test_circuit.py`

**Interfaces:**
- Consumes: nothing beyond sympy.
- Produces: `Branch(name, tail, head, C=0, L=None, EJ=None, V=None)` with `.type` in `('C','J','L','V')`; `Circuit(branches, ground, t=Symbol('t', real=True))` with `.branches` (sorted C|J|L|V), `.internal_nodes`, `.source_nodes`, `.nodes` (int | gnd | src), `.branch_names`; `incidence(circuit) -> ImmutableMatrix` (Omega).
- Produces (tests): `tests/circuits.py` with `harmonium()`, `zero_pi()`, `kite()`, `rhombus()`, `coupled()`, `parallel(*branches)`, `pos(names)`, and constants `HARMONIUM_CYCLES`, `HARMONIUM_Z`, `ZERO_PI_CYCLES`, `ZERO_PI_Z`, `KITE_CYCLES`, `KITE_Z`, `RHOMBUS_CYCLES`, `RHOMBUS_Z`, `coupled_zero_pis()`, `COUPLED_ZERO_PI_Z`, `COUPLED_ZERO_PI_TYPES`.

- [ ] **Step 1: Write the failing tests**

Create `tests/circuits.py`:

```python
"""Circuits from the notes, built exactly as drawn there."""

import sympy as sym

from circuit_quantization import Branch, Circuit

pos = lambda names: sym.symbols(names, positive=True)  # noqa: E731


def harmonium():
    """fig:harmonium, branches oriented low -> high node (eq:harm_Omega)."""
    C, CJ1, CJ2, CJ3, L1, L2, L3, EJ1, EJ2, EJ3 = pos("C C_J1 C_J2 C_J3 L_1 L_2 L_3 E_J1 E_J2 E_J3")
    return Circuit([
        Branch("C", 1, 5, C=C),
        Branch("J1", 1, 2, C=CJ1, EJ=EJ1),
        Branch("J2", 1, 3, C=CJ2, EJ=EJ2),
        Branch("J3", 4, 5, C=CJ3, EJ=EJ3),
        Branch("L1", 2, 4, L=L1),
        Branch("L2", 3, 4, L=L2),
        Branch("L3", 4, 5, L=L3),
    ], ground=5)


HARMONIUM_CYCLES = {  # eq:harm_G
    "k": {"J1": 1, "J2": -1, "L1": 1, "L2": -1},
    "f": {"J3": 1, "L3": -1},
    "cb": {"C": 1, "J1": -1, "J3": -1, "L1": -1},
}
HARMONIUM_Z = sym.Matrix([[1, 0, 0, 0], [0, 1, 1, 1], [0, 1, -1, 1], [0, 1, 0, 0]])  # eq:harm_Z


def zero_pi():
    """fig:zeropi_kite left, symmetric; ground node 5 is isolated (eq:zp_Omega)."""
    CJ, C, L, EJ = pos("C_J C L E_J")
    return Circuit([
        Branch("J1", 1, 2, C=CJ, EJ=EJ),
        Branch("J2", 3, 4, C=CJ, EJ=EJ),
        Branch("L1", 2, 3, L=L),
        Branch("L2", 4, 1, L=L),
        Branch("C1", 1, 3, C=C),
        Branch("C2", 2, 4, C=C),
    ], ground=5)


ZERO_PI_CYCLES = {  # eq:zp_G
    "ring": {"J1": 1, "J2": 1, "L1": 1, "L2": 1},
    "t1": {"J1": 1, "L1": 1, "C1": -1},
    "t2": {"J2": 1, "L1": 1, "C2": -1},
}
ZERO_PI_Z = sym.Matrix([[-1, -1, 1, 1], [1, 1, 1, 1], [-1, 1, -1, 1], [1, -1, -1, 1]]) / 2  # eq:zp_Z


def kite():
    """cos(2 phi) transmon, island 1, arm midpoints 2 and 3, ground 4 (eq:kite_node)."""
    CS, CJ1, CJ2, L1, L2, EJ1, EJ2 = pos("C_S C_J1 C_J2 L_1 L_2 E_J1 E_J2")
    return Circuit([
        Branch("CS", 1, 4, C=CS),
        Branch("J1", 1, 2, C=CJ1, EJ=EJ1),
        Branch("J2", 1, 3, C=CJ2, EJ=EJ2),
        Branch("L1", 2, 4, L=L1),
        Branch("L2", 3, 4, L=L2),
    ], ground=4)


KITE_CYCLES = {"kite": {"J1": 1, "J2": -1, "L1": 1, "L2": -1}, "cb": {"CS": 1, "J1": -1, "L1": -1}}  # eq:kite_G
KITE_Z = sym.Matrix([[1, 0, 0], [0, 1, 1], [0, -1, 1]])  # eq:kite_Z


def rhombus():
    """fig:rhombus with equal ground capacitances; J_k from node k to k+1."""
    CJ, CB, Cg, EJ, alpha = pos("C_J C_B C_g E_J alpha")
    return Circuit([
        Branch("J1", 1, 2, C=CJ, EJ=EJ),
        Branch("J2", 2, 3, C=CJ, EJ=EJ),
        Branch("J3", 3, 4, C=CJ, EJ=EJ),
        Branch("J4", 4, 1, C=alpha * CJ, EJ=alpha * EJ),
        Branch("CB", 1, 3, C=CB),
        *(Branch(f"Cg{i}", i, 5, C=Cg) for i in range(1, 5)),
    ], ground=5)


RHOMBUS_CYCLES = {  # eq:rh_G
    "ring": {"J1": 1, "J2": 1, "J3": 1, "J4": 1},
    "shunt": {"J1": -1, "J2": -1, "CB": 1},
    "g12": {"J1": -1, "Cg1": 1, "Cg2": -1},
    "g23": {"J2": -1, "Cg2": 1, "Cg3": -1},
    "g34": {"J3": -1, "Cg3": 1, "Cg4": -1},
}
RHOMBUS_Z = sym.Matrix([  # eq:rh_Z at equal C_g, given there as Z^{-1}
    [-1, 1, 0, 0], [0, -1, 1, 0], [0, 0, -1, 1], [sym.Rational(1, 4)] * 4,
]).inv()


def coupled():
    """fig:coupled_circuit_example: transmon on island 3 coupled to a fluxonium, three sources."""
    Ct, Cc, Cf, Cg, CV1, CV2, CV3, CV4, CJ1, CJ2, CJ3, Lf = pos(
        "C_t C_c C_f C_g C_V1 C_V2 C_V3 C_V4 C_J1 C_J2 C_J3 L_f")
    EJ1, EJ2, EJ3, V1, V2, V3 = sym.symbols("E_J1 E_J2 E_J3 V_1 V_2 V_3", positive=True)
    return Circuit([
        Branch("Ct", 1, 3, C=Ct),
        Branch("Cc", 1, 2, C=Cc),
        Branch("Cf", 2, 4, C=Cf),
        Branch("CV1", 1, 5, C=CV1),
        Branch("CV2", 1, 6, C=CV2),
        Branch("CV3", 2, 6, C=CV3),
        Branch("CV4", 2, 7, C=CV4),
        Branch("Cg", 3, 4, C=Cg),
        Branch("J1", 1, 3, C=CJ1, EJ=EJ1),
        Branch("J2", 1, 3, C=CJ2, EJ=EJ2),
        Branch("J3", 2, 4, C=CJ3, EJ=EJ3),
        Branch("Lf", 2, 4, L=Lf),
        Branch("V1", 4, 5, V=V1),
        Branch("V2", 4, 6, V=V2),
        Branch("V3", 4, 7, V=V3),
    ], ground=4)


def parallel(*branches):
    """Branches all from node 1 to ground 0."""
    return Circuit(list(branches), ground=0)


def coupled_zero_pis():
    """Two symmetric zero-pi qubits with a pure n_theta1 n_theta2 coupler.

    Circuit (a) of the zero-pi two-qubit gate paper: qubit nodes 1-4 and 5-8, each
    with C_0 to ground (node 0); nodes 1, 4, 5, 8 couple to node 10 and nodes
    2, 3, 6, 7 to node 9 through C_c.
    """
    CJ, C, L, C0, Cc, EJ = pos("C_J C L C_0 C_c E_J")
    branches = []
    for q, (a, b, c, d) in enumerate(((1, 2, 3, 4), (5, 6, 7, 8)), start=1):
        branches += [
            Branch(f"J{q}a", a, b, C=CJ, EJ=EJ), Branch(f"J{q}b", c, d, C=CJ, EJ=EJ),
            Branch(f"L{q}a", b, c, L=L), Branch(f"L{q}b", d, a, L=L),
            Branch(f"C{q}a", a, c, C=C), Branch(f"C{q}b", b, d, C=C),
            *(Branch(f"C0_{n}", n, 0, C=C0) for n in (a, b, c, d)),
        ]
    branches += [Branch(f"Cc_{n}", n, 10, C=Cc) for n in (1, 4, 5, 8)]
    branches += [Branch(f"Cc_{n}", n, 9, C=Cc) for n in (2, 3, 6, 7)]
    return Circuit(branches, ground=0)


def _coupled_zero_pi_Z():
    # Rows theta, phi, zeta, Sigma of the paper's transform, whose inverse is its transpose.
    M = sym.Matrix([[-1, 1, 1, -1], [-1, 1, -1, 1], [1, 1, -1, -1], [1, 1, 1, 1]]) / 2
    Z = sym.zeros(10, 10)
    Z[0:4, 0:4] = M.T
    Z[4:8, 4:8] = M.T
    Z[8:10, 8:10] = sym.Matrix([[1, 1], [1, -1]]) / 2  # varphi_+- = varphi_9 +- varphi_10
    return Z


COUPLED_ZERO_PI_Z = _coupled_zero_pi_Z()
COUPLED_ZERO_PI_TYPES = ("c", "e", "h", "fr", "c", "e", "h", "fr", "fr", "fr")
```

Create `tests/test_circuit.py`:

```python
import pytest
import sympy as sym

from circuit_quantization import Branch, Circuit, incidence
from circuits import coupled, harmonium

C, EJ, L = sym.symbols("C E_J L", positive=True)


@pytest.mark.parametrize("kwargs, message", [
    (dict(EJ=EJ), "needs C > 0"),
    (dict(C=C, L=L, EJ=EJ), "at most one"),
    (dict(C=C, V=1), "must have C = 0"),
    (dict(C=sym.Symbol("x")), "provably positive"),
])
def test_branch_validation(kwargs, message):
    with pytest.raises(ValueError, match=message):
        Branch("b", 1, 0, **kwargs)


def test_source_rules():
    with pytest.raises(ValueError, match="connect to ground"):
        Circuit([Branch("J", 1, 0, C=C, EJ=EJ), Branch("V", 1, 2, V=1)], ground=0)
    with pytest.raises(ValueError, match="Only capacitors"):
        Circuit([Branch("J", 1, 2, C=C, EJ=EJ), Branch("V", 0, 2, V=1)], ground=0)


def test_ordering_and_incidence_match_notes():
    circuit = harmonium()
    assert circuit.nodes == (1, 2, 3, 4, 5)
    assert circuit.branch_names == ("C", "J1", "J2", "J3", "L1", "L2", "L3")
    assert incidence(circuit) == sym.Matrix([  # eq:harm_Omega
        [1, 1, 1, 0, 0, 0, 0],
        [0, -1, 0, 0, 1, 0, 0],
        [0, 0, -1, 0, 0, 1, 0],
        [0, 0, 0, 1, -1, -1, 1],
        [-1, 0, 0, -1, 0, 0, -1],
    ])


def test_branches_sorted_by_type_keep_user_order():
    circuit = coupled()
    assert circuit.branch_names[:8] == ("Ct", "Cc", "Cf", "CV1", "CV2", "CV3", "CV4", "Cg")
    assert circuit.nodes == (1, 2, 3, 4, 5, 6, 7)
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `MPLCONFIGDIR=/tmp/cq-mpl conda run -n cqed python -m pytest tests/test_circuit.py -q -p no:cacheprovider`

Expected: FAIL during collection, e.g. `ImportError: cannot import name 'Branch'`.

- [ ] **Step 3: Implement**

Create `src/circuit_quantization/circuit.py`:

```python
"""Circuit description, incidence matrix, and cycle basis (Sec I)."""

from dataclasses import dataclass, field

import networkx as nx
import sympy as sym

# Branch order of eq:Omega_columns.
TYPES = ("C", "J", "L", "V")


@dataclass(frozen=True)
class Branch:
    """One oriented branch, tail -> head (Sec I.A, tab:elements).

    C [F] is an attribute any branch may carry.
    At most one of L [H], EJ [GHz], or V [V] makes it an inductor, junction, or source.
    V is the source node's potential relative to ground.
    """

    name: str
    tail: object
    head: object
    C: object = 0
    L: object = None
    EJ: object = None
    V: object = None

    def __post_init__(self):
        for attr in ("C", "L", "EJ", "V"):
            value = getattr(self, attr)
            if value is not None:
                object.__setattr__(self, attr, sym.sympify(value))
        if self.tail == self.head:
            raise ValueError(f"Branch {self.name!r} is a self-loop")
        if sum(x is not None for x in (self.L, self.EJ, self.V)) > 1:
            raise ValueError(f"Branch {self.name!r} may carry at most one of L, EJ, V")
        if not (self.C == 0 or self.C.is_positive):
            raise ValueError(f"Branch {self.name!r}: C must be 0 or provably positive")
        if self.type in ("C", "J") and self.C == 0:
            raise ValueError(f"Branch {self.name!r}: a {self.type} branch needs C > 0")
        if self.type == "V" and self.C != 0:
            raise ValueError(f"Source branch {self.name!r} must have C = 0")
        for attr in ("L", "EJ"):
            value = getattr(self, attr)
            if value is not None and value.is_positive is False:
                raise ValueError(f"Branch {self.name!r}: {attr} must be positive")

    @property
    def type(self):
        if self.EJ is not None:
            return "J"
        if self.L is not None:
            return "L"
        if self.V is not None:
            return "V"
        return "C"


def _ordered(nodes):
    nodes = list(dict.fromkeys(nodes))
    try:
        return sorted(nodes)
    except TypeError:
        return nodes


@dataclass(frozen=True)
class Circuit:
    """Branches plus a ground node, with nodes ordered int | gnd | src (Sec I.B)."""

    branches: tuple
    ground: object
    t: sym.Symbol = field(default=sym.Symbol("t", real=True))

    def __post_init__(self):
        branches = tuple(self.branches)
        names = [b.name for b in branches]
        if len(set(names)) != len(names):
            raise ValueError("Branch names must be unique")
        sources = []
        for b in branches:
            if b.type == "V":
                if self.ground not in (b.tail, b.head):
                    raise ValueError(f"Source {b.name!r} must connect to ground")
                sources.append(b.head if b.tail == self.ground else b.tail)
        if len(set(sources)) != len(sources):
            raise ValueError("Each source node carries exactly one source")
        for b in branches:
            if b.type in ("J", "L") and {b.tail, b.head} & set(sources):
                raise ValueError(f"Only capacitors may end on a source node: {b.name!r}")
        # Sort by type C | J | L | V, keeping user order within a type (eq:Omega_columns).
        branches = tuple(sorted(branches, key=lambda b: TYPES.index(b.type)))
        object.__setattr__(self, "branches", branches)
        object.__setattr__(self, "source_nodes", tuple(_ordered(sources)))
        others = [n for b in branches for n in (b.tail, b.head)]
        internal = [n for n in _ordered(others) if n != self.ground and n not in sources]
        if not internal:
            raise ValueError("Circuit has no internal nodes")
        object.__setattr__(self, "internal_nodes", tuple(internal))

    @property
    def nodes(self):
        return self.internal_nodes + (self.ground,) + self.source_nodes

    @property
    def branch_names(self):
        return tuple(b.name for b in self.branches)


def incidence(circuit):
    """Omega: +1 at each branch's tail and -1 at its head (Sec I.B)."""
    row = {node: i for i, node in enumerate(circuit.nodes)}
    Omega = sym.zeros(len(circuit.nodes), len(circuit.branches))
    for k, b in enumerate(circuit.branches):
        Omega[row[b.tail], k] = 1
        Omega[row[b.head], k] = -1
    return sym.ImmutableMatrix(Omega)
```

Replace `src/circuit_quantization/__init__.py`:

```python
"""Symbolic quantization of superconducting circuits (rebuild in progress)."""

from .circuit import Branch, Circuit, incidence
from .units import (
    farad_to_ghz,
    ghz_to_farad,
    ghz_to_henry,
    henry_to_ghz,
    phase_to_weber,
    weber_to_phase,
)
```

- [ ] **Step 4: Run the whole suite to verify it passes**

Run: `MPLCONFIGDIR=/tmp/cq-mpl conda run -n cqed python -m pytest tests -q -p no:cacheprovider`

Expected: PASS, 10 passed.

- [ ] **Step 5: Commit**

```bash
git add src tests
git commit -m "Rebuild: circuit model and incidence matrix (Sec I.A-B)" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```


---

### Task 3: Cycle basis: fundamental_cycles, Loops, rebase (Sec I.C, Appendix A)
**Files:**
- Modify: `src/circuit_quantization/circuit.py` (append)
- Modify: `src/circuit_quantization/__init__.py`
- Modify: `tests/test_circuit.py` (replace import block, append tests)

**Interfaces:**
- Consumes: `Circuit`, `incidence` from Task 2.
- Produces: `default_tree(circuit) -> list[str]`; `fundamental_cycles(circuit, tree=None) -> dict[str, dict[str, int]]` (keys are closure branch names, superconducting first); `Loops(circuit, cycles, fluxes=None)` with `.circuit`, `.names` (tuple), `.G` (ImmutableMatrix N_ell x N_b), `.Phi_ext` (ImmutableMatrix N_ell x 1, default symbols `Symbol('Phi_<name>', real=True)`); `Loops.rebase(G_new, names=None) -> Loops`.

- [ ] **Step 1: Write the failing tests**

In `tests/test_circuit.py`, replace everything above the first top-level `@` decorator or `def` (whichever comes first) with:

```python
import pytest
import sympy as sym

from circuit_quantization import Branch, Circuit, Loops, fundamental_cycles, incidence
from circuits import HARMONIUM_CYCLES, coupled, harmonium, zero_pi

C, EJ, L = sym.symbols("C E_J L", positive=True)
```

Append to `tests/test_circuit.py`:

```python
def test_default_cycles_of_coupled_example():
    # fig:coupled_circuit_example: 2 superconducting + 7 capacitively broken loops.
    circuit = coupled()
    cycles = fundamental_cycles(circuit)
    assert len(cycles) == 9
    loops = Loops(circuit, cycles)
    assert loops.G * incidence(circuit).T == sym.zeros(9, 7)  # eq:G_OmegaT
    assert set(loops.G) <= {-1, 0, 1}
    kinds = [b.type for b in circuit.branches]
    superconducting = [all(kinds[k] in "JL" for k in range(15) if loops.G[i, k]) for i in range(9)]
    assert superconducting == [True, True] + [False] * 7  # eq:G_blocks
    # One capacitively broken loop closes through a source branch.
    assert any(loops.G[i, circuit.branch_names.index("V1")] for i in range(9))


def test_cycles_from_explicit_tree():
    circuit = harmonium()
    cycles = fundamental_cycles(circuit, tree=["C", "J1", "J2", "J3"])
    assert set(cycles) == {"L1", "L2", "L3"}
    assert cycles["L3"] == {"J3": -1, "L3": 1}


def test_floating_circuit_cycle_count():
    # zero-pi: isolated ground, two components, N_ell = 6 - 3.
    assert len(fundamental_cycles(zero_pi())) == 3


def test_loops_flux_defaults_are_visible():
    circuit = harmonium()
    loops = Loops(circuit, HARMONIUM_CYCLES, {"cb": 0})
    Phi_k, Phi_f = sym.symbols("Phi_k Phi_f", real=True)
    assert loops.Phi_ext == sym.Matrix([Phi_k, Phi_f, 0])


def test_loops_rejects_incomplete_or_invalid_bases():
    circuit = harmonium()
    with pytest.raises(ValueError, match="1 cycle\\(s\\) missing"):
        Loops(circuit, {k: HARMONIUM_CYCLES[k] for k in ("k", "f")})
    with pytest.raises(ValueError, match="not a graph cycle"):
        Loops(circuit, {**HARMONIUM_CYCLES, "k": {"J1": 1}})
    with pytest.raises(ValueError, match="dependent"):
        Loops(circuit, {**HARMONIUM_CYCLES, "cb": HARMONIUM_CYCLES["k"]})


def test_rebase_maps_fluxes_and_rejects_non_unimodular():
    circuit = harmonium()
    loops = Loops(circuit, HARMONIUM_CYCLES, {"cb": 0})
    T = sym.Matrix([[1, 0, 0], [1, 1, 0], [0, 0, 1]])
    rebased = loops.rebase(T * loops.G)
    assert rebased.Phi_ext == T * loops.Phi_ext
    with pytest.raises(ValueError, match="det"):
        loops.rebase(2 * loops.G)
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `MPLCONFIGDIR=/tmp/cq-mpl conda run -n cqed python -m pytest tests/test_circuit.py -q -p no:cacheprovider`

Expected: FAIL during collection, e.g. `ImportError: cannot import name 'Loops'`.

- [ ] **Step 3: Implement**

Append to `src/circuit_quantization/circuit.py` (after two blank lines):

```python
def _graph(circuit):
    graph = nx.MultiGraph()
    graph.add_nodes_from(circuit.nodes)
    graph.add_edges_from((b.tail, b.head) for b in circuit.branches)
    return graph


def default_tree(circuit):
    """A spanning forest grown over S_ind (J, L) first, then S_cap (C, V).

    Growing S_ind first makes the closure rows inside S_ind span that
    subgraph's cycle space, so G comes out ordered as in eq:G_blocks.
    """
    forest = nx.utils.UnionFind(circuit.nodes)
    tree = []
    for b in sorted(circuit.branches, key=lambda b: b.type not in ("J", "L")):
        if forest[b.tail] != forest[b.head]:
            forest.union(b.tail, b.head)
            tree.append(b.name)
    return tree


def fundamental_cycles(circuit, tree=None):
    """Fundamental cycles of a spanning tree, as signed branch dicts (Appendix A).

    Returns {closure branch name: {branch name: +1 or -1}}, superconducting
    cycles first (eq:G_blocks).
    """
    names = circuit.branch_names
    tree = default_tree(circuit) if tree is None else list(tree)
    if set(tree) - set(names):
        raise ValueError(f"Unknown tree branches: {sorted(set(tree) - set(names))}")
    # Delete one root row per connected component, the ground in its own.
    roots = set()
    for component in nx.connected_components(_graph(circuit)):
        roots.add(circuit.ground if circuit.ground in component
                  else next(n for n in circuit.nodes if n in component))
    keep = [i for i, n in enumerate(circuit.nodes) if n not in roots]
    tree_cols = [names.index(name) for name in tree]
    closure_cols = sorted(
        (k for k in range(len(names)) if names[k] not in tree),
        key=lambda k: circuit.branches[k].type not in ("J", "L"),
    )
    Omega = incidence(circuit)
    Omega_t = Omega.extract(keep, tree_cols)
    Omega_c = Omega.extract(keep, closure_cols)
    if Omega_t.rows != Omega_t.cols or Omega_t.det() == 0:
        raise ValueError("tree is not a spanning forest of the circuit")
    G_t = -Omega_c.T * Omega_t.T.inv()  # eq:tree_B, Appendix A
    cycles = {}
    for row, k in enumerate(closure_cols):
        cycle = {names[j]: int(G_t[row, col]) for col, j in enumerate(tree_cols) if G_t[row, col] != 0}
        cycle[names[k]] = 1
        cycles[names[k]] = cycle
    return cycles


class Loops:
    """A complete cycle basis G and the external flux through each cycle (Sec I.C).

    Every cycle not given a value in ``fluxes`` gets a visible symbol Phi_<name> [Wb].
    Pass 0 explicitly for a cycle that encloses no flux.
    """

    def __init__(self, circuit, cycles, fluxes=None):
        fluxes = dict(fluxes or {})
        names = circuit.branch_names
        if set(fluxes) - set(cycles):
            raise ValueError(f"Fluxes for unknown cycles: {sorted(set(fluxes) - set(cycles))}")
        rows = []
        for loop, cycle in cycles.items():
            if set(cycle) - set(names):
                raise ValueError(f"Cycle {loop!r} uses unknown branches: {sorted(set(cycle) - set(names))}")
            rows.append([cycle.get(name, 0) for name in names])
        G = sym.Matrix(rows) if rows else sym.zeros(0, len(names))
        Omega = incidence(circuit)
        for loop, row in zip(cycles, rows):
            if any(sym.Matrix([row]) * Omega.T):
                raise ValueError(f"Cycle {loop!r} is not a graph cycle (eq:G_OmegaT)")
        if G.rank() < G.rows:
            raise ValueError("Cycles are linearly dependent")
        N_ell = len(names) - Omega.rank()
        if G.rows < N_ell:
            raise ValueError(f"{N_ell - G.rows} cycle(s) missing: a complete basis has {N_ell}")
        self.circuit = circuit
        self.names = tuple(cycles)
        self.G = sym.ImmutableMatrix(G)
        self.Phi_ext = sym.ImmutableMatrix(
            len(self.names), 1,
            [fluxes.get(loop, sym.Symbol(f"Phi_{loop}", real=True)) for loop in self.names],
        )

    def rebase(self, G_new, names=None):
        """Re-express the same fluxes in another cycle basis G_new = T G (eq:G_basis_change)."""
        G_new = sym.Matrix(G_new)
        G = sym.Matrix(self.G)
        T = G_new * G.T * (G * G.T).inv()
        if T * G != G_new:
            raise ValueError("G_new rows are not in the cycle space")
        if any(not x.is_integer for x in T) or abs(T.det()) != 1:
            raise ValueError("G_new = T G needs T integer with det T = +-1")
        names = names or tuple(f"loop_{i + 1}" for i in range(G_new.rows))
        branch_names = self.circuit.branch_names
        cycles = {
            loop: {branch_names[k]: int(G_new[i, k]) for k in range(G_new.cols) if G_new[i, k] != 0}
            for i, loop in enumerate(names)
        }
        return Loops(self.circuit, cycles, dict(zip(names, T * self.Phi_ext)))
```

Replace `src/circuit_quantization/__init__.py`:

```python
"""Symbolic quantization of superconducting circuits (rebuild in progress)."""

from .circuit import Branch, Circuit, Loops, fundamental_cycles, incidence
from .units import (
    farad_to_ghz,
    ghz_to_farad,
    ghz_to_henry,
    henry_to_ghz,
    phase_to_weber,
    weber_to_phase,
)
```

- [ ] **Step 4: Run the whole suite to verify it passes**

Run: `MPLCONFIGDIR=/tmp/cq-mpl conda run -n cqed python -m pytest tests -q -p no:cacheprovider`

Expected: PASS, 16 passed.

- [ ] **Step 5: Commit**

```bash
git add src tests
git commit -m "Rebuild: cycle basis from a spanning tree, Loops, rebase (Sec I.C)" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```


---

### Task 4: Node Lagrangian matrices (Sec II)
**Files:**
- Create: `src/circuit_quantization/lagrangian.py`
- Modify: `src/circuit_quantization/__init__.py`
- Test: `tests/test_lagrangian.py`

**Interfaces:**
- Consumes: `incidence` from Task 2.
- Produces: `lagrangian(circuit) -> Lagrangian` with fields `Omega, Omega_int, C_b, C_int, C_int_src, Lb_inv, Lstar_int, omega_J_int, omega_L_int, V` (ImmutableMatrix) and `J, L` (tuples of branch indices), `EJ` (tuple).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_lagrangian.py`:

```python
import sympy as sym

from circuit_quantization import lagrangian
from circuits import coupled, harmonium, pos


def test_harmonium_node_matrices():
    C, CJ1, CJ2, CJ3, L1, L2, L3 = pos("C C_J1 C_J2 C_J3 L_1 L_2 L_3")
    lag = lagrangian(harmonium())
    assert lag.C_b == sym.diag(C, CJ1, CJ2, CJ3, 0, 0, 0)  # eq:harm_Cb
    assert lag.C_int == sym.Matrix([  # eq:harm_node_matrices
        [C + CJ1 + CJ2, -CJ1, -CJ2, 0],
        [-CJ1, CJ1, 0, 0],
        [-CJ2, 0, CJ2, 0],
        [0, 0, 0, CJ3],
    ])
    assert lag.Lstar_int == sym.Matrix([
        [0, 0, 0, 0],
        [0, 1 / L1, 0, -1 / L1],
        [0, 0, 1 / L2, -1 / L2],
        [0, -1 / L1, -1 / L2, 1 / L1 + 1 / L2 + 1 / L3],
    ])
    assert lag.omega_J_int.T == sym.Matrix([[1, -1, 0, 0], [1, 0, -1, 0], [0, 0, 0, 1]])  # eq:harm_wJ
    assert lag.omega_L_int.T == sym.Matrix([[0, 1, 0, -1], [0, 0, 1, -1], [0, 0, 0, 1]])


def test_coupled_example_drive_block():
    Ct, Cc, Cf, Cg, CV1, CV2, CV3, CV4, CJ1, CJ2, CJ3, Lf = pos(
        "C_t C_c C_f C_g C_V1 C_V2 C_V3 C_V4 C_J1 C_J2 C_J3 L_f")
    lag = lagrangian(coupled())
    CT, C1, C2 = Ct + CJ1 + CJ2, Cc + CV1 + CV2, Cc + Cf + CJ3 + CV3 + CV4  # eq:fig1_shorthand
    assert lag.C_int == sym.Matrix([[C1 + CT, -Cc, -CT], [-Cc, C2, 0], [-CT, 0, Cg + CT]])  # eq:fig1_matrices
    assert lag.C_int_src == sym.Matrix([[CV1, CV2, 0], [0, CV3, CV4], [0, 0, 0]])  # eq:fig1_Cintsrc
    assert lag.V == sym.Matrix(sym.symbols("V_1 V_2 V_3", positive=True))
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `MPLCONFIGDIR=/tmp/cq-mpl conda run -n cqed python -m pytest tests/test_lagrangian.py -q -p no:cacheprovider`

Expected: FAIL during collection, e.g. `ImportError: cannot import name 'lagrangian'`.

- [ ] **Step 3: Implement**

Create `src/circuit_quantization/lagrangian.py`:

```python
"""Element and incidence matrices of the node Lagrangian (Sec II)."""

from dataclasses import dataclass

import sympy as sym

from .circuit import incidence


@dataclass(frozen=True)
class Lagrangian:
    Omega: sym.ImmutableMatrix  # rows int | gnd | src, columns C | J | L | V
    Omega_int: sym.ImmutableMatrix
    C_b: sym.ImmutableMatrix  # eq:Cb_diag
    C_int: sym.ImmutableMatrix  # eq:C_full_partition
    C_int_src: sym.ImmutableMatrix  # eq:C_full_partition, note the sign
    Lb_inv: sym.ImmutableMatrix  # L_b^{-1}, inductors only
    Lstar_int: sym.ImmutableMatrix  # eq:L_full_partition
    omega_J_int: sym.ImmutableMatrix
    omega_L_int: sym.ImmutableMatrix
    V: sym.ImmutableMatrix  # source-node voltages
    J: tuple  # branch indices of junctions
    L: tuple  # branch indices of inductors
    EJ: tuple


def lagrangian(circuit):
    branches = circuit.branches
    n = len(circuit.internal_nodes)
    Omega = sym.Matrix(incidence(circuit))
    Omega_int = Omega[:n, :]
    J = tuple(k for k, b in enumerate(branches) if b.type == "J")
    L = tuple(k for k, b in enumerate(branches) if b.type == "L")

    C_b = sym.diag(*(b.C for b in branches))  # eq:Cb_diag
    C_full = Omega * C_b * Omega.T  # eq:C_full_partition
    C_int = C_full[:n, :n]
    C_int_src = -C_full[:n, n + 1 :]

    omega_J_int = Omega_int[:, list(J)]
    omega_L_int = Omega_int[:, list(L)]
    Lb_inv = sym.diag(*(1 / branches[k].L for k in L)) if L else sym.zeros(0, 0)
    Lstar_int = omega_L_int * Lb_inv * omega_L_int.T  # eq:L_full_partition

    source = {(b.head if b.tail == circuit.ground else b.tail): b.V for b in branches if b.type == "V"}
    V = sym.Matrix([source[node] for node in circuit.source_nodes]) if source else sym.zeros(0, 1)

    freeze = sym.ImmutableMatrix
    return Lagrangian(
        Omega=freeze(Omega), Omega_int=freeze(Omega_int), C_b=freeze(C_b),
        C_int=freeze(C_int), C_int_src=freeze(C_int_src), Lb_inv=freeze(Lb_inv),
        Lstar_int=freeze(Lstar_int), omega_J_int=freeze(omega_J_int),
        omega_L_int=freeze(omega_L_int), V=freeze(V), J=J, L=L,
        EJ=tuple(branches[k].EJ for k in J),
    )
```

Replace `src/circuit_quantization/__init__.py`:

```python
"""Symbolic quantization of superconducting circuits (rebuild in progress)."""

from .circuit import Branch, Circuit, Loops, fundamental_cycles, incidence
from .lagrangian import lagrangian
from .units import (
    farad_to_ghz,
    ghz_to_farad,
    ghz_to_henry,
    henry_to_ghz,
    phase_to_weber,
    weber_to_phase,
)
```

- [ ] **Step 4: Run the whole suite to verify it passes**

Run: `MPLCONFIGDIR=/tmp/cq-mpl conda run -n cqed python -m pytest tests -q -p no:cacheprovider`

Expected: PASS, 18 passed.

- [ ] **Step 5: Commit**

```bash
git add src tests
git commit -m "Rebuild: node capacitance, susceptance, incidence blocks (Sec II)" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```


---

### Task 5: Mode classification and basis checks (Sec III.A, Sec IV.B.1)
**Files:**
- Create: `src/circuit_quantization/modes.py` (first part)
- Modify: `src/circuit_quantization/__init__.py`
- Test: `tests/test_modes.py`

**Interfaces:**
- Consumes: `Lagrangian` from Task 4.
- Produces: mode type strings `'c','e','h','fr','fz','cy'`, constants `DYNAMICAL`, `NONDYNAMICAL`, `TYPES`; `Coordinates` dataclass (`Z, types, C_tilde, C_tilde_inv, Lstar_tilde, omega_J_tilde, omega_L_tilde`); `subspaces(lag) -> dict[str, Matrix]` (keys c,e,h,fr,fz,cy,J,L,C,JL); `mode_basis(lag) -> (ImmutableMatrix, tuple)`; `check_basis(lag, Z_full, types) -> None` (raises ValueError).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_modes.py`:

```python
import pytest
import sympy as sym

from circuit_quantization import check_basis, lagrangian, mode_basis
from circuit_quantization.modes import subspaces
from circuits import HARMONIUM_Z, coupled, harmonium, kite


def zero(M):
    return sym.Matrix(M).applyfunc(sym.simplify).is_zero_matrix


def same_span(A, B):
    return sym.Matrix(A).rank() == sym.Matrix(B).rank() == sym.Matrix.hstack(A, B).rank()


def test_coupled_example_subspaces():
    sub = subspaces(lagrangian(coupled()))
    assert same_span(sub["J"], sym.Matrix([1, 0, 1]))  # eq:fig1_nullspaces
    assert same_span(sub["L"], sym.Matrix([[1, 0], [0, 0], [0, 1]]))
    assert sub["C"].cols == 0
    assert [sub[t].cols for t in ("c", "e", "h", "fr", "fz", "cy")] == [1, 1, 0, 1, 0, 0]


def test_automatic_kite_basis_separates_the_harmonic_direction():
    lag = lagrangian(kite())
    Z, types = mode_basis(lag)
    assert types == ("c", "e", "h")
    check_basis(lag, Z, types)
    omega = lag.omega_J_int.T * Z
    assert omega[:, 0] == sym.Matrix([1, 1])  # compact period Phi0 (eq:Scc_selection)
    assert omega[:, 2] == sym.zeros(2, 1)  # harmonic column sees no junction


@pytest.mark.parametrize("types, message", [
    (("e", "e", "e", "e"), "1 'c' modes"),
    (("h", "e", "e", "e"), "not a valid 'h'"),
])
def test_check_basis_rejects_wrong_types(types, message):
    with pytest.raises(ValueError, match=message):
        check_basis(lagrangian(harmonium()), HARMONIUM_Z, types)


def test_check_basis_rejects_non_primitive_compact_lattice():
    Z = HARMONIUM_Z.copy()
    Z[:, 0] = Z[:, 0] / 2
    with pytest.raises(ValueError, match="integer|primitive"):
        check_basis(lagrangian(harmonium()), Z, ("c", "e", "e", "e"))
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `MPLCONFIGDIR=/tmp/cq-mpl conda run -n cqed python -m pytest tests/test_modes.py -q -p no:cacheprovider`

Expected: FAIL during collection, e.g. `ImportError: cannot import name 'check_basis'`.

- [ ] **Step 3: Implement**

Create `src/circuit_quantization/modes.py`:

```python
"""Mode classification, decoupling, and residual freedom (Sec III).

Mode types are the superscripts of eq:Z_mode_blocks:
"c" compact, "e" extended, "h" harmonic (dynamical) and
"fr" free, "fz" frozen, "cy" cyclic (non-dynamical).
"""

from dataclasses import dataclass
from functools import reduce
from itertools import combinations
from math import gcd

import sympy as sym

DYNAMICAL = ("c", "e", "h")
NONDYNAMICAL = ("fr", "fz", "cy")
TYPES = DYNAMICAL + NONDYNAMICAL


@dataclass(frozen=True)
class Coordinates:
    Z: sym.ImmutableMatrix  # retained block, Phi_int = Z Phi_tilde
    types: tuple
    C_tilde: sym.ImmutableMatrix  # eq:Zd_schur
    C_tilde_inv: sym.ImmutableMatrix
    Lstar_tilde: sym.ImmutableMatrix
    omega_J_tilde: sym.ImmutableMatrix  # Z^T omega_J_int
    omega_L_tilde: sym.ImmutableMatrix  # Z^T omega_L_int


# --- subspaces, each a matrix of basis columns ------------------------------


def _primitive(v):
    """Scale a rational vector to coprime integers, for readability."""
    denominators = [sym.fraction(sym.nsimplify(x))[1] for x in v]
    v = v * reduce(sym.ilcm, denominators, 1)
    divisor = reduce(gcd, (int(x) for x in v), 0)
    return v / divisor if divisor else v


def _columns(vectors, n):
    return sym.Matrix.hstack(*vectors) if vectors else sym.zeros(n, 0)


def _null(A, n):
    return _columns([_primitive(v) for v in A.nullspace()], n)


def _perp(S):
    return _null(S.T, S.rows)


def _sum(A, B):
    return _columns(sym.Matrix.hstack(A, B).columnspace(), A.rows)


def _cap(A, B):
    return _perp(_sum(_perp(A), _perp(B)))


def _contains(S, v):
    return sym.Matrix.hstack(S, v).rank() == S.rank()


def subspaces(lag):
    """Column bases of the six mode subspaces of tab:var_basis.

    x^T C_int x = sum_k C_k (Omega_k^T x)^2 with C_k > 0, so N(C_int) is the
    nullspace of the capacitive columns of Omega_int, and likewise for L*_int.
    All three nullspaces therefore come from integer incidence matrices.
    """
    n = lag.Omega_int.rows
    capacitive = [k for k in range(lag.C_b.rows) if lag.C_b[k, k] != 0]
    J = _null(lag.omega_J_int.T, n)
    L = _null(lag.omega_L_int.T, n)
    C = _null(lag.Omega_int[:, capacitive].T, n)
    JL, CL = _cap(J, L), _cap(C, L)
    return {
        "c": _cap(L, _perp(JL)),
        "e": _perp(_sum(J, L)),
        "h": _cap(J, _perp(_sum(C, JL))),
        "fr": _cap(JL, _perp(CL)),
        "fz": _cap(C, _perp(CL)),
        "cy": CL,
        "J": J, "L": L, "C": C, "JL": JL,
    }


def _normalize_compact(Z_c, omega_J_int):
    """Give each compact coordinate period Phi0: S_cc = (omega_{J,K}^T)^{-1}, eq:Scc_selection."""
    if Z_c.cols == 0:
        return Z_c
    W = omega_J_int.T * Z_c
    K = list(W.T.rref()[1])  # independent rows of omega_{J,c}^T
    return Z_c * W.extract(K, list(range(W.cols))).inv()


def mode_basis(lag):
    """Square Z = (Z^c Z^e Z^h Z^fr Z^fz Z^cy) of eq:Z_mode_blocks and its column types."""
    sub = subspaces(lag)
    sub["c"] = _normalize_compact(sub["c"], lag.omega_J_int)
    Z = sym.Matrix.hstack(*(sub[t] for t in TYPES))
    types = tuple(t for t in TYPES for _ in range(sub[t].cols))
    return sym.ImmutableMatrix(Z), types


# --- validation ------------------------------------------------------------


def _is_integer(x):
    return sym.nsimplify(sym.simplify(x)).is_integer


def check_basis(lag, Z_full, types):
    """Check a square basis against tab:var_basis and eq:compact_period_lattice."""
    Z = sym.Matrix(Z_full)
    n = lag.Omega_int.rows
    types = tuple(types)
    if Z.shape != (n, n) or len(types) != n:
        raise ValueError(f"Z_full must be {n}x{n} with {n} types")
    if set(types) - set(TYPES):
        raise ValueError(f"Unknown mode types: {sorted(set(types) - set(TYPES))}")
    if sym.simplify(Z.det()) == 0:
        raise ValueError("Z_full is singular")
    sub = subspaces(lag)
    required = {"c": "L", "h": "J", "fr": "JL", "fz": "C", "cy": "cy"}
    for j, t in enumerate(types):
        if t in required and not _contains(sub[required[t]], Z[:, j]):
            raise ValueError(f"Column {j} is not a valid {t!r} direction (tab:var_basis)")
    for t in ("c",) + NONDYNAMICAL:
        if types.count(t) != sub[t].cols:
            raise ValueError(f"Circuit has {sub[t].cols} {t!r} modes, Z_full declares {types.count(t)}")
    compact = [j for j, t in enumerate(types) if t == "c"]
    if compact:
        W = (lag.omega_J_int.T * Z[:, compact]).applyfunc(sym.simplify)  # omega_{J,c}^T
        if not all(_is_integer(x) for x in W):
            raise ValueError("omega_J,c^T must be integer (Sec IV.B.1)")
        minors = [sym.simplify(W.extract(list(rows), list(range(W.cols))).det()) for rows in combinations(range(W.rows), W.cols)]
        if reduce(gcd, (int(m) for m in minors), 0) != 1:
            raise ValueError("Compact periods are not primitive (eq:compact_period_lattice)")
```

Replace `src/circuit_quantization/__init__.py`:

```python
"""Symbolic quantization of superconducting circuits (rebuild in progress)."""

from .circuit import Branch, Circuit, Loops, fundamental_cycles, incidence
from .lagrangian import lagrangian
from .modes import check_basis, mode_basis
from .units import (
    farad_to_ghz,
    ghz_to_farad,
    ghz_to_henry,
    henry_to_ghz,
    phase_to_weber,
    weber_to_phase,
)
```

- [ ] **Step 4: Run the whole suite to verify it passes**

Run: `MPLCONFIGDIR=/tmp/cq-mpl conda run -n cqed python -m pytest tests -q -p no:cacheprovider`

Expected: PASS, 23 passed.

- [ ] **Step 5: Commit**

```bash
git add src tests
git commit -m "Rebuild: mode classification of tab:var_basis and basis checks" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```


---

### Task 6: Decoupling and truncation (Sec III.B)
**Files:**
- Modify: `src/circuit_quantization/modes.py` (append)
- Modify: `src/circuit_quantization/__init__.py`
- Modify: `tests/test_modes.py` (replace import block, append tests)

**Interfaces:**
- Consumes: `Coordinates`, `DYNAMICAL`, `NONDYNAMICAL`, `_columns` from Task 5.
- Produces: `coordinates(lag, Z, types) -> Coordinates`; `decouple(lag, Z_full, types) -> Coordinates` (retained block only, types restricted to c/e/h in column order).

- [ ] **Step 1: Write the failing tests**

In `tests/test_modes.py`, replace everything above the first top-level `@` decorator or `def` (whichever comes first) with:

```python
import pytest
import sympy as sym

from circuit_quantization import Branch, Circuit, check_basis, decouple, lagrangian, mode_basis
from circuit_quantization.modes import subspaces
from circuits import (
    COUPLED_ZERO_PI_TYPES, COUPLED_ZERO_PI_Z, HARMONIUM_Z, RHOMBUS_Z, ZERO_PI_Z,
    coupled, coupled_zero_pis, harmonium, kite, pos, rhombus, zero_pi,
)
```

Append to `tests/test_modes.py`:

```python
def test_coupled_example_given_Z_reproduces_schur_complement():
    Ct, Cc, Cf, Cg, CV1, CV2, CV3, CV4, CJ1, CJ2, CJ3 = pos(
        "C_t C_c C_f C_g C_V1 C_V2 C_V3 C_V4 C_J1 C_J2 C_J3")
    CT, C1, C2 = Ct + CJ1 + CJ2, Cc + CV1 + CV2, Cc + Cf + CJ3 + CV3 + CV4
    lag = lagrangian(coupled())
    Z = sym.Matrix([[1, 0, 1], [0, 1, 0], [0, 0, 1]])  # eq:fig1_Z
    check_basis(lag, Z, ("c", "e", "fr"))
    coords = decouple(lag, Z, ("c", "e", "fr"))
    expected = sym.Matrix([  # eq:fig1_schur
        [CT + C1 * Cg / (C1 + Cg), -Cc * Cg / (C1 + Cg)],
        [-Cc * Cg / (C1 + Cg), C2 - Cc**2 / (C1 + Cg)],
    ])
    assert zero(coords.C_tilde - expected)


def test_automatic_zero_pi_truncates_cyclic_mode():
    lag = lagrangian(zero_pi())
    Z, types = mode_basis(lag)
    assert sorted(types) == sorted(("c", "e", "h", "cy"))
    coords = decouple(lag, Z, types)
    assert coords.types == ("c", "e", "h")
    assert coords.Z.shape == (4, 3)


def test_zero_pi_given_Z_matches_notes():
    CJ, C, L = pos("C_J C L")
    lag = lagrangian(zero_pi())
    types = ("e", "c", "h", "cy")
    check_basis(lag, ZERO_PI_Z, types)
    coords = decouple(lag, ZERO_PI_Z, types)
    assert coords.C_tilde == sym.diag(2 * CJ, 2 * (C + CJ), 2 * C)  # eq:zp_transformed
    assert coords.Lstar_tilde == sym.diag(2 / L, 0, 2 / L)
    assert coords.omega_J_tilde.T == sym.Matrix([[-1, -1, 0], [-1, 1, 0]])


def test_harmonium_given_Z_promotes_harmonic_to_extended():
    lag = lagrangian(harmonium())
    check_basis(lag, HARMONIUM_Z, ("c", "e", "e", "e"))
    coords = decouple(lag, HARMONIUM_Z, ("c", "e", "e", "e"))
    assert coords.omega_J_tilde.T == sym.Matrix([[1, -1, -1, -1], [1, -1, 1, -1], [0, 1, 0, 0]])  # eq:harm_Ltilde
    assert coords.omega_L_tilde.T == sym.Matrix([[0, 0, 1, 1], [0, 0, -1, 1], [0, 1, 0, 0]])


def test_rhombus_given_Z_free_mode_already_decoupled():
    CJ, CB, Cg, alpha = pos("C_J C_B C_g alpha")
    lag = lagrangian(rhombus())
    types = ("c", "c", "c", "fr")
    check_basis(lag, RHOMBUS_Z, types)
    coords = decouple(lag, RHOMBUS_Z, types)
    assert coords.Z == RHOMBUS_Z[:, :3]  # M_C = 0 for the weighted centroid
    expected = sym.Matrix([  # eq:rh_transformed
        [CB + (1 + alpha) * CJ + 3 * Cg / 4, CB + alpha * CJ + Cg / 2, alpha * CJ + Cg / 4],
        [CB + alpha * CJ + Cg / 2, CB + (1 + alpha) * CJ + Cg, alpha * CJ + Cg / 2],
        [alpha * CJ + Cg / 4, alpha * CJ + Cg / 2, (1 + alpha) * CJ + 3 * Cg / 4],
    ])
    assert zero(coords.C_tilde - expected)
    assert coords.omega_J_tilde.T == sym.Matrix([[-1, 0, 0], [0, -1, 0], [0, 0, -1], [1, 1, 1]])


def test_mode_by_mode_decoupling_equals_block_formula():
    # Two floating transmons coupled by C_x: two free modes coupled to each other in C.
    Ca, Cb, Cx, C1, C2, C3, C4 = pos("C_a C_b C_x C_1 C_2 C_3 C_4")
    circuit = Circuit([
        Branch("Ja", 1, 2, C=Ca, EJ=1), Branch("Jb", 3, 4, C=Cb, EJ=1), Branch("Cx", 2, 3, C=Cx),
        Branch("C1", 1, 0, C=C1), Branch("C2", 2, 0, C=C2), Branch("C3", 3, 0, C=C3), Branch("C4", 4, 0, C=C4),
    ], ground=0)
    lag = lagrangian(circuit)
    Z, types = mode_basis(lag)
    assert types.count("fr") == 2
    coords = decouple(lag, Z, types)
    dyn = [j for j, t in enumerate(types) if t in ("c", "e", "h")]
    free = [j for j, t in enumerate(types) if t == "fr"]
    Z1, Z2 = Z[:, dyn], Z[:, free]
    M = -(Z2.T * lag.C_int * Z2).inv() * (Z2.T * lag.C_int * Z1)  # eq:decouple_final
    assert zero(coords.Z - (Z1 + Z2 * M))  # eq:ZY_blocks


def test_frozen_node_gives_series_inductance():
    CJ, La, Lb = pos("C_J L_a L_b")
    circuit = Circuit([Branch("J", 1, 0, C=CJ, EJ=1), Branch("La", 1, 2, L=La), Branch("Lb", 2, 0, L=Lb)], ground=0)
    lag = lagrangian(circuit)
    Z, types = mode_basis(lag)
    assert types == ("e", "fz")
    coords = decouple(lag, Z, types)
    assert zero(coords.Lstar_tilde - sym.Matrix([[1 / (La + Lb)]]))


def test_rhombus_decoupling_produces_weighted_centroid():
    # Appendix B.4: for unequal C_g the free-mode decoupling gives weights C_g_i / sum C_g.
    CJ, CB, EJ, alpha = pos("C_J C_B E_J alpha")
    Cg = pos("C_g1:5")
    circuit = Circuit([
        Branch("J1", 1, 2, C=CJ, EJ=EJ), Branch("J2", 2, 3, C=CJ, EJ=EJ), Branch("J3", 3, 4, C=CJ, EJ=EJ),
        Branch("J4", 4, 1, C=alpha * CJ, EJ=alpha * EJ), Branch("CB", 1, 3, C=CB),
        *(Branch(f"Cg{i}", i, 5, C=Cg[i - 1]) for i in range(1, 5)),
    ], ground=5)
    lag = lagrangian(circuit)
    phases = [[-1, 1, 0, 0], [0, -1, 1, 0], [0, 0, -1, 1]]
    weighted = sym.Matrix(phases + [[g / sum(Cg) for g in Cg]]).inv()  # eq:rh_Z
    plain = sym.Matrix(phases + [[sym.Rational(1, 4)] * 4]).inv()
    types = ("c", "c", "c", "fr")
    check_basis(lag, plain, types)
    assert zero(decouple(lag, plain, types).Z - weighted[:, :3])


def test_coupled_zero_pis_give_pure_theta_theta_coupling():
    # Zero-pi two-qubit gate paper, circuit (a): after removing Sigma_1, Sigma_2, varphi_+-,
    # the retained charging matrix couples only theta_1 and theta_2.
    CJ, C, C0, Cc = pos("C_J C C_0 C_c")
    lag = lagrangian(coupled_zero_pis())
    Z, types = mode_basis(lag)
    assert sorted(types) == sorted(COUPLED_ZERO_PI_TYPES)
    check_basis(lag, COUPLED_ZERO_PI_Z, COUPLED_ZERO_PI_TYPES)
    coords = decouple(lag, COUPLED_ZERO_PI_Z, COUPLED_ZERO_PI_TYPES)
    assert coords.types == ("c", "e", "h", "c", "e", "h")
    Ct = 2 * C + 2 * CJ + C0 + Cc  # tilde C_theta
    Cp = 2 * CJ + C0 + Cc  # tilde C_phi
    theta = sym.Matrix([[2 * Ct - Cc, Cc], [Cc, 2 * Ct - Cc]]) / (2 * Ct * (Ct - Cc))
    expected = sym.zeros(6, 6)
    expected[0, 0], expected[0, 3], expected[3, 0], expected[3, 3] = theta
    expected[1, 1] = expected[4, 4] = 1 / Cp
    expected[2, 2] = expected[5, 5] = 1 / (2 * C + C0 + Cc)
    assert zero(coords.C_tilde_inv - expected)
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `MPLCONFIGDIR=/tmp/cq-mpl conda run -n cqed python -m pytest tests/test_modes.py -q -p no:cacheprovider`

Expected: FAIL during collection, e.g. `ImportError: cannot import name 'decouple'`.

- [ ] **Step 3: Implement**

Append to `src/circuit_quantization/modes.py` (after two blank lines):

```python
# --- decoupling and truncation ----------------------------------------------


def _orthogonalize(vectors, form):
    """Gram-Schmidt in the bilinear form, so each step divides by a positive scalar."""
    result = []
    for v in vectors:
        for u in result:
            v = v - (u.T * form * v)[0] / (u.T * form * u)[0] * u
        result.append(v.applyfunc(sym.cancel))
    return result


def coordinates(lag, Z, types):
    """Transformed objects of the retained block Z (eq:Zd_schur)."""
    Z = sym.Matrix(Z)
    C_tilde = (Z.T * lag.C_int * Z).applyfunc(sym.cancel)
    if sym.cancel(C_tilde.det()) == 0:
        raise ValueError("Retained capacitance matrix is singular")
    freeze = sym.ImmutableMatrix
    return Coordinates(
        Z=freeze(Z), types=tuple(types), C_tilde=freeze(C_tilde),
        C_tilde_inv=freeze(C_tilde.inv().applyfunc(sym.cancel)),
        Lstar_tilde=freeze((Z.T * lag.Lstar_int * Z).applyfunc(sym.cancel)),
        omega_J_tilde=freeze(Z.T * lag.omega_J_int),
        omega_L_tilde=freeze(Z.T * lag.omega_L_int),
    )


def decouple(lag, Z_full, types):
    """Retained block Z_1 + Z_2 M of eq:ZY_blocks, then truncation (Sec III.B).

    M_C = -C~_22^{-1} C~_21 (eq:decouple_final) acts on the free block and
    M_L = -L~*_22^{-1} L~*_21 (eq:decouple_final_L) on the frozen block.
    Each block is first orthogonalized in its own form, which makes C~_22 and
    L~*_22 diagonal, so M is applied one column and one mode at a time.
    """
    Z = sym.Matrix(Z_full)
    columns = [Z[:, j] for j in range(Z.cols)]
    free = _orthogonalize([z for z, t in zip(columns, types) if t == "fr"], lag.C_int)
    frozen = _orthogonalize([z for z, t in zip(columns, types) if t == "fz"], lag.Lstar_int)
    retained, retained_types = [], []
    for z, t in zip(columns, types):
        if t not in DYNAMICAL:
            continue
        for v in free:
            z = z - (v.T * lag.C_int * z)[0] / (v.T * lag.C_int * v)[0] * v
        for w in frozen:
            z = z - (w.T * lag.Lstar_int * z)[0] / (w.T * lag.Lstar_int * w)[0] * w
        retained.append(z.applyfunc(sym.cancel))
        retained_types.append(t)
    if not retained:
        raise ValueError("Circuit has no dynamical modes")
    Z_1 = sym.Matrix.hstack(*retained)
    Z_2 = _columns([z for z, t in zip(columns, types) if t in NONDYNAMICAL], Z.rows)
    for name, form in (("capacitance", lag.C_int), ("susceptance", lag.Lstar_int)):
        if (Z_1.T * form * Z_2).applyfunc(sym.cancel) != sym.zeros(Z_1.cols, Z_2.cols):
            raise AssertionError(f"Nonzero {name} cross block after decoupling (eq:ZY_blocks)")
    return coordinates(lag, Z_1, retained_types)
```

Replace `src/circuit_quantization/__init__.py`:

```python
"""Symbolic quantization of superconducting circuits (rebuild in progress)."""

from .circuit import Branch, Circuit, Loops, fundamental_cycles, incidence
from .lagrangian import lagrangian
from .modes import check_basis, decouple, mode_basis
from .units import (
    farad_to_ghz,
    ghz_to_farad,
    ghz_to_henry,
    henry_to_ghz,
    phase_to_weber,
    weber_to_phase,
)
```

- [ ] **Step 4: Run the whole suite to verify it passes**

Run: `MPLCONFIGDIR=/tmp/cq-mpl conda run -n cqed python -m pytest tests -q -p no:cacheprovider`

Expected: PASS, 32 passed.

- [ ] **Step 5: Commit**

```bash
git add src tests
git commit -m "Rebuild: mode-by-mode decoupling and truncation (eq:ZY_blocks)" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```


---

### Task 7: Residual freedom S_1 and transform (Sec III.C)
**Files:**
- Modify: `src/circuit_quantization/modes.py` (append)
- Modify: `src/circuit_quantization/__init__.py`
- Modify: `tests/test_modes.py` (replace import block, append tests)

**Interfaces:**
- Consumes: `Coordinates`, `_is_integer` from Task 5; `decouple` from Task 6.
- Produces: `residual_freedom(coords, keep_harmonic=True) -> Matrix` with symbols `s_<i>_<j>`; `transform(coords, S1) -> Coordinates` (raises ValueError on disallowed mixing or non-unimodular S_cc).

- [ ] **Step 1: Write the failing tests**

In `tests/test_modes.py`, replace everything above the first top-level `@` decorator or `def` (whichever comes first) with:

```python
import pytest
import sympy as sym

from circuit_quantization import (
    Branch, Circuit, check_basis, decouple, lagrangian, mode_basis, residual_freedom, transform,
)
from circuit_quantization.modes import subspaces
from circuits import (
    COUPLED_ZERO_PI_TYPES, COUPLED_ZERO_PI_Z, HARMONIUM_Z, RHOMBUS_Z, ZERO_PI_Z,
    coupled, coupled_zero_pis, harmonium, kite, pos, rhombus, zero_pi,
)
```

Append to `tests/test_modes.py`:

```python
def test_residual_freedom_patterns():
    lag = lagrangian(kite())
    Z, types = mode_basis(lag)
    coords = decouple(lag, Z, types)  # types (c, e, h)
    a = residual_freedom(coords)
    b = residual_freedom(coords, keep_harmonic=False)
    nonzero = lambda S: [[int(x != 0) for x in S.row(i)] for i in range(3)]  # noqa: E731
    assert nonzero(a) == [[1, 1, 0], [0, 1, 0], [0, 1, 1]]  # S_{1,a}, eq:S1_structure
    assert nonzero(b) == [[1, 1, 0], [0, 1, 1], [0, 1, 1]]  # S_{1,b}


def test_transform_is_a_congruence_and_updates_types():
    lag = lagrangian(kite())
    Z, types = mode_basis(lag)
    coords = decouple(lag, Z, types)
    S = sym.Matrix([[1, 1, 0], [0, 1, 1], [0, 0, 1]])
    new = transform(coords, S)
    assert new.types == ("c", "e", "e")  # S_eh != 0 promotes harmonic to extended
    assert zero(new.C_tilde - S.T * coords.C_tilde * S)
    assert new.Z == coords.Z * S


@pytest.mark.parametrize("S, message", [
    (sym.Matrix([[1, 0, 0], [0, 1, 0], [1, 0, 1]]), "mixes 'h' into 'c'"),
    (sym.Matrix([[2, 0, 0], [0, 1, 0], [0, 0, 1]]), "GL"),
])
def test_transform_rejects_disallowed_mixing(S, message):
    lag = lagrangian(kite())
    coords = decouple(lag, *mode_basis(lag))
    with pytest.raises(ValueError, match=message):
        transform(coords, S)
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `MPLCONFIGDIR=/tmp/cq-mpl conda run -n cqed python -m pytest tests/test_modes.py -q -p no:cacheprovider`

Expected: FAIL during collection, e.g. `ImportError: cannot import name 'residual_freedom'`.

- [ ] **Step 3: Implement**

Append to `src/circuit_quantization/modes.py` (after two blank lines):

```python
# --- residual freedom -------------------------------------------------------

# (old type, new type) pairs a column may be mixed along, eq:S1_structure.
_S1a = {("c", "c"), ("c", "e"), ("e", "e"), ("h", "e"), ("h", "h")}
_S1b = _S1a | {("e", "h")}


def residual_freedom(coords, keep_harmonic=True):
    """Symbolic S_1 with free entries where eq:S1_structure allows them.

    keep_harmonic=True gives S_{1,a}; False gives S_{1,b}, which lets harmonic
    columns acquire extended components. S_cc must also lie in GL(N_c, Z)
    (eq:Scc_unimodular); transform() checks that once values are chosen.
    """
    allowed = _S1a if keep_harmonic else _S1b
    types = coords.types
    return sym.Matrix(
        len(types), len(types),
        lambda i, j: sym.Symbol(f"s_{i + 1}_{j + 1}") if (types[i], types[j]) in allowed else 0,
    )


def transform(coords, S1):
    """Apply Z -> Z S_1 to the retained coordinates (eq:S1_structure, eq:Scc_unimodular)."""
    S = sym.Matrix(S1)
    types = coords.types
    if S.shape != (len(types), len(types)) or sym.simplify(S.det()) == 0:
        raise ValueError("S1 must be square, invertible, and match the retained modes")
    for i in range(S.rows):
        for j in range(S.cols):
            if S[i, j] != 0 and (types[i], types[j]) not in _S1b:
                raise ValueError(f"S1[{i}, {j}] mixes {types[i]!r} into {types[j]!r} (eq:S1_structure)")
    compact = [j for j, t in enumerate(types) if t == "c"]
    S_cc = S.extract(compact, compact).applyfunc(sym.simplify)
    if compact and (not all(_is_integer(x) for x in S_cc) or abs(S_cc.det()) != 1):
        raise ValueError("S_cc must lie in GL(N_c, Z) (eq:Scc_unimodular)")
    new_types = tuple(
        "e" if t == "h" and any(S[i, j] != 0 for i, u in enumerate(types) if u == "e") else t
        for j, t in enumerate(types)
    )
    freeze = sym.ImmutableMatrix
    C_tilde = (S.T * coords.C_tilde * S).applyfunc(sym.cancel)
    return Coordinates(
        Z=freeze(coords.Z * S), types=new_types, C_tilde=freeze(C_tilde),
        C_tilde_inv=freeze(C_tilde.inv().applyfunc(sym.cancel)),
        Lstar_tilde=freeze((S.T * coords.Lstar_tilde * S).applyfunc(sym.cancel)),
        omega_J_tilde=freeze(S.T * coords.omega_J_tilde),
        omega_L_tilde=freeze(S.T * coords.omega_L_tilde),
    )
```

Replace `src/circuit_quantization/__init__.py`:

```python
"""Symbolic quantization of superconducting circuits (rebuild in progress)."""

from .circuit import Branch, Circuit, Loops, fundamental_cycles, incidence
from .lagrangian import lagrangian
from .modes import check_basis, decouple, mode_basis, residual_freedom, transform
from .units import (
    farad_to_ghz,
    ghz_to_farad,
    ghz_to_henry,
    henry_to_ghz,
    phase_to_weber,
    weber_to_phase,
)
```

- [ ] **Step 4: Run the whole suite to verify it passes**

Run: `MPLCONFIGDIR=/tmp/cq-mpl conda run -n cqed python -m pytest tests -q -p no:cacheprovider`

Expected: PASS, 36 passed.

- [ ] **Step 5: Commit**

```bash
git add src tests
git commit -m "Rebuild: residual freedom S_1 and validated transform (eq:S1_structure)" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```


---

### Task 8: Irrotational flux assignment (Sec V)
**Files:**
- Create: `src/circuit_quantization/flux.py`
- Modify: `src/circuit_quantization/__init__.py`
- Test: `tests/test_flux.py`

**Interfaces:**
- Consumes: `Loops` (Task 3), `lagrangian` (Task 4), `mode_basis` (Task 5), `decouple` (Task 6).
- Produces: `irrotational_B(lag, coords, loops) -> ImmutableMatrix` of shape N_b x N_ell, rows in `circuit.branch_names` order.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_flux.py`:

```python
import sympy as sym

from circuit_quantization import Branch, Loops, decouple, irrotational_B, lagrangian, mode_basis
from circuits import (
    HARMONIUM_CYCLES, HARMONIUM_Z, KITE_CYCLES, KITE_Z, RHOMBUS_CYCLES, RHOMBUS_Z,
    ZERO_PI_CYCLES, ZERO_PI_Z, harmonium, kite, parallel, pos, rhombus, zero_pi,
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
    B = column(circuit, Loops(circuit, {"sq": {"J1": 1, "J2": -1}}), "sq")
    assert sym.simplify(B["J1"] - C2 / (C1 + C2)) == 0
    assert sym.simplify(B["J2"] + C1 / (C1 + C2)) == 0


def test_fluxonium_divider_with_and_without_inductor_capacitance():
    CJ, CL, L = pos("C_J C_L L")
    for C_L, expected in ((0, {"J": 0, "L": -1}), (CL, {"J": CL / (CJ + CL), "L": -CJ / (CJ + CL)})):
        circuit = parallel(Branch("J", 1, 0, C=CJ, EJ=1), Branch("L", 1, 0, C=C_L, L=L))
        B = column(circuit, Loops(circuit, {"f": {"J": 1, "L": -1}}), "f")
        assert all(sym.simplify(B[k] - v) == 0 for k, v in expected.items())


def test_shared_inductor():
    C1, C2, L = pos("C_1 C_2 L")
    circuit = parallel(Branch("J1", 1, 0, C=C1, EJ=1), Branch("J2", 1, 0, C=C2, EJ=1), Branch("L", 1, 0, L=L))
    B = assign(circuit, Loops(circuit, {"1": {"J1": 1, "L": -1}, "2": {"J2": 1, "L": -1}}))
    expected = sym.Matrix([[C2, -C2], [-C1, C1], [-C1, -C2]]) / (C1 + C2)  # fig:shared_inductor
    assert (B - expected).applyfunc(sym.simplify).is_zero_matrix


def test_shunt_placement_is_set_by_the_capacitively_broken_flux():
    C1, C2, C, x = pos("C_1 C_2 C x")
    Phi = sym.Symbol("Phi", real=True)
    circuit = parallel(Branch("J1", 1, 0, C=C1, EJ=1), Branch("J2", 1, 0, C=C2, EJ=1), Branch("C", 1, 0, C=C))
    loops = Loops(circuit, {"sq": {"J1": 1, "J2": -1}, "cb": {"J1": 1, "C": -1}}, {"sq": Phi, "cb": x * Phi})
    offsets = assign(circuit, loops) * loops.Phi_ext
    S = C1 + C2 + C
    J1, J2 = circuit.branch_names.index("J1"), circuit.branch_names.index("J2")
    assert sym.simplify(offsets[J1] - (C * x + C2) / S * Phi) == 0  # eq:shunt_placement
    assert sym.simplify(offsets[J2] + (C * (1 - x) + C1) / S * Phi) == 0


def test_capacitive_coupler_loop():
    CJ1, CJ2, Cc = pos("C_J1 C_J2 C_c")
    circuit = parallel(Branch("J1", 1, 0, C=CJ1, EJ=1), Branch("J2", 2, 0, C=CJ2, EJ=1), Branch("Cc", 1, 2, C=Cc))
    B = column(circuit, Loops(circuit, {"c": {"J1": 1, "J2": -1, "Cc": -1}}), "c")
    D = Cc * (CJ1 + CJ2) + CJ1 * CJ2
    assert sym.simplify(B["J1"] - Cc * CJ2 / D) == 0  # eq:coupler_loop
    assert sym.simplify(B["J2"] + Cc * CJ1 / D) == 0


def test_harmonium_B():
    circuit = harmonium()
    B = assign(circuit, Loops(circuit, HARMONIUM_CYCLES, {"cb": 0}), HARMONIUM_Z, ("c", "e", "e", "e"))
    assert B[:4, :] == sym.zeros(4, 3)  # B_C = B_J = 0
    assert B[4:, :] == sym.Matrix([[0, 0, -1], [-1, 0, -1], [0, -1, 0]])  # eq:harm_B


def test_zero_pi_B():
    circuit = zero_pi()
    loops = Loops(circuit, ZERO_PI_CYCLES, {"t1": 0, "t2": 0})
    B = column(circuit, loops, "ring", ZERO_PI_Z, ("e", "c", "h", "cy"))
    assert (B["J1"], B["J2"], B["L1"], B["L2"]) == (0, 0, 0, 1)  # eq:zp_B


def test_kite_B():
    circuit = kite()
    B = column(circuit, Loops(circuit, KITE_CYCLES, {"cb": 0}), "kite", KITE_Z, ("c", "e", "e"))
    assert (B["CS"], B["J1"], B["J2"], B["L1"], B["L2"]) == (0, 0, 0, 0, -1)


def test_rhombus_B_isolated_limit():
    CJ, CB, Cg, alpha = pos("C_J C_B C_g alpha")
    circuit = rhombus()
    ring = column(circuit, Loops(circuit, RHOMBUS_CYCLES), "ring", RHOMBUS_Z, ("c", "c", "c", "fr"))
    B = {k: sym.simplify(v.subs(Cg, 0)) for k, v in ring.items()}
    D = 2 * CB * (1 + alpha) + CJ * (1 + 3 * alpha)
    expected = {"J1": alpha * CJ / D, "J2": alpha * CJ / D, "J3": alpha * (2 * CB + CJ) / D, "J4": (2 * CB + CJ) / D}
    assert all(sym.simplify(B[k] - v) == 0 for k, v in expected.items())  # eq:rh_B


def test_branch_offsets_invariant_under_rebase():
    circuit = harmonium()
    loops = Loops(circuit, HARMONIUM_CYCLES, {"cb": 0})
    T = sym.Matrix([[1, 0, 0], [1, 1, 0], [-1, 0, 1]])
    rebased = loops.rebase(T * loops.G)
    before = assign(circuit, loops, HARMONIUM_Z, ("c", "e", "e", "e")) * loops.Phi_ext
    after = assign(circuit, rebased, HARMONIUM_Z, ("c", "e", "e", "e")) * rebased.Phi_ext
    assert (before - after).applyfunc(sym.simplify).is_zero_matrix
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `MPLCONFIGDIR=/tmp/cq-mpl conda run -n cqed python -m pytest tests/test_flux.py -q -p no:cacheprovider`

Expected: FAIL during collection, e.g. `ImportError: cannot import name 'irrotational_B'`.

- [ ] **Step 3: Implement**

Create `src/circuit_quantization/flux.py`:

```python
"""Irrotational external-flux assignment (Sec V)."""

import sympy as sym


def irrotational_B(lag, coords, loops):
    """B = (I - Omega_int^T Z C~^{-1} Z^T Omega_int C_b) G^+, eq:B_irrot_direct_closed with W = 0."""
    G = sym.Matrix(loops.G)
    N_b = lag.Omega.cols
    if G.rows == 0:
        return sym.ImmutableMatrix(sym.zeros(N_b, 0))
    G_plus = G.T * (G * G.T).inv()  # eq:G_right_inverse
    Z, Omega_int, C_b = coords.Z, lag.Omega_int, lag.C_b
    P = sym.eye(N_b) - Omega_int.T * Z * coords.C_tilde_inv * Z.T * Omega_int * C_b
    B = (P * G_plus).applyfunc(sym.cancel)
    if (G * B - sym.eye(G.rows)).applyfunc(sym.cancel) != sym.zeros(G.rows, G.rows):
        raise AssertionError("B violates G B = I (eq:GB_constraint)")
    if (Z.T * Omega_int * C_b * B).applyfunc(sym.cancel) != sym.zeros(Z.cols, G.rows):
        raise AssertionError("B is not irrotational (eq:irrotational_expanded)")
    return sym.ImmutableMatrix(B)
```

Replace `src/circuit_quantization/__init__.py`:

```python
"""Symbolic quantization of superconducting circuits (rebuild in progress)."""

from .circuit import Branch, Circuit, Loops, fundamental_cycles, incidence
from .flux import irrotational_B
from .lagrangian import lagrangian
from .modes import check_basis, decouple, mode_basis, residual_freedom, transform
from .units import (
    farad_to_ghz,
    ghz_to_farad,
    ghz_to_henry,
    henry_to_ghz,
    phase_to_weber,
    weber_to_phase,
)
```

- [ ] **Step 4: Run the whole suite to verify it passes**

Run: `MPLCONFIGDIR=/tmp/cq-mpl conda run -n cqed python -m pytest tests -q -p no:cacheprovider`

Expected: PASS, 46 passed.

- [ ] **Step 5: Commit**

```bash
git add src tests
git commit -m "Rebuild: irrotational flux assignment (eq:B_irrot_direct_closed)" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```


---

### Task 9: Hamiltonian, quantize(), and the public API (Sec IV, Sec VI)
**Files:**
- Create: `src/circuit_quantization/hamiltonian.py`, `src/circuit_quantization/quantize.py`
- Modify: `src/circuit_quantization/__init__.py` (final version)
- Test: `tests/test_hamiltonian.py`, `tests/test_scqubits.py`

**Interfaces:**
- Consumes: everything above.
- Produces: `hamiltonian(lag, coords, loops, B, t) -> Hamiltonian` (fields `H, varphi, n, n_g, varphi_ext, E_C, E_L, E_Lb`); `quantize(circuit, loops, Z_full=None, types=None, coords=None) -> Result`; `Result` fields `circuit, loops, lag, coords, B, ham`, property `H`, method `branch_index(name)`, attribute passthrough to `lag`/`coords`/`ham`/`loops`, and `_repr_latex_()`. Compact modes carry sector symbols `n_g<j>` (j = 1-based retained mode index).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_hamiltonian.py`:

```python
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
```

Create `tests/test_scqubits.py`:

```python
import numpy as np
import pytest
import sympy as sym

from circuit_quantization import Branch, Circuit, lagrangian, mode_basis

scq = pytest.importorskip("scqubits")


def same_span(A, B):
    return sym.Matrix(A).rank() == sym.Matrix(B).rank() == sym.Matrix.hstack(A, B).rank()


def test_transmon_molecule_matches_scqubits():
    yaml = """branches:
- [JJ, 1, 2, 5, 1]
- [JJ, 3, 4, 5, 1]
- [C, 1, 3, 1]
- [C, 2, 4, 1]
"""
    symbolic = scq.SymbolicCircuit.from_yaml(yaml, from_file=False, initiate_sym_calc=False)
    sc_Z, sc_modes = symbolic.variable_transformation_matrix()
    sc_Z = sym.Matrix(np.asarray(sc_Z, dtype=int))

    circuit = Circuit([
        Branch("J1", 1, 2, C=1, EJ=5), Branch("J2", 3, 4, C=1, EJ=5),
        Branch("C13", 1, 3, C=1), Branch("C24", 2, 4, C=1),
    ], ground=0)
    Z, types = mode_basis(lagrangian(circuit))
    assert types.count("c") == len(sc_modes["periodic"]) == 2
    assert types.count("fr") == len(sc_modes["free"]) == 1
    assert types.count("cy") == len(sc_modes["sigma"]) == 1

    ours_nd = Z[:, [j for j, t in enumerate(types) if t in ("fr", "cy")]]
    theirs_nd = sc_Z[:, [i - 1 for i in sc_modes["free"] + sc_modes["sigma"]]]
    assert same_span(ours_nd, theirs_nd)
    ours_c = Z[:, [j for j, t in enumerate(types) if t == "c"]]
    theirs_c = sc_Z[:, [i - 1 for i in sc_modes["periodic"]]]
    # Compact columns may differ by discarded free and global directions.
    assert same_span(sym.Matrix.hstack(ours_c, ours_nd), sym.Matrix.hstack(theirs_c, theirs_nd))
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `MPLCONFIGDIR=/tmp/cq-mpl conda run -n cqed python -m pytest tests/test_hamiltonian.py tests/test_scqubits.py -q -p no:cacheprovider`

Expected: FAIL during collection, e.g. `ImportError: cannot import name 'quantize'`.

- [ ] **Step 3: Implement**

Create `src/circuit_quantization/hamiltonian.py`:

```python
"""Frequency-unit Hamiltonian (Sec IV, Sec VI)."""

from dataclasses import dataclass

import sympy as sym

from .units import GHz, e, h, phi0


@dataclass(frozen=True)
class Hamiltonian:
    H: sym.Expr  # H/h in GHz, eq:full_H_dimless
    varphi: sym.ImmutableMatrix
    n: sym.ImmutableMatrix
    n_g: sym.ImmutableMatrix
    varphi_ext: sym.ImmutableMatrix
    E_C: sym.ImmutableMatrix
    E_L: sym.ImmutableMatrix
    E_Lb: sym.ImmutableMatrix


def hamiltonian(lag, coords, loops, B, t):
    N_d = coords.Z.cols
    varphi = sym.Matrix(sym.symbols(f"varphi_1:{N_d + 1}", real=True))
    n = sym.Matrix(sym.symbols(f"n_1:{N_d + 1}", real=True))
    Z, B = sym.Matrix(coords.Z), sym.Matrix(B)
    Phi_ext = sym.Matrix(loops.Phi_ext)
    varphi_ext = Phi_ext / phi0  # eq:dimless_defs

    # eq:EC_EL_matrix, divided by 1e9 for GHz.
    E_C = (e**2 / (2 * h * GHz) * coords.C_tilde_inv).applyfunc(sym.cancel)
    E_L = (phi0**2 / (h * GHz) * coords.Lstar_tilde).applyfunc(sym.cancel)
    E_Lb = (phi0**2 / (h * GHz) * lag.Lb_inv).applyfunc(sym.cancel)

    # eq:Q_g_tilde_expanded; the flux term vanishes for irrotational B.
    Q_g = Z.T * (-lag.C_int_src * lag.V + lag.Omega_int * lag.C_b * B * Phi_ext.diff(t))
    # Each compact mode also carries a static sector label (Sec IV.B.1).
    sector = [sym.Symbol(f"n_g{j + 1}", real=True) if typ == "c" else 0 for j, typ in enumerate(coords.types)]
    n_g = (Q_g / (2 * e)).applyfunc(sym.cancel) + sym.Matrix(sector)

    B_L = B[list(lag.L), :]
    B_J = B[list(lag.J), :]
    q = n - n_g
    H = 4 * (q.T * E_C * q)[0]  # charging
    H += (varphi.T * E_L * varphi)[0] / 2  # inductive
    H += (varphi.T * coords.omega_L_tilde * E_Lb * B_L * varphi_ext)[0]  # inductive-flux cross term
    junction_phase = coords.omega_J_tilde.T * varphi + B_J * varphi_ext
    H -= sum(EJ * sym.cos(phase) for EJ, phase in zip(lag.EJ, junction_phase))  # Josephson

    freeze = sym.ImmutableMatrix
    return Hamiltonian(
        H=H, varphi=freeze(varphi), n=freeze(n), n_g=freeze(n_g),
        varphi_ext=freeze(varphi_ext), E_C=freeze(E_C), E_L=freeze(E_L), E_Lb=freeze(E_Lb),
    )
```

Create `src/circuit_quantization/quantize.py`:

```python
"""End-to-end pipeline: Sec I -> II -> III -> V -> IV."""

from collections import Counter
from dataclasses import dataclass

import sympy as sym

from .circuit import Circuit, Loops
from .flux import irrotational_B
from .hamiltonian import Hamiltonian, hamiltonian
from .lagrangian import Lagrangian, lagrangian
from .modes import Coordinates, check_basis, decouple, mode_basis


@dataclass(frozen=True)
class Result:
    circuit: Circuit
    loops: Loops
    lag: Lagrangian
    coords: Coordinates
    B: sym.ImmutableMatrix
    ham: Hamiltonian

    @property
    def H(self):
        return self.ham.H

    def branch_index(self, name):
        return self.circuit.branch_names.index(name)

    def __getattr__(self, name):
        # Expose every named matrix (res.Z, res.C_int, res.E_C, ...) directly.
        for part in ("lag", "coords", "ham", "loops"):
            obj = object.__getattribute__(self, part)
            if hasattr(obj, name):
                return getattr(obj, name)
        raise AttributeError(name)

    def _repr_latex_(self):
        counts = ", ".join(f"{t}: {k}" for t, k in Counter(self.coords.types).items())
        return f"$\\displaystyle H/h = {sym.latex(self.H)}$ GHz  (modes {counts})"


def quantize(circuit, loops, Z_full=None, types=None, coords=None):
    lag = lagrangian(circuit)
    if coords is None:
        if Z_full is None:
            Z_full, types = mode_basis(lag)
        check_basis(lag, Z_full, types)
        coords = decouple(lag, Z_full, types)
    B = irrotational_B(lag, coords, loops)
    ham = hamiltonian(lag, coords, loops, B, circuit.t)
    return Result(circuit=circuit, loops=loops, lag=lag, coords=coords, B=B, ham=ham)
```

Replace `src/circuit_quantization/__init__.py`:

```python
"""Symbolic quantization of superconducting circuits.

Companion code to the circuit quantization notes; each module follows one section:

    units.py        Sec VI   constants and converters
    circuit.py      Sec I    Branch, Circuit, Loops, fundamental_cycles
    lagrangian.py   Sec II   element and incidence matrices
    modes.py        Sec III  mode classification, decoupling, residual freedom
    flux.py         Sec V    irrotational flux assignment
    hamiltonian.py  Sec IV   frequency-unit Hamiltonian

Symbols in the notes and their names here:

    Omega, Omega_int    incidence matrix and its internal rows
    omega_J_int         junction columns of Omega_int
    omega_L_int         inductor columns of Omega_int
    C_b, Lb_inv         branch capacitances, inverse inductances
    C_int, C_int_src    internal and drive capacitance blocks
    Lstar_int           internal susceptance L*_int
    G, Phi_ext          loop matrix and external fluxes [Wb]
    Z, types            retained coordinate transformation and mode types
    C_tilde, Lstar_tilde, omega_J_tilde, omega_L_tilde   transformed objects
    B                   irrotational flux-assignment matrix
    E_C, E_L, E_Lb      frequency-unit matrices [GHz]
    n_g                 offset charges [Cooper pairs]
    H                   Hamiltonian H/h [GHz]
"""

from .circuit import Branch, Circuit, Loops, fundamental_cycles, incidence
from .flux import irrotational_B
from .hamiltonian import hamiltonian
from .lagrangian import lagrangian
from .modes import check_basis, decouple, mode_basis, residual_freedom, transform
from .quantize import Result, quantize
from .units import (
    farad_to_ghz,
    ghz_to_farad,
    ghz_to_henry,
    henry_to_ghz,
    phase_to_weber,
    weber_to_phase,
)
```

- [ ] **Step 4: Run the whole suite to verify it passes**

Run: `MPLCONFIGDIR=/tmp/cq-mpl conda run -n cqed python -m pytest tests -q -p no:cacheprovider`

Expected: PASS, 54 passed (2 scqubits deprecation warnings are expected).

- [ ] **Step 5: Commit**

```bash
git add src tests
git commit -m "Rebuild: frequency-unit Hamiltonian and quantize() (eq:full_H_dimless)" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```


---

### Task 10: Equation-reference checker, README, and mathematical contract

**Files:**
- Create: `tools/eqrefs.py`
- Modify: `README.md` (full replacement), `docs/mathematical-contract.md` (full replacement)

**Interfaces:**
- Consumes: the `eq:`/`tab:`/`fig:`/`apx:`/`sec:` labels cited in `src/` and `tests/`.
- Produces: `python tools/eqrefs.py <tex> [--numbers <aux>]`, exit 0 iff every cited label exists in the `.tex`.

- [ ] **Step 1: Create `tools/eqrefs.py`**

```python
"""Check that every notes label cited in the code exists in the .tex source.

Usage: python tools/eqrefs.py path/to/full_matrix_quantization.tex [--numbers path/to/.aux]
With --numbers, also print the printed equation number of each label.
"""

import argparse
import pathlib
import re
import sys

LABEL = re.compile(r"\b((?:eq|tab|fig|apx|sec):[A-Za-z0-9_]+)")


def cited_labels(root):
    cited = {}
    for path in sorted(root.rglob("*.py")):
        for line_no, line in enumerate(path.read_text().splitlines(), 1):
            for label in LABEL.findall(line):
                cited.setdefault(label, []).append(f"{path}:{line_no}")
    return cited


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tex", type=pathlib.Path)
    parser.add_argument("--numbers", type=pathlib.Path, help="the .aux file")
    args = parser.parse_args()
    root = pathlib.Path(__file__).resolve().parents[1]
    defined = set(re.findall(r"\\label\{([^}]*)\}", args.tex.read_text()))
    numbers = {}
    if args.numbers:
        for label, number in re.findall(r"\\newlabel\{([^}]*)\}\{\{([^}]*)\}", args.numbers.read_text()):
            numbers[label] = number
    cited = {}
    for folder in ("src", "tests"):
        cited.update(cited_labels(root / folder))
    missing = sorted(label for label in cited if label not in defined)
    for label in sorted(cited):
        mark = "MISSING" if label in missing else numbers.get(label, "")
        print(f"{label:40s} {mark}")
    if missing:
        print(f"\n{len(missing)} cited label(s) not in {args.tex}", file=sys.stderr)
        for label in missing:
            print(f"  {label}: {', '.join(cited[label])}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it against the notes**

Run: `conda run -n cqed python tools/eqrefs.py /Users/eweissler/overleaf/circuit_quantization/Notes/full_matrix_quantization.tex --numbers /Users/eweissler/overleaf/circuit_quantization/Notes/full_matrix_quantization.aux`

Expected: exit 0; one line per cited label with its printed number (e.g. `eq:harm_B  B12`) and no `MISSING`.

- [ ] **Step 3: Replace `README.md`**

````markdown
# circuit-quantization

Symbolic quantization of lumped superconducting circuits, written as a companion to the notes
*Matrix Formulation of Circuit Quantization: Lagrangian, External Flux, and Frequency Units*.
Each module follows one section of the notes, and comments cite the notes' equation labels.

| Module | Notes | Contents |
|---|---|---|
| `units.py` | Sec VI | exact SI constants, `ghz_to_farad`, `ghz_to_henry`, ... |
| `circuit.py` | Sec I, App A | `Branch`, `Circuit`, incidence matrix, `fundamental_cycles`, `Loops` |
| `lagrangian.py` | Sec II | `C_b`, `C_int`, `C_int_src`, `Lstar_int`, `omega_J_int`, ... |
| `modes.py` | Sec III | mode classification, decoupling and truncation, residual freedom `S_1` |
| `flux.py` | Sec V | irrotational flux assignment `B` |
| `hamiltonian.py` | Sec IV, VI | `H/h` in GHz |
| `quantize.py` | | runs the whole chain |

## Example

```python
import sympy as sym
import circuit_quantization as cq
from circuit_quantization import Branch, Circuit, Loops, quantize

EC, EL, EJ = sym.symbols("E_C E_L E_J", positive=True)
varphi_ext = sym.Symbol("varphi_ext", real=True)

fluxonium = Circuit([
    Branch("J", 1, 0, C=cq.ghz_to_farad(EC), EJ=EJ),
    Branch("L", 1, 0, L=cq.ghz_to_henry(EL)),
], ground=0)

cycles = cq.fundamental_cycles(fluxonium)          # {"L": {"J": -1, "L": 1}}
loops = Loops(fluxonium, cycles, {"L": cq.phase_to_weber(varphi_ext)})
res = quantize(fluxonium, loops)
res            # renders H/h in a notebook
res.E_C, res.B # every intermediate is available under its notes name
```

## Conventions

- **Units.** `C` in F, `L` in H, source `V` in V, loop fluxes in Wb, time in s. `EJ` and the Hamiltonian are `E/h` in GHz. Use `phase_to_weber` to pass a dimensionless external phase.
- **Ordering.** Nodes are ordered internal | ground | source, branches capacitor | junction | inductor | source. `res.circuit.branch_names` gives the branch order.
- **Loops.** `Loops` requires a complete cycle basis. Every cycle carries a visible flux `Phi_<name>` unless you set it explicitly, including capacitively broken cycles, whose flux changes the physics of a time-dependent drive (notes Sec V.E).
- **Coordinates.** `quantize` builds the mode-separated basis of the notes' Table II automatically. Pass `Z_full` and `types` to use your own square basis, or `coords=cq.transform(res.coords, S1)` with an `S1` from `cq.residual_freedom(res.coords)`.

## Development

```bash
conda run -n cqed python -m pip install -e ".[test]"
conda run -n cqed python -m pytest -q
conda run -n cqed python tools/eqrefs.py path/to/full_matrix_quantization.tex
```

The tests reproduce the worked examples of the notes' Appendix B and the examples in the body.
`examples/` has one notebook per Appendix B circuit.
````

- [ ] **Step 4: Check the README example runs**

Run the `## Example` block from the README in `conda run -n cqed python`, printing `cycles` and `res.H`.

Expected: `cycles == {'L': {'J': -1, 'L': 1}}` and `H = 4*E_C*n_1**2 - E_J*cos(varphi_1) + E_L*varphi_1**2/2 + E_L*varphi_1*varphi_ext`.

- [ ] **Step 5: Replace `docs/mathematical-contract.md` with the re-pinned notes hash**

```bash
HASH=$(shasum -a 256 "/Users/eweissler/overleaf/circuit_quantization/Notes/full_matrix_quantization.tex" | cut -d' ' -f1)
cat > docs/mathematical-contract.md <<EOF
# Mathematical contract

The source of truth is \`Notes/full_matrix_quantization.tex\`.
Code comments and tests cite its \`\\label\` names; \`tools/eqrefs.py\` checks that every cited label exists.

| Source | SHA-256 |
|---|---|
| \`Notes/full_matrix_quantization.tex\` | \`$HASH\` |

When the code and the notes disagree, the notes take precedence: fix the code, or raise the discrepancy against the notes.
EOF
```

- [ ] **Step 6: Commit**

```bash
git add tools README.md docs/mathematical-contract.md
git commit -m "Rebuild: README, equation-label checker, re-pinned contract" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 11: Example notebooks for Appendix B

**Files:**
- Create: `tools/make_examples.py`
- Create (generated): `examples/harmonium.ipynb`, `examples/zero_pi.ipynb`, `examples/kite.ipynb`, `examples/rhombus.ipynb`

**Interfaces:**
- Consumes: the public API from Task 9.
- Produces: four notebooks, committed without outputs; regenerate with `python tools/make_examples.py`.

- [ ] **Step 1: Create `tools/make_examples.py`**

```python
"""Write the Appendix B example notebooks (run once; the notebooks are committed)."""

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

SETUP = """import sympy as sym
import circuit_quantization as cq
from circuit_quantization import Branch, Circuit, Loops, quantize

sym.init_printing()
F, H, W = cq.ghz_to_farad, cq.ghz_to_henry, cq.phase_to_weber"""

EXAMPLES = {
    "harmonium": ("The harmonium (notes Appendix B.1, fig:harmonium)", """\
EC, EC1, EC2, EC3, EL1, EL2, EL3, EJ1, EJ2, EJ3 = sym.symbols(
    "E_C E_CJ1 E_CJ2 E_CJ3 E_L1 E_L2 E_L3 E_J1 E_J2 E_J3", positive=True)
vk, vf = sym.symbols("varphi_k varphi_f", real=True)

circuit = Circuit([
    Branch("C", 1, 5, C=F(EC)),
    Branch("J1", 1, 2, C=F(EC1), EJ=EJ1),
    Branch("J2", 1, 3, C=F(EC2), EJ=EJ2),
    Branch("J3", 4, 5, C=F(EC3), EJ=EJ3),
    Branch("L1", 2, 4, L=H(EL1)),
    Branch("L2", 3, 4, L=H(EL2)),
    Branch("L3", 4, 5, L=H(EL3)),
], ground=5)
cycles = {  # eq:harm_G
    "k": {"J1": 1, "J2": -1, "L1": 1, "L2": -1},
    "f": {"J3": 1, "L3": -1},
    "cb": {"C": 1, "J1": -1, "J3": -1, "L1": -1},
}
loops = Loops(circuit, cycles, {"k": W(vk), "f": W(vf), "cb": 0})
Z = sym.Matrix([[1, 0, 0, 0], [0, 1, 1, 1], [0, 1, -1, 1], [0, 1, 0, 0]])  # eq:harm_Z
res = quantize(circuit, loops, Z, ("c", "e", "e", "e"))
res""", ["res.E_C", "res.E_L", "res.B"]),
    "zero_pi": ("The symmetric 0-pi qubit (notes Appendix B.2)", """\
CJ, C, L, EJ = sym.symbols("C_J C L E_J", positive=True)
circuit = Circuit([
    Branch("J1", 1, 2, C=CJ, EJ=EJ),
    Branch("J2", 3, 4, C=CJ, EJ=EJ),
    Branch("L1", 2, 3, L=L),
    Branch("L2", 4, 1, L=L),
    Branch("C1", 1, 3, C=C),
    Branch("C2", 2, 4, C=C),
], ground=5)  # floating: ground carries no branch
cycles = {  # eq:zp_G
    "ring": {"J1": 1, "J2": 1, "L1": 1, "L2": 1},
    "t1": {"J1": 1, "L1": 1, "C1": -1},
    "t2": {"J2": 1, "L1": 1, "C2": -1},
}
loops = Loops(circuit, cycles, {"t1": 0, "t2": 0})
Z = sym.Matrix([[-1, -1, 1, 1], [1, 1, 1, 1], [-1, 1, -1, 1], [1, -1, -1, 1]]) / 2  # eq:zp_Z
res = quantize(circuit, loops, Z, ("e", "c", "h", "cy"))
res""", ["res.C_tilde", "res.Lstar_tilde", "res.B"]),
    "kite": ("The cos(2 phi) transmon (notes Appendix B.3)", """\
ECS, EC1, EC2, EL1, EL2, EJ1, EJ2 = sym.symbols("E_CS E_CJ1 E_CJ2 E_L1 E_L2 E_J1 E_J2", positive=True)
vext = sym.Symbol("varphi_ext", real=True)
circuit = Circuit([
    Branch("CS", 1, 4, C=F(ECS)),
    Branch("J1", 1, 2, C=F(EC1), EJ=EJ1),
    Branch("J2", 1, 3, C=F(EC2), EJ=EJ2),
    Branch("L1", 2, 4, L=H(EL1)),
    Branch("L2", 3, 4, L=H(EL2)),
], ground=4)
cycles = {"kite": {"J1": 1, "J2": -1, "L1": 1, "L2": -1}, "cb": {"CS": 1, "J1": -1, "L1": -1}}  # eq:kite_G
loops = Loops(circuit, cycles, {"kite": W(vext), "cb": 0})
Z = sym.Matrix([[1, 0, 0], [0, 1, 1], [0, -1, 1]])  # eq:kite_Z
res = quantize(circuit, loops, Z, ("c", "e", "e"))
res""", ["res.E_C", "cq.residual_freedom(res.coords)"]),
    "rhombus": ("The asymmetric rhombus (notes Appendix B.4)", """\
CJ, CB, Cg, EJ, alpha = sym.symbols("C_J C_B C_g E_J alpha", positive=True)
circuit = Circuit([
    Branch("J1", 1, 2, C=CJ, EJ=EJ),
    Branch("J2", 2, 3, C=CJ, EJ=EJ),
    Branch("J3", 3, 4, C=CJ, EJ=EJ),
    Branch("J4", 4, 1, C=alpha * CJ, EJ=alpha * EJ),
    Branch("CB", 1, 3, C=CB),
    *(Branch(f"Cg{i}", i, 5, C=Cg) for i in range(1, 5)),
], ground=5)
cycles = cq.fundamental_cycles(circuit)  # or eq:rh_G
loops = Loops(circuit, cycles, {name: 0 for name in cycles if name != "J4"})
res = quantize(circuit, loops)  # automatic basis: three compact modes, free mode removed
res""", ["res.coords.types", "res.B.subs(Cg, 0).applyfunc(sym.simplify)"]),
}

for name, (title, build, shows) in EXAMPLES.items():
    cells = [new_markdown_cell(f"# {title}\n\nEach step follows the notes; equation labels are given in comments."),
             new_code_cell(SETUP), new_code_cell(build)]
    cells += [new_code_cell(expr) for expr in shows]
    nbformat.write(new_notebook(cells=cells), f"examples/{name}.ipynb")
    print("wrote", name)
```

- [ ] **Step 2: Generate the notebooks**

Run: `conda run -n cqed python tools/make_examples.py`

Expected: `wrote harmonium`, `wrote zero_pi`, `wrote kite`, `wrote rhombus`.

- [ ] **Step 3: Execute each notebook to verify it runs**

```bash
for n in harmonium zero_pi kite rhombus; do
  MPLCONFIGDIR=/tmp/cq-mpl conda run -n cqed jupyter nbconvert --to notebook --execute \
    --output-dir /tmp/cq-notebooks examples/$n.ipynb > /dev/null 2>&1; echo "$n exit $?"
done
```

Expected: all four `exit 0`.

- [ ] **Step 4: Commit**

```bash
git add tools/make_examples.py examples/*.ipynb
git commit -m "Rebuild: Appendix B example notebooks" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 12: Remove the old implementation and scratch files

**Files:**
- Delete: `reference/`, `check_*.py`, `confirm_opus5.py`, `test_script.py`, `test_script2.py`, `sec1.tex`, `src/circuit_quantization.egg-info/`

**Interfaces:**
- Consumes: a passing suite from Tasks 1-11.
- Produces: a repository containing only the rebuilt package, tests, tools, examples, and docs. The deleted files remain in the Task 1 snapshot commit.

- [ ] **Step 1: Delete**

```bash
git rm -r -q reference check_*.py confirm_opus5.py test_script.py test_script2.py sec1.tex
rm -rf src/circuit_quantization.egg-info
conda run -n cqed python -m pip install -e . -q
```

- [ ] **Step 2: Verify nothing references the removed code**

Run: `grep -rn "reference/\|old_package\|choose_coordinates\|sircuitenum" --include=*.py --include=*.md --include=*.toml . | grep -v docs/superpowers`

Expected: no output.

- [ ] **Step 3: Run the suite and the label checker**

Run: `MPLCONFIGDIR=/tmp/cq-mpl conda run -n cqed python -m pytest  -q -p no:cacheprovider` then `conda run -n cqed python tools/eqrefs.py /Users/eweissler/overleaf/circuit_quantization/Notes/full_matrix_quantization.tex`

Expected: 54 passed (2 scqubits deprecation warnings); the label checker exits 0.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "Rebuild: remove the Codex implementation and scratch scripts" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```
