# circuit-quantization

`circuit-quantization` builds symbolic Hamiltonians for lumped superconducting
circuits. Define capacitors, Josephson junctions, inductors, voltage sources,
and loop fluxes. The package then constructs the circuit matrices, separates
the dynamical modes, removes nondynamical coordinates, assigns external flux,
and returns the Hamiltonian in GHz.

The package supports grounded and floating circuits, compact and noncompact
coordinates, automatic or user-supplied mode bases, coordinate optimization,
JSON interchange, circuit drawing, and export to scqubits.

## Highlights

- Symbolic circuit construction with exact SymPy expressions.
- Automatic classification of compact, extended, harmonic, free, frozen, and
  cyclic modes.
- Consistent loop bases and irrotational external-flux assignment.
- Hamiltonians in the standard reduced-phase and Cooper-pair-number variables.
- Exact compact-charge shears, analytic normal modes, and numerical searches
  over valid residual coordinate transformations.
- Optional symbolic coordinate optimization with Singular and Z3.
- JSON serialization, Schemdraw diagrams, and scqubits export.
- Experimental product-state and hierarchical-basis tools for comparing
  coordinate choices in nonlinear circuits.

## Installation

Python 3.11 or newer is required. Install the package from the repository:

```bash
git clone https://github.com/eliweissler/circuit-quantization.git
cd circuit-quantization
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

The editable install is convenient for notebooks and local development. Use
`python -m pip install .` instead if you do not want an editable checkout.

Optional features are grouped as extras:

| Extra | Install command | Adds |
|---|---|---|
| Drawing | `python -m pip install -e ".[draw]"` | Schemdraw, Matplotlib, SVG/PDF/PNG circuit output |
| scqubits | `python -m pip install -e ".[scqubits]"` | Direct construction of `scqubits.Circuit` objects |
| Symbolic optimizer | `python -m pip install -e ".[optimize]"` | Z3 and optimizer support, plus the Singular executable |
| Tests | `python -m pip install -e ".[test]"` | pytest and the scqubits integration tests |

The supplied Conda environment installs the package, every Python extra, and
the Singular executable:

```bash
conda env create -f environment.yml
conda activate circuit_quantization
pytest -q
```

## Quick start

The following example constructs a symbolic fluxonium. Capacitance and
inductance energies are specified in GHz and converted to SI values internally.
The external flux is supplied as a dimensionless reduced phase.

```python
import sympy as sym
import circuit_quantization as cq

E_C, E_L, E_J = sym.symbols("E_C E_L E_J", positive=True)
varphi_ext = sym.Symbol("varphi_ext", real=True)

circuit = cq.Circuit([
    cq.Branch("J", 1, 0, C=cq.ghz_to_farad(E_C), EJ=E_J),
    cq.Branch("L", 1, 0, L=cq.ghz_to_henry(E_L)),
], ground=0)

loops = cq.Loops.from_branch_dict(
    circuit,
    {"f": ("J", "L")},
    {"f": cq.phase_to_weber(varphi_ext)},
)

result = cq.quantize(circuit, loops, simplify=True)

assert result.coords.types == ("e",)
print(result.H)
# 4*E_C*n_1**2 - E_J*cos(varphi_1)
# + E_L*varphi_1**2/2 - E_L*varphi_1*varphi_ext
```

`result.H` is the symbolic Hamiltonian `H/h` in GHz. A `Result` also exposes
the circuit, loop basis, Lagrangian matrices, retained coordinates, flux
assignment, and structured Hamiltonian coefficients:

```python
result.coords.Z             # retained coordinates embedded in node space
result.coords.types         # compact, extended, or harmonic labels
result.ham.E_C              # charging-energy matrix in GHz
result.ham.E_L              # inductive-energy matrix in GHz
result.ham.n_g              # offset charges in Cooper-pair units
result.B                    # branch-by-loop flux assignment
```

In a notebook, displaying `result` renders the Hamiltonian as LaTeX.

## Circuits and loops

Each `Branch` connects two node labels. Node `0` is conventional but not
special unless it is passed as `ground=0`. A circuit with `ground=None` is
floating.

| Parameter | Units |
|---|---|
| Capacitance `C` | farads |
| Inductance `L` | henries |
| Junction energy `EJ` | `E_J/h` in GHz |
| Voltage source `V` | volts |
| Loop flux | webers |
| Hamiltonian | `H/h` in GHz |

Branch names can generate their own symbols:

```python
capacitor = cq.Branch("C1", 1, 0)  # C_1
junction = cq.Branch("J1", 1, 0)   # C_J1 and E_J1
inductor = cq.Branch("L1", 1, 0)   # L_1
source = cq.Branch("V1", 2, 0)     # V_1

circuit = cq.Circuit([junction, inductor], ground=0)
print(circuit.variables)
```

`Loops.from_circuit` constructs a deterministic fundamental-cycle basis and
prints a short summary of each loop. Every loop receives a symbolic flux unless
one is supplied explicitly.

```python
loops = cq.Loops.from_circuit(circuit)

# Set fluxes on capacitively broken cycles to zero by default.
loops = cq.Loops.from_circuit(circuit, zero_capacitive=True)

# Specify a physical cycle basis and fluxes explicitly.
loops = cq.Loops.from_branch_dict(
    circuit,
    {"f": ("J1", "L1")},
    {"f": 0},
)
```

Tuple entries identify the branches in one simple loop. The incidence matrix
determines their relative signs. Pass an explicit signed dictionary when the
orientation must be fixed. `Loops.from_tree` accepts a custom spanning tree,
and `loops.rebase(...)` changes the cycle basis while transforming the fluxes
with it.

## Coordinates and mode transformations

The automatic basis retains three kinds of dynamical coordinates:

| Label | Meaning |
|---|---|
| `c` | Compact rotor with an integer charge lattice |
| `e` | Extended coordinate that participates in junction and inductive terms |
| `h` | Harmonic coordinate with no junction participation |

Free, frozen, and cyclic coordinates are separated and removed before the
Hamiltonian is built. Supply both `Z_full` and `types` to `quantize` when a
specific full basis is required.

Residual transformations act on retained coordinates as `Z -> Z S1`:

```python
allowed = result.residual_freedom(keep_harmonic=False)

S_charge = cq.compact_charge_decoupling(result.coords)
charge_decoupled = result.transform(S_charge)

S_modes, _, _ = cq.normal_mode_decoupling(result.lag, result.coords)
normal_modes = result.transform(S_modes)
```

`compact_charge_decoupling` removes compact-noncompact charging couplings
without changing the compact periods. `normal_mode_decoupling` then solves the
generalized eigenproblem on the noncompact block. Normal modes may mix extended
and harmonic directions, so the transformed harmonic axes can acquire junction
participation.

### Numerical coordinate search

`numerical_S1` minimizes a user-selected cost over an invertible
parameterization of the admissible transformation. The built-in normal-mode
cost reproduces the analytic generalized-eigenvector construction.

```python
pair = cq.Circuit([
    cq.Branch("J1", 1, 0, C=80e-15, EJ=4.0),
    cq.Branch("L1", 1, 0, L=300e-9),
    cq.Branch("J2", 2, 0, C=70e-15, EJ=4.5),
    cq.Branch("L2", 2, 0, L=250e-9),
    cq.Branch("Cc", 1, 2, C=10e-15),
], ground=0)

pair_loops = cq.Loops.from_branch_dict(
    pair,
    {
        "f1": ("J1", "L1"),
        "f2": ("J2", "L2"),
        "coupler": ("J1", "Cc", "J2"),
    },
    {"f1": 0, "f2": 0, "coupler": 0},
)

pair_result = cq.quantize(pair, pair_loops)
S1, search = cq.numerical_S1(
    pair_result.lag,
    pair_result.coords,
    values={},
    cost=cq.cost_normal_modes,
    relaxed=True,
)

assert search.cost < 1e-12
optimized = pair_result.transform(S1)
```

`cost_targeted` masks selected quadratic couplings. `cost_localization` supplies
the continuous junction-localization objective. The returned diagnostics contain
the optimizer result, transformed quadratic forms, and normalized couplings.

### Symbolic coordinate search

The optional `circuit_quantization.optimize` package searches for exact
transformations that remove selected junction, inductive, and charging
couplings. It uses Singular for polynomial systems and Z3 to choose small
integer junction coefficients.

```python
from circuit_quantization.optimize import choose_instance, choose_S1

families, h_class = choose_S1(result.lag, result.coords)
S1, weight = choose_instance(result.lag, result.coords, families[0])
optimized = result.transform(S1)
```

This search can be expensive. Install the `optimize` extra and make sure the
`Singular` executable is available on `PATH`.

## Save and load Hamiltonians

The `kronfree-spec` JSON format stores the symbolic Hamiltonian without rerunning
the circuit reduction:

```python
cq.save_spec(result, "fluxonium.json")
spec = cq.load_spec("fluxonium.json")

H = spec.hamiltonian()
spec.E_C
spec.E_L
spec.parameters
spec.modes

# Work with a JSON-ready dictionary instead of a file.
data = cq.to_spec(result)
assert cq.load_spec(data) == spec
```

`HamiltonianSpec` stores immutable matrices, cosine terms, mode labels, and real
parameter symbols. It reconstructs the Hamiltonian but not the full circuit or
quantization pipeline. See [the format specification](docs/spec-format.md) for
the schema, validation rules, and compatibility limits.

## Numerical product-state tools

The experimental product-state module evaluates a self-consistent product
ground state in compact charge bases and noncompact phase-grid bases. It can
also optimize the residual coordinates by minimizing the product-state energy.

```python
import numpy as np

ham = cq.NumericHamiltonian(
    types=("e", "e"),
    E_C=np.array([[1.0, 0.15], [0.15, 1.0]]),
    E_L=np.diag([0.5, 0.5]),
    flux_linear=np.zeros(2),
    n_g=np.zeros(2),
    cosines=(
        (4.0, np.array([1.0, 0.0]), np.pi),
        (4.0, np.array([0.0, 1.0]), np.pi),
    ),
)

state = cq.product_state(ham, max_points=120)
S1, info = cq.product_state_S1(
    ham,
    relaxed=True,
    regularization=1e-4,
    max_points=120,
)
```

`numeric_hamiltonian` evaluates a serialized `HamiltonianSpec` at numerical
parameter values. `exact_ground_state`, `infidelity`, and
`subspace_infidelity` provide small-system references. Product-state
optimization can be much slower than the quadratic coordinate searches.

## Circuit diagrams

Install the `draw` extra, then render a circuit inline or to a file:

```python
cq.draw_circuit_diagram(circuit)
cq.draw_circuit_diagram(circuit, "fluxonium.svg")  # SVG, PDF, or PNG
```

The default layout is planar with a spring-layout fallback. Pass `layout` as
`"spring"`, `"polygon"`, or an explicit `{node: (x, y)}` dictionary. Parallel
branches are drawn in separate lanes. Flux labels can be added with
`flux_labels={label: (x, y)}`.

## scqubits export

`scqubits_yaml` generates a scqubits circuit description without importing
scqubits. `to_scqubits` requires the `scqubits` extra and constructs the object
directly.

```python
parameters = {E_C: 0.3, E_J: 8.0, E_L: 0.5}

yaml = cq.scqubits_yaml(circuit, subs=parameters)
sc = cq.to_scqubits(
    circuit,
    subs=parameters,
    closure_branches=["L"],
    ext_basis="harmonic",
    cutoff=40,
)

sc.Φ1 = 0.5
sc.eigenvals(5)
```

The exporter converts SI capacitances and inductances to GHz. Voltage sources
are not supported. Loop fluxes and offset charges are not copied from a
`Result`; set them on the returned scqubits object using scqubits conventions.

## Important conventions

- Nodes are ordered as internal, ground, then source nodes. Each group uses
  natural sorting, so `2 < 10` and `"q2" < "q10"`.
- Branches are ordered as capacitors, junctions, inductors, then voltage
  sources. `circuit.branch_names` gives the final order.
- Compact phases are `2*pi` periodic and their conjugate charges are integers.
  A residual transformation must preserve this lattice.
- Every physical loop has a visible flux unless it is set explicitly to zero.
  Capacitively broken loops can matter for time-dependent drives.
- `coords.Z` embeds retained coordinates into node space. `coords.R` extracts
  retained coordinates and satisfies `R * Z = I`.
- `subs_constants` replaces the package's exact physical-constant symbols by
  their SI values. Circuit parameters remain symbolic.

## Package map

| Module | Purpose |
|---|---|
| `circuit.py` | Branch and circuit definitions, node and branch ordering |
| `loops.py` | Cycle bases, spanning trees, loop fluxes, and rebasing |
| `lagrangian.py` | Capacitance, inductive, incidence, and source matrices |
| `modes.py` | Mode classification, truncation, residual freedom, and analytic decoupling |
| `numerical.py` | Invertible numerical parameterizations and coordinate costs |
| `product_state.py` | Product-state solvers, exact references, and coordinate optimization |
| `flux.py` | Irrotational and user-controlled flux assignments |
| `hamiltonian.py` | Symbolic Hamiltonian construction and simplification |
| `quantize.py` | End-to-end quantization and the `Result` interface |
| `io.py` | JSON serialization |
| `draw.py` | Schemdraw circuit diagrams |
| `qpackage_interface.py` | scqubits YAML and object export |
| `optimize/` | Optional exact symbolic search with Singular and Z3 |

## Examples

The `examples/` directory contains complete workflows:

- `harmonium.ipynb`, `zero_pi.ipynb`, `kite.ipynb`, and `rhombus.ipynb` build
  representative superconducting circuits.
- `lclc_box.ipynb` demonstrates a circuit with nondynamical coordinates.
- `drawing.ipynb` shows diagram layouts and annotations.
- `product_state_demo.py` compares reference, normal-mode, and optimized
  product-state coordinates.
- `hierarchical_kite.py` compares truncated hierarchical bases for the KITE.

The numerical product-state scripts perform large eigensolves and can take
substantially longer than the notebooks.

## Development

Run the complete test suite from the repository root:

```bash
pytest -q
```

Run one area while developing:

```bash
pytest tests/test_modes.py -q
pytest tests/test_numerical.py -q
pytest tests/test_product_state.py -q
pytest tests/optimize -q
```

The optimizer tests are skipped automatically when their optional dependencies
are unavailable. The project is licensed under GPL-3.0-only.
