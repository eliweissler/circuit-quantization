# circuit-quantization

Symbolic quantization of general superconducting circuits, including
non-dynamical-mode removal, classical voltage drives, and time-dependent
external flux in the projected irrotational gauge.

The public model is branch-centric. Capacitance and inductance inputs use SI
units; Josephson energies and the returned Hamiltonian use ordinary frequency
units in GHz (`E/h`, not `E/ħ`).

```python
import sympy as sp
import circuit_quantization as cq

EC, EL, EJ = sp.symbols("E_C E_L E_J", positive=True)
Phi = sp.symbols("Phi_ext", real=True)

fluxonium = cq.Circuit(
    branches=(
        cq.Branch(
            "J", 0, 1,
            capacitance=cq.ghz_to_farad(EC),
            josephson_energy=EJ,
        ),
        cq.Branch("L", 0, 1, inductance=cq.ghz_to_henry(EL)),
    ),
    ground=0,
    loop_fluxes=(cq.LoopFlux("loop", {"J": 1, "L": -1}, Phi),),
)

result = cq.quantize(fluxonium)
print(result.hamiltonian)  # H/h in GHz
```

`coordinate_strategy="structural"` returns the direct note-derived basis.
`coordinate_strategy="canonical"` runs the complete polynomial optimizer
vendored from `sircuitenum`: SageMath/Singular solves the compatible decoupling
systems and Z3 selects rational transformations with minimal integer nonlinear
incidence. The copied solver lives under
`circuit_quantization.optimization._vendor`; its branch-centric adapter is in
the parent submodule. Canonical optimization requires SageMath, Singular, and
the Python packages in the `optimization` extra. The existing `cqed` Conda
environment supplies the full stack.

An explicit node-flux transformation can instead be supplied with one mode
type per column:

```python
result = cq.quantize(
    circuit,
    transformation=Z,  # phi_node = Z @ phi_new
    mode_types=(cq.ModeType.COMPACT, cq.ModeType.EXTENDED),
)
```

Explicit transformations are validated for dimensions, invertibility,
capacitance rank, and consistent compact/harmonic declarations.

The optimized 1%-asymmetric non-degenerate noise-resilient qubit from Hays,
Kim, and Oliver (PRX Quantum 6, 040321; arXiv:2502.15459v1) is a runnable
example:

```bash
DOT_SAGE=/tmp/circuit-quantization-sage \
conda run -n cqed python examples/quantize_harmonium.py
```

## Units

- Capacitance: F
- Inductance: H
- Josephson energy: `E_J/h` in GHz
- Voltage-source value: source-node voltage relative to ground, in V
- External loop flux: Wb
- Symbolic time: s
- Hamiltonian: `H/h` in GHz

The canonical rescaling is

```text
varphi = Phi / phi_0,  n = Q / 2e,  phi_0 = hbar / 2e.
```

For time in seconds, evolution under the returned Hamiltonian satisfies
`i d|psi>/dt = 2*pi*1e9*H_GHz |psi>`.

## Development

The supported development environment is the existing `cqed` Conda
environment:

```bash
conda run -n cqed python -m pip install -e .
DOT_SAGE=/tmp/circuit-quantization-sage \
MPLCONFIGDIR=/tmp/circuit-quantization-mpl \
XDG_CACHE_HOME=/tmp/circuit-quantization-cache \
conda run -n cqed pytest -q -p no:cacheprovider
```

See [docs/mathematical-contract.md](docs/mathematical-contract.md) for the
equations and pinned source provenance.
