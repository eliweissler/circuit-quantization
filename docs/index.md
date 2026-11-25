# circuit-quantization

`circuit-quantization` builds symbolic Hamiltonians for lumped superconducting circuits.
Define capacitors, Josephson junctions, inductors, voltage sources, and loop fluxes.
The package constructs the circuit matrices, separates the dynamical modes, removes nondynamical coordinates, assigns external flux, and returns the Hamiltonian $H/h$ in GHz.

It handles grounded and floating circuits, compact and noncompact coordinates, automatic or user-supplied mode bases, analytic and numerical normal modes, JSON interchange, circuit drawing, and export to scqubits.
[KronFreeCircuits.jl](https://github.com/eliweissler/KronFreeCircuits.jl) reads the Hamiltonians it produces through the [interchange format](spec-format.md).

## Quick start

A symbolic fluxonium, with energies in GHz and the external flux as a reduced phase:

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
result.H
# 4*E_C*n_1**2 - E_J*cos(varphi_1) + E_L*varphi_1**2/2 - E_L*varphi_1*varphi_ext
```

```{toctree}
:maxdepth: 2
:caption: Contents

install
examples/index
spec-format
api
```
