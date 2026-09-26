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
