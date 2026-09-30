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
res            # renders H_GHz in a notebook
res.ham.E_C, res.B # intermediate matrices use their notes names
```

## Conventions

`Loops.from_circuit(circuit)` constructs a default `branch_dict` of
fundamental cycles from a deterministic spanning tree (J/L branches first,
then C/V, in circuit order). Each loop is named after its closure branch,
which is listed first. It prints one line per
loop with signed branches, classification, and assigned flux:

```python
loops = Loops.from_circuit(fluxonium)  # L: +L -J [superconducting]; flux = Phi_L
loops.branch_dict                    # dict[str, tuple[str, ...]]
loops = Loops(fluxonium, {"f": ("J", "L")}, {"f": 0})
```

Pass `zero_capacitive=True` to default every loop containing a capacitor or
voltage-source branch to zero flux. For custom fluxes, use
`Loops(circuit, branch_dict, fluxes=...)`. For a custom tree, use
`Loops(circuit, cq.fundamental_cycles(circuit, tree=tree), fluxes=...)`.
Junction capacitance and an inductor's parallel capacitance
do not break a superconducting loop. The default remains `False`, so setting
these physical fluxes to zero is an explicit choice.

Tuple entries identify one simple loop; they need not follow traversal order.
The incidence matrix determines relative signs, with the first listed branch
fixed to +1. Reordering that first branch can reverse the flux convention.
Explicit signed dictionaries remain supported for specifying orientations or
rebasing existing fluxes. An open path or a selection containing multiple
independent loops is rejected.

Omit branch values to generate symbols and infer the element from its name:

```python
capacitor = Branch("C1", 1, 0)  # positive C_1 [F]
junction = Branch("J1", 1, 0)  # positive C_J1 [F], E_J1 [GHz]
inductor = Branch("L1", 1, 0)  # positive L_1 [H], no parasitic C
source = Branch("V1", 2, 0)    # real V_1 [V], either sign allowed

circuit = Circuit([junction, inductor], ground=0)
CJ1, EJ1, L1 = circuit.variables
loops = Loops(circuit, {"k": {"J1": 1, "L1": -1}})
Phi_k, = loops.variables       # real Phi_k [Wb]
```

Use `Branch("J1", 1, 0, energy_symbols=True)` to generate `E_CJ1` and `E_J1`,
or the same option on `C1`/`L1` for `E_C1`/`E_L1`. Generated capacitance and
inductance energies are converted from GHz to SI internally. Explicit values
always keep their usual units and take precedence over name-based inference.
An omitted junction capacitance is generated even when `EJ` is supplied;
explicit `C=0` remains invalid for a junction.

`Branch.variables`, `Circuit.variables`, and `Loops.variables` are tuples of
unique free symbols, excluding physical constants and numerical values. Branch
parameters follow `C, L, EJ, V` order, circuits follow their sorted branch order,
and loop variables follow loop order. Shared symbols appear once; symbols within
each explicit expression are sorted. Rebased loops expose the original flux
symbols appearing in their transformed expressions. Set a loop's flux explicitly
to zero to omit its symbol.

- **Units.** `C` in F, `L` in H, source `V` in V, loop fluxes in Wb, time in s. `EJ` and the Hamiltonian are `E/h` in GHz. Use `phase_to_weber` to pass a dimensionless external phase.
- **Ordering.** Nodes are ordered internal | ground | source, each group sorted naturally by label (`2 < 10`, `"q2" < "q10"`; ints and strings mix). Branches are ordered capacitor | junction | inductor | source. `res.circuit.branch_names` gives the branch order.
- **Loops.** `Loops` requires a complete cycle basis whose coefficients are 0 or ±1, so every row is a physical loop. Every cycle carries a visible flux `Phi_<name>` unless you set it explicitly, including capacitively broken cycles, whose flux changes the physics of a time-dependent drive (notes Sec V.E).
- **Coordinates.** `quantize` builds the mode-separated basis of the notes' Table II automatically. Pass your own square `Z_full` together with `types`; the basis need not be decoupled yet. To explore residual mixing, `res.residual_freedom()` shows the allowed entries of S_1 in `res.coords.types` order, and `res.transform(S)` returns a new result for the same circuit and loops in coordinates `Z S`. Final mode types follow junction and inductor participation, allowing extended and harmonic axes to mix in either direction. For intermediate matrices alone, use `cq.transform(res.lag, res.coords, S)`.

To factor coefficients and combine trigonometric terms in the final Hamiltonian,
use `quantize(circuit, loops, Z_full, types, simplify=True)` (also works with the
automatic basis). For an existing expression, use `cq.simplify_hamiltonian(res.H)`.
This optional step keeps the additive terms separate and can take time for large
symbolic circuits.

## Symbolic constants

`units.e`, `units.h`, `units.hbar`, and `units.Phi0` are positive symbols;
`units.phi0 = units.Phi0 / (2*pi)`. Their exact SI values are available as
`e_num`, `h_num`, `hbar_num`, `Phi0_num`, and `phi0_num` in `units`.
The named symbols are independent in symbolic algebra; their physical identities
hold after numerical substitution. Hamiltonians use GHz by default. Converters
use `units.h_GHz`, the symbolic joules-per-GHz factor (`h_GHz_num = 10^9 * h_num`),
displayed simply as $h$, so coefficients contain no explicit powers of a billion.
The SI `units.h` is displayed as $h_{SI}$ to keep the two symbols distinct. `h` and `hbar`
retain their SI meanings; `subs_constants` also evaluates `h_GHz`.
Notebook results display $H_{\mathrm{GHz}}$, the numerical energy in GHz;
plain $h$ on the RHS denotes the joules-per-GHz conversion constant.

```python
cq.subs_constants(res.H)           # exact SI constants, circuit parameters stay symbolic
cq.subs_constants(res.ham.E_C)     # also works on matrices
cq.subs_constants(res.H).evalf(6)  # decimal coefficients
```

## Spanning trees and loop bases

```python
for tree in cq.spanning_trees(circuit):
    cycles = cq.fundamental_cycles(circuit, tree=tree)
    candidate = Loops(circuit, cycles)
    # Existing physical fluxes must be rebased along with G:
    rebased = loops.rebase(candidate.G, names=candidate.names)
```

The iterator yields branch-name tuples, keeping parallel branches distinct.
For disconnected circuits (including an isolated ground), it enumerates spanning
forests. It exhaustively checks branch subsets, so it is intended for small circuits.

A tree determines a fundamental cycle basis once orientations and ordering are
fixed, but different trees can produce the same cycle matrix. For example, a
triangle has three spanning trees and just one independent loop. Retaining the
closure-branch names identifies which tree generated the basis. Arbitrary cycle
bases need not be fundamental bases of any tree.

## Circuit diagrams

Install the optional drawing dependencies with `pip install -e ".[draw]"`.
`draw_circuit_diagram` adapts sircuitenum's Schemdraw renderer to `Circuit`:

```python
cq.draw_circuit_diagram(fluxonium)  # last expression displays inline in Jupyter
cq.draw_circuit_diagram(fluxonium, "fluxonium.svg")  # also PDF or PNG
```

Layouts are `"planar"` (default, spring fallback), `"spring"`, `"polygon"`, or an
explicit `{node: (x, y)}` dictionary. Explicit coordinates are multiplied by
`scale` (default 4). Parallel branches fan out into separate lanes; `spread`
controls their spacing. Use `label=False` to hide branch names or
`node_labels=False` to hide node labels. `label_loc={node: "left"}` changes a
node label's position. Automatic layouts may need adjustment for crowded graphs.

Add flux annotations with `flux_labels={r"$\odot\,\Phi_k$": (x, y)}`.
Positions use the same units as an explicit layout and are multiplied by `scale`.
Labels support Matplotlib mathtext; they annotate the display without changing
`Loops` or inferring flux orientation from the layout.

Ground receives a ground symbol, including an isolated reference ground in a
floating circuit. Voltage-source polarity is positive at the source node,
independent of branch orientation. A boxed junction includes its intrinsic
capacitance; an inductor's nonzero capacitance is displayed in parallel, without
adding a branch to the circuit model. The returned Schemdraw `Drawing` supports
inline notebook rendering and further customization; no LaTeX installation is needed.

## scqubits export

Install `pip install '.[scqubits]'`. The exporter adapts
`sircuitenum.qpackage_interface.to_SCqubits` to this package's `Circuit`:

```python
parameters = {EC: 0.3, EJ: 8, EL: 0.5}
yaml = cq.scqubits_yaml(fluxonium, subs=parameters)
sc = cq.to_scqubits(
    fluxonium, subs=parameters, closure_branches=["L"],
    ext_basis="harmonic", cutoff=40,
)
sc.Φ1 = 0.5  # flux in units of Phi0, in scqubits' closure-loop convention
sc.eigenvals(5)
```

`scqubits_yaml` needs no scqubits installation. Both functions convert SI
capacitances and inductances to GHz and evaluate physical constants. Supply
`subs` for remaining symbols, including when requesting `symbolic=True`
(a `scqubits.SymbolicCircuit`). Exported parameters are named `EJ_i`, `EC_i`,
and `EL_i`, using the zero-based index in `circuit.branches`; each starts with
the supplied numerical value. Junction capacitance stays in `JJ`; inductor
capacitance becomes a separate parallel `C`. Voltage sources are unsupported
and raise an error.

Ground maps to node 0. By default, `circuit.internal_nodes` maps to nodes
1, 2, …; pass `node_order` to change that order. A supplied
`transformation_matrix` must have its rows in that same order. This supports
the named node order and transformation used in the double-Harmonium notebooks.
`closure_branches` takes original branch names in external-flux order.
The exporter does not transfer `Loops`, flux values, or offset charges: assign
these on the returned object in scqubits' convention. Its default loop handling
excludes capacitive loops; `use_dynamic_flux_grouping=True` includes them.
See [scqubits configuration](https://scqubits.readthedocs.io/en/latest/api-doc/_autosummary/scqubits.core.circuit.Circuit.html)
for transformation and closure-branch conventions.

`cutoff` accepts a scalar or a dictionary keyed by scqubits' 1-based mode indices.
`trunc_num` optionally enables one subsystem per dynamical mode, with a scalar
or a list of truncation dimensions. Other keyword arguments go to the scqubits
constructor; further configuration is available on the returned object.

## Development

```bash
conda run -n cqed python -m pip install -e ".[test]"
conda run -n cqed python -m pytest -q
conda run -n cqed python tools/eqrefs.py path/to/full_matrix_quantization.tex
```

The tests reproduce the worked examples of the notes' Appendix B and the examples in the body.
`examples/` has one notebook per Appendix B circuit.

`tests/test_sircuitenum_regressions.py` ports quantization regression cases from
`sircuitenum/tests/test_quantize.py`, without requiring sircuitenum at runtime.
It covers mode counts, formerly singular bases, serial decoupling, and agreement
between automatic and explicit quantization. Topologies run with both distinct
and equal element values. An isolated reference ground preserves the original
floating circuits, and the original `sigma` mode maps to `cy`. Tests of
sircuitenum-specific Hamiltonian hashes and transformation enumeration are not
ported; their corresponding APIs are absent here.
