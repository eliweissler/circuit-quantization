# Hamiltonian interchange format: kronfree-spec version 1

This document defines the Hamiltonian interchange contract produced and consumed by `circuit-quantization`.
The format identifier is `"kronfree-spec"` for compatibility with KronFreeCircuits, but Python save/load does not require Julia.
Changes to the shared format require coordination with consuming packages; readers must reject unsupported versions.
The Python implementation is `src/circuit_quantization/io.py`, with executable validation and round-trip examples in `tests/test_io.py`.

## Hamiltonian and units

The file encodes the following symbolic Hamiltonian:

\[
H/h\;[\mathrm{GHz}] =
4(\mathbf n-\mathbf n_g)^{\mathsf T}E_C(\mathbf n-\mathbf n_g)
+\tfrac12\boldsymbol\varphi^{\mathsf T}E_L\boldsymbol\varphi
+\boldsymbol\varphi^{\mathsf T}\mathbf b
-\sum_k E_{J,k}\cos(\boldsymbol\omega_k^{\mathsf T}\boldsymbol\varphi+\delta_k)
+c.
\]

Here `flux_linear` is \(\mathbf b\), each cosine supplies `energy`, `omega`, and `offset`, and `constant` is \(c\).
Phases and cosine offsets are dimensionless radians, and charges are in units of \(2e\).
`E_C`, `E_L`, `flux_linear`, cosine energies, and `constant` are in GHz (ordinary frequency, not angular frequency).
The coefficient expressions may retain SI-valued parameters such as capacitances in farads, inductances in henries, voltages in volts, and fluxes in webers; `units` describes the resulting Hamiltonian, not every parameter.
Parameter units are the producer's responsibility and are not inferred by the loader.
For manageable symbolic coefficients, use `ghz_to_farad(E_C)`, `ghz_to_henry(E_L)`, and `phase_to_weber(varphi_ext)` when constructing the circuit, or pass an appropriate `subs` mapping during export.

## JSON fields

| Field | Type | Meaning |
| --- | --- | --- |
| `format` | string | Required, exactly `"kronfree-spec"`. |
| `version` | integer | Required, exactly `1`; booleans are not integers here. |
| `units` | string | `"GHz"`; always emitted, defaults to GHz when omitted on import. |
| `parameters` | array of strings | Required, unique declared real parameter names in binding order. |
| `modes` | nonempty array of objects | Required, unique nonempty `name` labels and `type` values `c`, `e`, or `h`. |
| `E_C`, `E_L` | arrays of arrays of expression strings | Required, symmetric, each exactly N by N, where N is the number of modes. |
| `flux_linear`, `n_g` | arrays of expression strings | Required, each of length N. |
| `cosines` | array of objects | Required, possibly empty; each object requires `energy`, `omega`, and `offset`. |
| `constant` | expression string | Optional additive energy, default `"0"`. |
| `charge_cosines` | array | Shared-format extension for phase slips; this Python version accepts only an absent or empty array. |
| `_generator` | object | Export provenance: `package`, installed `version` (or `"unknown"`), and ordered branch names in `branches`. |

For both energy matrices, opposite off-diagonal entries must represent the same function; nonsymmetric input is rejected rather than averaged.
Identical expression text or equivalent parsed syntax (ignoring whitespace, redundant parentheses, and arithmetic dots) passes directly, including identical `abs` or `sign` calls.
Otherwise, a pair containing `abs` or `sign` is rejected as non-analytic before SymPy simplification can erase those calls.
The remaining pairs are evaluated at three fixed pseudorandom real parameter points in `[0.5, 2]`, in sorted parameter-name order, and must agree to `1e-12` relative to the larger magnitude, with no absolute tolerance floor.
Domain errors and nonfinite results reject the pair; evaluation uses the original expressions so a simplification such as `sqrt(-x)^2 -> -x` cannot hide a domain error.
This follows Julia's validation criteria, but Python and Julia use independent random-number streams; finite sampling is a consistency check, not an exhaustive identity proof.

Mode types mean compact (`c`), extended (`e`), and harmonic (`h`).
All matrices, vectors, and cosine multipliers follow `modes` order, which export takes from `res.coords.types`.
Only retained dynamical modes are exported; free, frozen, and cyclic coordinates are not modes in this file.
Mode names are descriptive labels: `HamiltonianSpec.hamiltonian()` uses real coordinate symbols `varphi_1` through `varphi_N` and `n_1` through `n_N`, in mode order.
Pass `mode_names=("theta", "phi", "zeta")` to `to_spec` or `save_spec` to choose labels; otherwise they default to `varphi_1` through `varphi_N`.
The sequence must contain exactly N unique, nonempty strings.
Parameter names that collide with these coordinate symbols are rejected by Python.

Compact-sector parameters retain their quantization names `n_g<k>`, where k is the one-based index among **all retained modes**, not just compact modes.
For example, mode types `(c, e, c)` have sector labels `n_g1` and `n_g3`.
Changing `mode_names` does not rename these parameters or change the Hamiltonian.
To choose sector names, use the existing substitution interface, for example `subs={sym.Symbol("n_g3", real=True): sym.Symbol("n_g_theta2", real=True)}`.
This keeps existing quantization results compatible while allowing a consumer's naming convention.

Each cosine has an expression string `energy`, a length-N array `omega` of finite JSON numbers, and an expression string `offset`.
Its contribution is **minus** `energy * cos(omegaᵀ varphi + offset)`.
An `omega` entry on a compact mode must be exactly integral; Python does not round near-integers to satisfy this requirement.
Export rejects symbolic multipliers that remain after substitutions, and identifies the junction and mode in the error.
Integer multipliers remain integers; noninteger multipliers use JSON floating-point numbers and therefore may round.
Exact rational *coefficients* instead use expression strings such as `"1 // 3"` and remain exact.

Unknown keys are rejected unless their names begin with `_`, including inside mode and cosine objects.
Underscore metadata is ignored when constructing `HamiltonianSpec`; it does not restore circuit topology or the quantization pipeline.
Duplicate JSON object keys are rejected.
`operators`, `noise_channels`, and nonempty `charge_cosines` are unsupported and rejected rather than discarded.

## Scalar expression grammar

Expressions contain decimal integer or floating-point literals (optionally with `e`/`E` scientific notation), declared parameters, parentheses, the constants `pi`, `π`, and `ℯ`, and these operations:

```text
+ - * / ^ // sqrt exp cos sin tan cosh sinh tanh log abs sign conj real imag
```

Consumers must parse integer literals with arbitrary precision: SI-valued exports can contain exact integers far beyond 64 bits, such as `44173801000000000000000000000000`.

The dotted operators `.+`, `.-`, `.*`, `./`, and `.^` have the same scalar meanings.
Operator precedence follows [Julia's arithmetic precedence](https://docs.julialang.org/en/v1/manual/mathematical-operations/#Operator-Precedence-and-Associativity): powers, unary signs, rational division `//`, multiplication/division, and addition/subtraction.
Powers associate to the right; the remaining binary operations associate to the left.
`//` requires rational numeric operands; use `/` for symbolic division.
Named functions take one argument, except `log(base, value)`, which is also supported.
Implicit multiplication, dotted function calls, assignments, indexing, attributes, strings, comments, and other executable constructs are not part of this grammar.

Parameter names use a portable subset of Julia identifiers: a Unicode letter or underscore followed by letters, digits, or underscores, excluding all-underscore names, Julia keywords, constants, and function names.
Python requires canonical Julia spelling in both declarations and expression references: identifiers changed by NFC normalization or Julia's custom character map are rejected with the canonical spelling suggested.
For example, use `μ` rather than `µ`, `ε` rather than `ɛ`, and `ħ` rather than `ℏ`; this prevents declarations such as `["µ", "μ"]` from becoming duplicate names in Julia.
This is NFC plus [Julia's identifier character transformations](https://docs.julialang.org/en/v1/stdlib/Unicode/#Unicode.julia_chartransform), not general NFKC normalization: `ℯ` remains Euler's constant and mathematical-letter variants that Julia keeps distinct remain distinct here.
In particular `pi`, `π`, `ℯ`, `e`, `im`, `cis`, and `cispi` are reserved.
Every referenced parameter must be declared; unused declared parameters are allowed on import.
The exporter lists free parameters in first textual appearance order through `E_C` (row-major), `E_L`, `flux_linear`, `n_g`, and each cosine's energy and offset.
Different source symbols sharing a name but carrying different assumptions cannot be represented unambiguously and are rejected.

All parameters are reconstructed with `real=True`; additional assumptions such as `positive=True` are not stored.
Compare source and loaded expressions after aligning symbols by name, rather than comparing differently assumed SymPy symbols directly.
The parser rejects explicit complex constants or constructs, known nonreal intermediate expressions, and nonfinite values.
It does not prove real-valuedness or finiteness for every parameter assignment; callers must respect the domains of division, logarithms, roots, and fractional powers.

The exporter uses SymPy's Julia printer with enough decimal digits to round-trip each float at its original precision, preserving exact rationals without rationalizing floats.
One constant spelling is normalized: SymPy prints Euler's constant as `e`, but this format uses `ℯ`, which the Julia coefficient parser accepts.
The importer builds SymPy objects directly from the restricted grammar and never evaluates file contents as Python or Julia code.

## Complete example

This fluxonium uses a loop oriented from the inductor back through the junction, with external phase assigned to the inductor.
The linear term is positive in this convention.

```json
{
  "format": "kronfree-spec",
  "version": 1,
  "units": "GHz",
  "parameters": ["E_C", "E_L", "varphi_ext", "E_J"],
  "modes": [{"name": "varphi_1", "type": "e"}],
  "E_C": [["E_C"]],
  "E_L": [["E_L"]],
  "flux_linear": ["E_L * varphi_ext"],
  "n_g": ["0"],
  "cosines": [{"energy": "E_J", "omega": [1], "offset": "0"}],
  "constant": "0"
}
```

The source Hamiltonian omits the flux-only inductive energy, so export writes `constant: "0"` rather than adding \(E_L\varphi_{\rm ext}^2/2\).
Import preserves a nonzero `constant` when another producer supplies one.
Voltage-source contributions and compact-sector labels are retained in `n_g` with their original signs; `n_g` is not forcibly zeroed on noncompact modes.

## Python interface and Julia handoff

`to_spec(res, *, subs=None, simplify=True, mode_names=None)` produces the dictionary without modifying `res`.
Substitutions are applied before exact physical constants are evaluated; `simplify=True` combines denominators with `sympy.together` before `sympy.cancel`, avoiding costly expansion of nested substituted fractions.
`save_spec(res, path, **kwargs)` validates the dictionary before opening the destination, writes indented UTF-8 JSON, and returns that dictionary.
`load_spec(source)` accepts a dictionary, JSON text, or a path, and returns a frozen `HamiltonianSpec` with immutable matrices and tuples.
Use `pathlib.Path` for filenames that could be mistaken for JSON text, for example a filename beginning with a digit.
Calling `spec.hamiltonian()` assembles the saved Hamiltonian without repeating mode classification, basis selection, or flux assignment.
It does not restore `Result`, circuit objects, intermediate quantization matrices, or coordinate-transformation methods.

Julia consumers must accept the printer's dotted arithmetic operators.
The Python tests verify round trips and format validation; they do not establish Julia loading or spectral equivalence.

## Flux choices and comparison conventions

`Loops.from_circuit(circuit)` assigns a symbolic flux to every fundamental loop, including capacitively broken loops such as the J1–Cc–J2 loop of two capacitively coupled transmons.
Those fluxes are physical inputs and are preserved in the exported cosine offsets and linear terms; large exact conversion factors appear if a remaining flux parameter is measured in webers.
For a model where all capacitively broken loops carry zero flux, construct `loops = cq.Loops.from_circuit(circuit, zero_capacitive=True)` before quantization.
For finer control, use the `fluxes` mapping in `Loops.from_tree` or `Loops.from_branch_dict`, or explicitly substitute unwanted flux parameters to zero on export.
Nonzero flux in such a loop can be intentional, so export does not warn or silently zero it.

Coefficient comparisons also require matching the coordinate basis, flux gauge, parameter conventions, and constant energy reference.
The default irrotational flux assignment can put flux into `flux_linear` where another package puts it into junction offsets.
For a static flux these descriptions are related by a coordinate shift, potentially with a flux-only energy difference because this package drops that constant from the source Hamiltonian.
Use a custom `B` in `quantize` to match the other package's gauge explicitly; the complete matrix must satisfy `loops.G * B = I`, including rows for capacitors.
For time-dependent flux, a different `B` can also introduce derivative contributions to `n_g`; a static coordinate shift alone does not describe the full transformed dynamics.

### A zero-pi junction-offset gauge

This self-contained example puts half of the ring flux on each junction, producing the zero-pi Josephson potential `-2*E_J*cos(theta)*cos(phi - varphi_ext/2)` used by scqubits at symmetry.
The harmonic zeta coordinate is scaled to the FullZeroPi convention.
The capacitive-loop fluxes are explicitly zero, but the capacitor rows of `B` are still needed for the loop constraints.

```python
import sympy as sym
import circuit_quantization as cq

ECJ, ECB, EL, EJ = sym.symbols("E_CJ E_CB E_L E_J", positive=True)
flux = sym.Symbol("varphi_ext", real=True)
circuit = cq.Circuit([
    cq.Branch("J1", 1, 2, C=cq.ghz_to_farad(ECJ), EJ=EJ),
    cq.Branch("J2", 3, 4, C=cq.ghz_to_farad(ECJ), EJ=EJ),
    cq.Branch("L1", 2, 3, L=cq.ghz_to_henry(EL)),
    cq.Branch("L2", 4, 1, L=cq.ghz_to_henry(EL)),
    cq.Branch("C1", 1, 3, C=cq.ghz_to_farad(ECB)),
    cq.Branch("C2", 2, 4, C=cq.ghz_to_farad(ECB)),
])
cycles = {
    "ring": {"J1": 1, "J2": 1, "L1": 1, "L2": 1},
    "t1": {"J1": 1, "L1": 1, "C1": -1},
    "t2": {"J2": 1, "L1": 1, "C2": -1},
}
loops = cq.Loops.from_branch_dict(circuit, cycles, {
    "ring": cq.phase_to_weber(-flux), "t1": 0, "t2": 0,
})
s = 1 / sym.sqrt(2)
Z = sym.Matrix([[1, 1, s, 1], [-1, -1, s, 1],
                [-1, 1, -s, 1], [1, -1, -s, 1]]) / 2
half = sym.Rational(1, 2)
# Rows: C1, C2, J1, J2, L1, L2; columns: ring, t1, t2.
assert circuit.branch_names == ("C1", "C2", "J1", "J2", "L1", "L2")
B = sym.Matrix([[half, -1, 0], [half, 0, -1],
                [half, 0, 0], [half, 0, 0], [0, 0, 0], [0, 0, 0]])
assert loops.G * B == sym.eye(3)
res = cq.quantize(circuit, loops, Z, ("c", "e", "h", "cy"), B=B)
data = cq.to_spec(res, mode_names=("theta", "phi", "zeta"))
spec = cq.load_spec(data)
assert spec.flux_linear == sym.zeros(3, 1)
assert all(offset == -flux / 2 for _, _, offset in spec.cosines)
```

### FullZeroPi asymmetry and sign conventions

The built-in scqubits `FullZeroPi` model uses a first-order expansion in capacitance asymmetries, while this package computes the circuit's exact inverse capacitance matrix.
After matching conventions, expect exact agreement at zero capacitance asymmetry and second-order charging-matrix differences away from it; see [Groszkowski et al., equations (4)–(5)](https://arxiv.org/html/1708.02886v1#S2).
This limitation concerns that built-in approximate model, not a general scqubits `Circuit` constructed from the same branches.

For scqubits **4.3.1**, `ZeroPi.sparse_kinetic_mat` (also used by FullZeroPi) implements the junction-capacitance cross term with the opposite sign to the paper and its own documented `dCJ = (CJ1 - CJ2)/CJavg` convention; the `dEJ` potential term follows the paper.
With the junction ordering and phases in the example above, compare using `dCJ_scqubits = -dCJ_circuit` while keeping `dEJ` unchanged; see the [scqubits ZeroPi source](https://scqubits.readthedocs.io/en/v4.3/_modules/scqubits/core/zeropi.html).
Check this convention again when changing scqubits versions, particularly when both junction asymmetries are nonzero.
Also align offset-charge conventions and additive energy constants before comparing Hamiltonians or absolute energies.
