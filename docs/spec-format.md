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
| `E_C`, `E_L` | arrays of arrays of expression strings | Required, each exactly N by N, where N is the number of modes. |
| `flux_linear`, `n_g` | arrays of expression strings | Required, each of length N. |
| `cosines` | array of objects | Required, possibly empty; each object requires `energy`, `omega`, and `offset`. |
| `constant` | expression string | Optional additive energy, default `"0"`. |
| `charge_cosines` | array | Shared-format extension for phase slips; this Python version accepts only an absent or empty array. |
| `_generator` | object | Export provenance: `package`, installed `version` (or `"unknown"`), and ordered branch names in `branches`. |

Mode types mean compact (`c`), extended (`e`), and harmonic (`h`).
All matrices, vectors, and cosine multipliers follow `modes` order, which export takes from `res.coords.types`.
Only retained dynamical modes are exported; free, frozen, and cyclic coordinates are not modes in this file.
Mode names are descriptive labels: `HamiltonianSpec.hamiltonian()` uses real coordinate symbols `varphi_1` through `varphi_N` and `n_1` through `n_N`, in mode order.
Parameter names that collide with these coordinate symbols are rejected by Python.

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

The dotted operators `.+`, `.-`, `.*`, `./`, and `.^` have the same scalar meanings.
Operator precedence follows [Julia's arithmetic precedence](https://docs.julialang.org/en/v1/manual/mathematical-operations/#Operator-Precedence-and-Associativity): powers, unary signs, rational division `//`, multiplication/division, and addition/subtraction.
Powers associate to the right; the remaining binary operations associate to the left.
`//` requires rational numeric operands; use `/` for symbolic division.
Named functions take one argument, except `log(base, value)`, which is also supported.
Implicit multiplication, dotted function calls, assignments, indexing, attributes, strings, comments, and other executable constructs are not part of this grammar.

Parameter names use a portable subset of Julia identifiers: a Unicode letter or underscore followed by letters, digits, or underscores, excluding all-underscore names, Julia keywords, constants, and function names.
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

`to_spec(res, *, subs=None, simplify=True)` produces the dictionary without modifying `res`.
Substitutions are applied before exact physical constants are evaluated; `simplify=True` cancels rational expressions before printing.
`save_spec(res, path, **kwargs)` validates the dictionary before opening the destination, writes indented UTF-8 JSON, and returns that dictionary.
`load_spec(source)` accepts a dictionary, JSON text, or a path, and returns a frozen `HamiltonianSpec` with immutable matrices and tuples.
Use `pathlib.Path` for filenames that could be mistaken for JSON text, for example a filename beginning with a digit.
Calling `spec.hamiltonian()` assembles the saved Hamiltonian without repeating mode classification, basis selection, or flux assignment.
It does not restore `Result`, circuit objects, intermediate quantization matrices, or coordinate-transformation methods.

Julia integration remains a separate task: the consumer must accept the printer's dotted arithmetic operators before relying on all exported files.
The Python tests verify round trips and format validation; they do not establish Julia loading or spectral equivalence.
No Julia code or fixtures are changed by this implementation.
