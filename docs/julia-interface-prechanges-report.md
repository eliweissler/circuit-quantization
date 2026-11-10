# Julia interface prechanges (2026-11-10)

Implemented on branch `julia-interface-prechanges`, based on
`0054685c240a250e480464d4e5e16cf17dedf976`. Nothing was pushed.
The source checkout could only be read in this session, so implementation and
validation used an isolated checkout. The accompanying four-commit patch applies
to the source checkout without touching its existing untracked files.

1. Residual transformations snap floating compact entries within absolute
   tolerance 1e-12 to exact integers, including normalization of float zeros.
   Exact noninteger inputs remain invalid. Errors identify the offending entry.
2. Fully numeric quadratic matrices are symmetrized on export when their
   asymmetry is within 1e-10 of the matrix scale. Floating off-diagonal pairs
   below 1e-15 of that scale become exact zeros. Larger discrepancies report the
   pair. Symbolic export and loader symmetry rules are unchanged.
   Transforming a Result also reuses its already validated flux assignment;
   newly supplied assignments still receive their existing exact checks.
3. `numeric_hamiltonian` accepts a Result or HamiltonianSpec and parameter values.
   `NumericHamiltonian.transformed` checks admissibility, updates mode types, and
   transforms source and compact-sector charges separately. Its frame changes
   contain no SymPy calls. Existing callers use this same checked method.
4. `to_record` and `load_record` preserve circuit topology, expressions and symbol
   assumptions, loops, flux assignments, full coordinates, cumulative residual
   frame, and explicit parameter substitutions. Record generation includes the
   git commit when available. Loading uses the existing expression parser.

README and spec-format documentation cover the new behavior and APIs.

## Performance

The KITE from `examples/product_state_demo.py`, using E_J=16.83, E_CJ=4.82,
E_L=1.27, E_CS=0.072, epsilon=0.03 and delta=0, measured **36.3–43.7 microseconds
per frame change** (five repeats of 1,000 calls; median 37.0 microseconds).
The tested S was:

```python
S = np.array([[1., .2354, .2306],
              [0., .6962, .7339],
              [0., -.5719, .629]])
```

This excludes the one-time symbolic-to-numeric conversion, as intended.

## Clarifications to the brief

- `numeric_hamiltonian` was already public and exported; it now additionally
  accepts a Result and separates the charge contributions.
- Float zero entries also needed normalization for symbolic block validation.
- `numerify(...).transform(S)` could fail before export because its unchanged B
  was revalidated exactly. This is addressed by reusing the validated assignment.
- The symbolic route creates fresh compact sector labels. Comparing with an
  unchanged substitution dictionary only describes the same sector when those
  values are invariant under S_cc transpose. Tests assign the transformed sector
  values to the fresh labels, and exclude sector offsets on noncompact modes.
- The example's private imports are mean-field solver internals, unrelated to
  numeric coefficient conversion, so no migration was necessary there.
- Rebuildable numeric records accept values through `to_record(..., subs=...)`.
  Previously erased substitutions from an already numerified Result cannot be
  inferred from its unchanged circuit metadata. Record the symbolic Result and
  pass its values explicitly. Unsupported undefined functions and arbitrary
  Python node objects are rejected rather than serialized as executable code.

## Validation

Validation uses the `circuit_quantization` conda environment with `PYTHONPATH=src`
so imports resolve to this checkout instead of the original editable install.
Regression tests cover both requested circuits, nonzero sector charges, numeric
source drives, compact/noncompact shears, harmonic reclassification and restoration,
invalid transformations, the direct numerify/float-transform export failure,
all nine named circuit fixtures, numeric records, separate static flux assignments,
and hostile metadata expressions. See the accompanying test log for full-suite
results.

Full suite: **800 passed, 4 skipped**, 70 warnings, in 266.28 seconds. The skips
are one opt-in `--runslow` Z3 test and three parameterizations with nothing to
diagonalize. Warnings come from optional scqubits/pyparsing dependencies. An
additional static-assignment record regression added after suite collection
also passed; all 11 record-specific tests pass together.
