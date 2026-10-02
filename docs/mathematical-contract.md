# Mathematical contract

The source of truth is `Notes/full_matrix_quantization.tex`.
Code comments and tests cite its `\label` names; `tools/eqrefs.py` checks that every cited label exists.

| Source | SHA-256 |
|---|---|
| `Notes/full_matrix_quantization.tex` | `0013cd4c6bc438ac39928ff6dda52df6435d1a066ef2258c0ccbfa2c724664ec` |

When the code and the notes disagree, the notes take precedence: fix the code, or raise the discrepancy against the notes.

The truncation update uses `eq:Z_truncated` and `eq:R_capacitance`: retained
coordinates are extracted by `R = P Z_dc^{-1}`, selecting rows only after
inverting the full decoupled matrix, including non-dynamical columns.
The capacitance identity is an equivalent check, not the construction used.
The particular flux assignment `eq:B_irrot_direct_closed` with `W = 0` is
unchanged; its homogeneous freedom satisfies `R W = 0`.
