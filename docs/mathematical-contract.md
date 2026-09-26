# Mathematical contract and provenance

The source of truth is the current working copy of
`/Users/eweissler/overleaf/circuit_quantization/Notes/full_matrix_quantization.tex`.
The implementation follows its branch model, SI Lagrangian, projected
irrotational condition, and frequency-unit appendix.

## Core invariants

For the oriented incidence and loop matrices,

```text
G Omega.T = 0
G B = I
Z.T Omega_int C_b B = 0
```

The retained transformation `Z` contains only compact, extended, and harmonic
columns. Free, frozen, and cyclic columns are decoupled before removal. In
practice, every dynamical column is decoupled separately with the reverse
one-ND-mode-at-a-time sweep from `sircuitenum.quantize.decouple_column`; the
implementation does not invert a combined rank-deficient ND block.

The frequency-unit matrices are

```text
E_C = e^2/(2h) C_tilde^-1
E_L = phi_0^2/h L_tilde^*
```

and are divided by `1e9` for GHz. The returned Hamiltonian is

```text
4 (n-ng).T E_C (n-ng)
+ 1/2 varphi.T E_L varphi
+ varphi.T omega_L_tilde E_Lb B_L varphi_ext
- sum_k EJ_k cos((omega_J_tilde.T varphi + B_J varphi_ext)_k).
```

## Pinned working-copy inputs

| Source | SHA-256 |
|---|---|
| `Notes/full_matrix_quantization.tex` | `268609ad2b6e8dfe679ed0f91b7bca7dd98d801aa5d3b0e0a8aaee67a6dd1a24` |
| `Notes/Variable Transformations/Short_Var_Trans_notes.tex` | `caf5986b6c92b72946314eff834bbac53dc8fff7bd00d38dd33b0d18d9a18d8e` |
| `Notes/drive_loop_flux_assignment.md` | `0d7b148d9ab4d01c26cd17b9a836b4c00efa009fc4d2f751a1fd587951864b2f` |
| `sircuitenum/quantize.py` | `789e3280adc6aee91e748feaf932cc3c3199152b660a2a51a861b222027888f5` |
| `sircuitenum/equationset.py` | `72fd3ce7b38cc0fcebfffb84977fc2b710e4a11e5c7a53def6544bc30fe0b7cf` |
| `sircuitenum/z3_interface.py` | `49a9389176977c4f0f86702d4c70ad5659c4a09d9001f39d0b7e3df272187730` |
| `sircuitenum/singular_interface.py` | `c5ab57d229dac8b776dbc43e73c9536519d1a2ebfb7b3ceaa45a885aa024bf78` |
| `tests/test_quantize.py` | `a274e7aa9f838c2778cce7c1dd3ae65455ce7e843e9b4c3ca95dbd311b788346` |

When a historical test conflicts with the pinned note, the note takes
precedence and the regression is rewritten around the derived equation.
