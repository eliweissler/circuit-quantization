# Minimal core rebuild — design

Date: 2026-09-26
Status: approved in conversation; awaiting written-spec review

## Purpose

Replace the current Codex-built package with a minimal, readable reference implementation to publish alongside the circuit quantization paper.
The source of truth is `Notes/full_matrix_quantization.tex`.
A reader should be able to open a section of the notes and the matching module side by side and follow every equation into code.

Primary use is from Jupyter notebooks.
Output is symbolic (sympy) only.

## Scope

In the first core release:

- Branch-centric circuit model with capacitance as a branch attribute (Sec I).
- Voltage-source nodes and the `V(t)` offset-charge drive.
- User-supplied complete cycle basis with visible flux symbols; helper to generate one from a spanning tree; basis change by unimodular `T`.
- Mode classification (Table I), compact normalization, decoupling and truncation of free/frozen/cyclic modes.
- User-supplied square transformation `Z_full`, validated by the same checks as the automatic one.
- Residual freedom `S_1` (keep harmonic / allow harmonic→extended) with a validating `transform`.
- Irrotational flux assignment `B`.
- Frequency-unit (GHz) symbolic Hamiltonian.

Deferred to later optional add-ons (the core must leave clean hooks for them, but not implement them):

- Canonical coordinate optimizer, as an add-on depending on `sircuitenum` (not vendored).
- Circuit drawing (sircuitenum or circuitikz).
- Enumeration over spanning trees / cycle bases.
- Numeric evaluation and export to scqubits.

## Conventions

- **Units.** Inputs are SI: `C` in F, `L` in H, `V` in V, loop flux in Wb, time in s. `EJ` is `E_J/h` in GHz. The Hamiltonian is `H/h` in GHz. Converters `ghz_to_farad`, `farad_to_ghz`, `ghz_to_henry`, `henry_to_ghz` (eq:EC_EL_scalar) are exported at top level. Constants are exact SI rationals.
- **Ordering.** Nodes are ordered `int | gnd | src`; branches are ordered by type `C | J | L | V` (eq:Omega_columns). Within a group, user order is kept. `Result` carries name↔index maps so users never depend on the ordering.
- **Orientation.** `Omega` column `k` has `+1` at the branch tail and `-1` at the head (Sec I.B).
- **Names.** Attributes use the notes' symbols in a fixed ASCII scheme: `Omega`, `Omega_int`, `omega_J_int`, `omega_L_int`, `C_b`, `C_int`, `C_int_src`, `Lstar_int`, `G`, `Phi_ext`, `Z`, `types`, `C_tilde`, `Lstar_tilde`, `omega_J_tilde`, `omega_L_tilde`, `B`, `E_C`, `E_L`, `E_Lb`, `n_g`, `H`. The package docstring gives the table once.
- **Equation references.** Comments and docstrings cite `\label` names plus the section, e.g. `# eq:B_irrot_direct_closed (Sec V.D.1)`. Labels are stable while numbers change. `tools/eqrefs.py` checks every cited label exists in the `.tex` and can print final numbers before publication.

## Package layout

```
src/circuit_quantization/
  __init__.py     public API and symbol table
  units.py        Sec VI   constants and converters
  circuit.py      Sec I    Branch, Circuit, Loops, fundamental_cycles
  lagrangian.py   Sec II   element and incidence matrices
  modes.py        Sec III  mode_basis, check_basis, decouple, residual_freedom, transform
  flux.py         Sec V    irrotational_B
  hamiltonian.py  Sec IV/VI  hamiltonian
  quantize.py     quantize() and Result
```

Target size: roughly 600–800 lines of code excluding comments.
Plain functions over frozen dataclasses; no class hierarchies, no caching layers.

## Components

### `circuit.py` (Sec I)

`Branch(name, tail, head, C=0, L=None, EJ=None, V=None)`.
At most one of `L`, `EJ`, `V`; the type (C, J, L, V) follows.
Validation per Table `tab:elements` and Sec I.A:

- a junction requires `C` provably positive;
- a pure capacitor requires `C` provably positive;
- a source requires `C = 0` and must touch ground; one source per source node;
- only pure-capacitor branches may end on a source node;
- `L > 0`, `EJ > 0` where sympy can decide.

`Circuit(branches, ground, t=Symbol("t", real=True))` holds the branch list and ground node.

`fundamental_cycles(circ, tree=None) -> dict[str, dict[str, int]]`.
Computes `G` exactly as in Appendix A: delete the ground (root) row of `Omega`, split into tree and closure blocks, and set `G = (G_t  I)` with `G_t = -Omega_c^T (Omega_t^T)^{-1}` (eq:tree_B).
Returns the rows as signed branch dicts keyed by closure branch name.
It returns data only; the user inspects or edits it and passes it to `Loops`.

The tree is the user's list of branch names, or by default a spanning forest of the `S_ind` subgraph (junctions and inductors) extended by `S_cap` branches, each grown in the user's branch order.
Growing `S_ind` first makes the closure rows inside `S_ind` span that subgraph's cycle space, so the default `G` comes out in the superconducting / capacitively broken block order of eq:G_blocks.
A projector onto `null(Omega)` built from the full Moore–Penrose pseudoinverse is not used, since its rows are not in `{0, ±1}` as `G` requires.

`Loops(circ, cycles, fluxes=None)`.

- `cycles` must form a complete basis of the cycle space: `G Omega^T = 0` (eq:G_OmegaT), independent rows, `N_ell = N_b - N_v + N_components` rows. Otherwise raise, stating how many are missing or which are dependent.
- Every cycle not listed in `fluxes` gets a visible real symbol `Phi_<name>`. Zero flux only by explicit `fluxes={name: 0}`. A time-dependent drive is passed as an expression in `circ.t`.
- Exposes `G` and `Phi_ext`.
- `rebase(G_new)`: compute `T = G_new G^+`; require `T` integer with `det T = ±1`; return `Loops` with `G_new` and `T Phi_ext`. Physically identical (`B' Phi_ext' = B Phi_ext`).

### `lagrangian.py` (Sec II)

`Lagrangian` dataclass built from a `Circuit`: `Omega` and its row blocks, `C_b` (eq:Cb_diag), `C_full` and its partition (eq:C_full_partition) giving `C_int` and `C_int_src`, `L_b^{-1}`, `Lstar_int`, `omega_J_int`, `omega_L_int`, source voltages.
No flux data; loops are separate.

### `modes.py` (Sec III)

1. `mode_basis(lag) -> (Z_full, types)`: square `Z = (Z^c Z^e Z^h Z^fr Z^fz Z^cy)` (eq:Z_mode_blocks) from the nullspaces `J`, `L`, `C` per `tab:var_basis`. Compact block normalized by `S_cc = (omega_{J,K}^T)^{-1}` (eq:Scc_selection).
2. `check_basis(lag, Z_full, types)`: square and invertible; each block satisfies its Table I membership and exclusion conditions; `omega_{J,c}^T` integer and the compact lattice primitive (eq:compact_period_lattice). Used by both the automatic and user-supplied paths.
3. `decouple(lag, Z_full, types) -> Coordinates`: computes the retained block `Z_1 + Z_2 M` of eq:ZY_blocks with `M_C`, `M_L` from eq:decouple_final and eq:decouple_final_L, then truncates. Implemented one dynamical column / one non-dynamical mode at a time so every inverted block is invertible; comments tie each step to the block formula. Asserts the capacitance and susceptance cross blocks vanish and no junction sees a discarded mode.
4. `residual_freedom(coords, keep_harmonic=True) -> Matrix`: symbolic `S_1` with free symbols in the allowed blocks of `S_{1,a}` (`keep_harmonic=True`) or `S_{1,b}` (eq:S1_structure).
5. `transform(coords, S1) -> Coordinates`: check `S1` matches the allowed zero pattern and `S_cc ∈ GL(N_c, Z)` (eq:Scc_unimodular); apply `Z -> Z S1`; update `types` (harmonic → extended where `S_eh ≠ 0`).

`Coordinates` holds `Z`, `types`, `C_tilde` (eq:Zd_schur), `Lstar_tilde`, `omega_J_tilde`, `omega_L_tilde`.

### `flux.py` (Sec V)

`irrotational_B(lag, coords, loops)`: `B = (I - Omega_int^T Z C_tilde^{-1} Z^T Omega_int C_b) G^+` with `G^+` from eq:G_right_inverse (eq:B_irrot_direct_closed, `W = 0`).
Asserts `G B = I` (eq:GB_constraint) and `Z^T Omega_int C_b B = 0` (eq:irrotational_expanded).

### `hamiltonian.py` (Sec IV, VI)

`hamiltonian(lag, coords, loops, B)` builds eq:full_H_dimless one labelled term per line:

- `E_C`, `E_L`, `E_Lb` (eq:EC_EL_matrix), in GHz;
- `n_g` from eq:Q_g_tilde_expanded: source part `-Z^T C_int_src V(t)`; EMF part computed and asserted zero; plus a static sector symbol `n_g<j>` on each compact mode (Sec IV.B.1, as in eq:harm_H_dimless);
- charging, inductive, inductive–flux cross term, Josephson cosines.

Symbols `varphi_i`, `n_i` (real). No global `simplify`.

### `quantize.py`

`quantize(circ, loops, Z_full=None, types=None, coords=None) -> Result`.
With neither, runs `mode_basis`; with `Z_full`/`types`, uses them; both then go through `check_basis` and `decouple`.
With `coords` (e.g. from `transform`), skips straight to flux and Hamiltonian.
`Result` is a frozen dataclass exposing every named quantity above plus `lag`, `coords`, `loops`, and `_repr_latex_` showing `H` with mode counts and the `Phi` symbols present.

## Error handling

Validation raises `ValueError` with a message naming the offending branch, loop, or matrix block.
Internal consistency checks (conditions the notes prove must hold) use `assert`-style checks that raise with the violated equation label.
No warnings-based fallbacks.

## Testing

pytest, one file per module.

- **Appendix B worked examples** against their labelled closed forms: harmonium (`eq:harm_Z`, `eq:harm_B`, `eq:harm_H`), 0–π (`eq:zp_Z`, `eq:zp_B`, `eq:zp_H`), cos 2φ kite (`eq:kite_Z`, `eq:kite_H`), asymmetric rhombus (`eq:rh_Z`, `eq:rh_B`).
- **Body examples:** `fig:coupled_circuit_example` (`eq:fig1_nullspaces`, `eq:fig1_Z`, `eq:fig1_schur`, 9 rows of `K`); isolated fluxonium and SQUID dividers; shared-inductor `B` (`fig:shared_inductor`); shunt placement (`eq:shunt_placement`); coupler loop (`eq:coupler_loop`).
- **Default cycle basis:** `G Omega^T = 0`, entries in `{0, ±1}`, superconducting rows first and spanning the `S_ind` cycle space (eq:G_blocks); includes the source-closing loop of `fig:coupled_circuit_example`.
- **Invariants:** `rebase` preserves `B Phi_ext`; `transform` gives congruent `C_tilde`, `Lstar_tilde`; per-column decoupling equals `Z_1 + Z_2 M`.
- **Cross-check:** transmon molecule against scqubits; skipped if scqubits is absent.

## Examples

`examples/` holds one short notebook per Appendix B circuit: build the circuit, display `H` and the key matrices, point to the notes section.
Existing Mathematica notebooks move unchanged to `examples/mathematica/`.

## Migration

1. New branch. First commit snapshots the current code as a reference.
2. Build the new package in place under `src/circuit_quantization/`.
3. Old tests are a behavioral reference only; no code carried over.
4. Remove the old implementation, the vendored optimizer, `_canonical.py`, and `egg-info` once the new tests pass.
5. Delete root scratch files: `check_*.py`, `confirm_opus5.py`, `test_script*.py`, `sec1.tex` (kept in git history via step 1).
6. Rewrite `README.md` and `docs/mathematical-contract.md`; re-pin the notes' hash.

Dependencies: `sympy`, `networkx`. Test extras: `pytest`, optional `scqubits`.
