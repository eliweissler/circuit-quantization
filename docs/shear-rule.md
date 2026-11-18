# How far to shear a harmonic mode onto a junction mode

`shear_S1(ham, pairs)` (`src/circuit_quantization/shear.py`) chooses the frame `x_f = y_f + a x_J` for given pairs of a
harmonic mode `f` and a junction mode `J` by the rule below. It is meant for a hierarchical (tree) solve that truncates
between `f` and `J`: you know the tree, so you name the pairs.

```python
ham = cq.numeric_hamiltonian(result, values)
S1, info = cq.shear_S1(ham, [(f, J), ...])     # mode indices in ham.types order
print(info)                                    # Omega_f, omega1, omega2, a*, s*, S[f, J] per pair; moments per junction
sheared = result.transform(S1)                 # or ham.transformed(S1)
```

## The rule

In a frame where `E_C` has no `f`–`J` entry and the pair couples through `E_L[f, J] x_f x_J`, the full shear
`a* = −E_L[f, J] / E_L[f, f]` removes that coupling. Shear by the fraction

    s* = (X + c) / (X + 2c + P),     c = 4 E_C[J, J] / Ω_f,     P = (8 E_C[J, J] / Ω_f)² ⟨n_J²⟩,

of `a*`, where `X = ⟨δx_J²⟩` and `⟨n_J²⟩` are ground-state moments of the junction mode's own Hamiltonian
`4 E_C[J, J] n² + ½ E_L[J, J] x² + b_J x − Σ E cos(k x + δ)` and `Ω_f = √(8 E_C[f, f] E_L[f, f])`. Equivalently

    s* = Ω_f / (Ω_f + ω_eff),   ω_eff = (ω₁ + ω₂²/Ω_f) / (1 + ω₁/Ω_f),   ω₁ = 4 E_C[J, J] / ⟨δx_J²⟩,   ω₂ = 8 E_C[J, J] √(⟨n_J²⟩/⟨δx_J²⟩):

full shear for a frame mode much stiffer than the junction's motion, none for a much softer one, half at resonance.
A harmonic junction has `ω₁ = ω₂ = ω_J`. For a junction whose low states spread over several wells (a fluxonium at
half flux) `ω_eff` is far below the well frequency: the phase fluctuation is carried by the slow inter-well
transition. The shear changes the junction's Hamiltonian (its inductance moves toward the Schur complement) and
`Ω_f` (`E_C[f, f]` gains `a² E_C[J, J]`), so `shear_S1` iterates to self-consistency; it converges in a few
iterations. The moments come from a grid solve of the one-mode Hamiltonian: eighth-order stencils, sized like
`default_bases` (about a hundred points, converged to eight digits on the cases below) and widened until the ground
state stays off the edges. The offset charge of an extended mode is a gauge choice and drops out.

The shear only matters when the tree truncates between `f` and `J`: if they are solved together in an untruncated
product space, any shear between them is a change of basis inside it.

## Derivation

**Setting.** One junction coordinate `x` with any single-coordinate potential `V(x)` and one harmonic mode `f`:

    H = 4E_C n² + V(x) + 4E_Cf n_f² + ½E_Lf f² + g x f.

Assumptions: (A1) `E_C` has no junction–frame entry and the frame block of `E_L` is diagonal, so every `(f, J)` pair of a
larger circuit has this form and the pairs add. (A2) The coupling is treated to first order in the states. (A3) In the
low states of interest the frame mode is in its ground state.

**Step 1, the sheared Hamiltonian (exact).** With `f = y + a x`, `a = −s g/E_Lf`, the momenta become `n → n − a n_y`,
`n_f = n_y`, and

    H = [4E_C n² + V(x) − (2s − s²)(g²/2E_Lf) x²] + [4E_Cf′ n_y² + ½E_Lf y²] + (1 − s) g x y + 8E_C s (g/E_Lf) n n_y,
    E_Cf′ = E_Cf + a² E_C.

The first bracket is the junction mode's Hamiltonian `H_J`, the second the frame mode with `Ω = √(8E_Cf′E_Lf)` and
zero-point spread `σ = (2E_Cf′/E_Lf)^¼`.

**Step 2, the coupling in the junction's eigenbasis (exact).** With `y = σ(b + b†)`, `n_y = i(b† − b)/2σ` and
`2σ²E_Lf = Ω`, the coupling is `H_c = b† O + b O†` with `O = g σ [(1 − s) x + i s (8E_C/Ω) n]`. Because `V` depends on
`x` only, `[H_J, x] = −8i E_C n`, so between eigenstates of `H_J`, `⟨m| i 8E_C n |k⟩ = −ω_mk x_mk` with
`ω_mk = E_m − E_k`, and

    ⟨m| O |k⟩ = g σ x_mk [(1 − s) − s ω_mk/Ω].                                                        (1)

The inductive coupling acts on a junction transition with weight 1; the charge coupling the shear creates acts with
weight `ω_mk/Ω` and the opposite sign. A full shear multiplies every junction–frame matrix element by `ω_mk/Ω`: a
high-pass filter on the junction's transitions.

**Step 3, harmonic junction (exact).** If `H_J` is an oscillator of frequency `ω`, (1) gives

    H_c = g σ_x σ { [(1 − s) − s ω/Ω] (a†b† + a b) + [(1 − s) + s ω/Ω] (a†b + a b†) },

so `s = Ω/(Ω + ω)` removes the pair-creation term exactly, leaving an exchange term that does not touch the product
vacuum.

**Step 4, general junction (approximation).** By (A3) and (1), the weight `H_c` moves from `|k, 0_f⟩` into the sector
with one frame quantum is

    ‖P₁ H_c |k, 0⟩‖² = g²σ² Σ_m |x_mk|² [(1 − s) − s ω_mk/Ω]² = g²σ² [(1 − s)² X − 2s(1 − s) c + s² P],     (2)
    X = ⟨δx²⟩_k,   c = 4E_C/Ω,   P = (8E_C/Ω)² ⟨n²⟩_k,

using the f-sum rule `Σ_m ω_mk |x_mk|² = 4E_C` for the cross term (the `m = k` term, a static force on the frame
mode, is left out of `X`). When the tree cuts between `J` and `f` and cannot keep the states with a frame quantum, the
second-order error of level `k` is (2) with each term divided by its denominator `ω_mk + Ω`. Taking the denominators as
equal (valid to relative order `ω/Ω` of the transitions that carry the weight; this is the approximation), the error is
smallest at

    s* = (X + c) / (X + 2c + P),      residual  g²σ² (X P − c²)/(X + 2c + P).                             (3)

For a harmonic junction (3) reduces to step 3 with zero residual; for any junction `X P ≥ c²` (Cauchy–Schwarz), so the
residual is the junction's anharmonicity.

**Several junctions and frame modes.** Under (A1), for `x_F = y_F + A x_J`, the rule applies to each entry:
`A_fJ = s*_fJ a*_fJ`. Two side effects are not in the rule: each frame mode's couplings to other coordinates are copied
onto the junction with factor `A_fJ` (junction–junction terms of the Schur complement), and `E_C[F, F]` gains
`A E_C[J, J] Aᵀ`.

## Checks

**Toy model** (one junction plus one oscillator, `E_Cf = 0.3874`, `g = 0.25 √(E_L E_Lf)`, the frame mode's `E_Lf`
scanned; best `s` from a tree that keeps `k_J` junction eigenstates and `k_f` Fock levels, scored by the worst error of
the transitions among the lowest six levels against a converged solve). The rule's fraction is the optimum to within
a few hundredths once the frame keeps three or more levels and the junction is not cut hard:

| junction | `E_Lf` | `Ω_f` | `s*` (rule) | best `s`, tree (40, 3) | (10, 4) |
|---|---|---|---|---|---|
| single-well coupler (`E_C` 0.775, `E_L` 12.11, `E_J` 3.5, flux 0.3) | 3.27 | 3.24 | 0.277 | 0.40 | 0.31 |
| | 20 | 7.94 | 0.487 | 0.53 | 0.52 |
| | 150 | 21.6 | 0.722 | 0.74 | 0.73 |
| fluxonium at half flux (`E_C` 0.553, `E_L` 1.635, `E_J` 8.98) | 3.27 | 3.23 | 0.801 | 0.83 | 0.69 |
| | 68.4 | 14.6 | 0.971 | 0.95 | 0.46 |
| | 150 | 21.6 | 0.982 | 0.96 | 0.55 |

For the coupler at `E_Lf = 3.27` (a frame mode softer than the junction) the full shear makes the (10, 4) tree's error
four times worse than no shear (996 against 237 MHz) and the rule's fraction five times better (46.5 MHz). For the
fluxonium the well frequency would give 0.69 instead of 0.971, far off: for a multi-well junction `ω_J` is not the well
frequency. When the junction is cut hard (10 or fewer states) the optimum is lower and scattered; the rule gives only
its direction there. `tests/test_shear.py` checks `shear_S1` against this table (0.971 and 0.277).

**A circuit** (two fluxonia and a tunable coupler, Hensel et al., arXiv:2611.05627; KronFreeCircuits.jl bench,
`hensel_two_fluxonia`). For the stiff frame mode against a fluxonium (Ω_f = 14.6 against ω₁ = 0.32, ω₂ = 1.35 GHz) the
rule gives 0.971; the error of a binary tree, scanned in the shear, is smallest at 0.970 to 0.974 over three tree
sizes. For the soft frame mode against the coupler (Ω_f = 3.2 against a well frequency of 7.0 to 9.6 GHz over the
flux sweep) the rule gives 0.26 to 0.31 and the scanned optimum is 0.34 to 0.36: a third of the shear is 3 to 8 times
more accurate than none and 30 to 150 times more accurate than the full shear in the tree that splits them. A
three-fluxonium circuit in the frame the rule picks was, at the same tree size, about 8 times more accurate than with
hand-picked fractions and 17 to 650 times more accurate than with the full shear.

## What the rule does not cover

- Shears with no direct coupling to act on (a frame mode onto a junction it does not couple to), and mixing among the
  frame modes: these trade couplings across particular cuts of a particular tree.
- The junction-cut regime (frame states all kept, junction states truncated hard): the optimum is below `s*`.
- Second order only: one frame quantum, equal energy denominators, frame mode in its ground state. For an occupied
  frame mode the optimum moves up toward 1.
- Flux dependence: `ω_J` moves with the external flux (the coupler's `s*` from 0.26 to 0.31); one frame for a whole
  sweep is a compromise.
- Cost: a partial shear keeps both the inductive and the charge coupling, one more term per pair than `s = 0` or 1.
- Junction cosines must act on the junction mode alone (`shear_S1` checks this), as in a local frame.
