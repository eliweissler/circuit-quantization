"""Properties that must hold for any circuit, checked on seeded random circuits."""

import pytest
import sympy as sym

from circuit_quantization import (
    Loops, compact_charge_decoupling, decouple, default_tree, hamiltonian, irrotational_B, spanning_trees,
    lagrangian, mode_basis, residual_freedom, transform,
)
from circuit_quantization.units import Phi0, e, h_GHz
from circuits import harmonium, kite, random_circuit

# Seed 7 has no dynamical modes; test_seed_without_dynamical_modes_is_rejected covers it.
SEEDS = [seed for seed in range(30) if seed != 7]


def build(circuit, loops=None):
    """The quantize pipeline, step by step, on the automatic basis."""
    lag = lagrangian(circuit)
    Z, types = mode_basis(lag)
    coords = decouple(lag, Z, types)
    loops = loops or Loops.from_tree(circuit, default_tree(circuit))
    B = irrotational_B(lag, coords, loops)
    return lag, types, coords, loops, B, hamiltonian(lag, coords, loops, B, circuit.t)


def canonical(H, ham):
    """Expand H, and the cosine arguments inside it, with compact sector labels set to 0."""
    H = sym.expand(sym.sympify(H).subs({s: 0 for s in ham.n_g.free_symbols if s.name.startswith("n_g")}))
    return H.replace(sym.cos, lambda arg: sym.cos(sym.expand(arg)))


def same_up_to_constant(H1, H2, ham):
    """H1 - H2 depends on no phase or charge (a flux-only constant is dropped in eq:U_full)."""
    difference = canonical(H1, ham) - canonical(H2, ham)
    return not sym.expand(difference).free_symbols & (set(ham.varphi) | set(ham.n))


@pytest.mark.parametrize("seed", SEEDS)
def test_assignment_satisfies_loop_and_irrotational_constraints(seed):
    lag, _, coords, loops, B, _ = build(random_circuit(seed))
    assert (loops.G * B - sym.eye(loops.G.rows)).applyfunc(sym.cancel).is_zero_matrix  # eq:GB_constraint
    assert (coords.Z.T * lag.Omega_int * lag.C_b * B).applyfunc(sym.cancel).is_zero_matrix  # eq:irrotational_expanded


@pytest.mark.parametrize("build_circuit", [harmonium, kite] + [lambda s=s: random_circuit(s) for s in SEEDS])
def test_hamiltonian_matches_an_independent_node_space_legendre_transform(build_circuit):
    # Rebuild H from the branch list alone, in the full decoupled basis Phi = Z_full y:
    # T = sum C_k (dPhi_k)^2 / 2 and U over inductors and junctions, with the same B.
    # Free directions keep a symbolic conserved charge, which must decouple from the retained
    # modes; frozen directions sit at the minimum of U; cyclic directions appear nowhere.
    # Then substitute y = varphi phi_0 and Q = 2e n on the retained modes.
    circuit = build_circuit()
    lag, types, coords, loops, B, ham = build(circuit)
    kinds = coords.types_full
    nodes = circuit.internal_nodes
    node = dict(zip(nodes, range(len(nodes))))
    y = sym.Matrix(sym.symbols(f"y_1:{len(nodes) + 1}", real=True))
    velocity = sym.Matrix(sym.symbols(f"v_1:{len(nodes) + 1}", real=True))
    Phi, dPhi = coords.Z_full * y, coords.Z_full * velocity

    def across(x, b):
        return (x[node[b.tail]] if b.tail in node else 0) - (x[node[b.head]] if b.head in node else 0)

    # Kinetic: Legendre transform over the directions with capacitance, retained and free.
    moving = [j for j, t in enumerate(kinds) if t in ("c", "e", "h", "fr")]
    kinetic = sum(b.C * across(dPhi, b) ** 2 for b in circuit.branches) / 2
    assert not kinetic.free_symbols & {velocity[j] for j in range(len(nodes)) if j not in moving}
    M = sym.hessian(kinetic, [velocity[j] for j in moving])
    charge = sym.Matrix([sym.Symbol(f"{'p' if kinds[j] == 'fr' else 'Q'}_{j + 1}", real=True) for j in moving])
    H = (charge.T * M.inv() * charge)[0] / 2

    # Potential, with each frozen direction at the minimum of U (it sees only inductors).
    offset = B * loops.Phi_ext
    U = 0
    for k, b in enumerate(circuit.branches):
        flux = across(Phi, b) + offset[k]
        if b.type == "L":
            U += flux**2 / (2 * b.L)
        elif b.type == "J":
            U -= b.EJ * h_GHz * sym.cos(2 * sym.pi * flux / Phi0)
    frozen = [y[j] for j, t in enumerate(kinds) if t == "fz"]
    if frozen:
        U = U.subs(sym.solve([sym.diff(U, x) for x in frozen], frozen, dict=True)[0])
    retained = [j for j, t in enumerate(kinds) if t in ("c", "e", "h")]
    assert U.free_symbols & set(y) <= {y[j] for j in retained}  # free and cyclic directions drop out
    H += U

    substitution = {y[j]: phase * Phi0 / (2 * sym.pi) for j, phase in zip(retained, ham.varphi)}
    substitution.update({sym.Symbol(f"Q_{j + 1}", real=True): 2 * e * n for j, n in zip(retained, ham.n)})
    reference = H.subs(substitution, simultaneous=True) / h_GHz
    # The free charges p_j may contribute their own energy, but nothing that couples to varphi or n.
    assert same_up_to_constant(ham.H, reference, ham)


@pytest.mark.parametrize("seed", SEEDS)
def test_hamiltonian_is_invariant_under_a_change_of_cycle_basis(seed):
    circuit = random_circuit(seed)
    _, _, _, loops, _, ham = build(circuit)
    # Any other spanning tree gives a unimodular change of basis G' = T G (eq:G_basis_change).
    other = next((candidate for tree in spanning_trees(circuit)
                  if (candidate := Loops.from_tree(circuit, tree)).G != loops.G), None)
    if other is None:
        pytest.skip("every spanning tree gives the same loops")
    rebased = loops.rebase(other.G, names=other.names)  # Phi_ext' = T Phi_ext
    *_, ham_rebased = build(circuit, rebased)
    assert canonical(ham_rebased.H, ham) == canonical(ham.H, ham)


@pytest.mark.parametrize("build_circuit", [kite] + [lambda s=s: random_circuit(s) for s in SEEDS])
def test_hamiltonian_is_invariant_under_residual_freedom(build_circuit):
    # Z -> Z S_1 is a canonical relabelling (Sec III.C): varphi = S_1 varphi', n = S_1^{-T} n'.
    circuit = build_circuit()
    lag, _, coords, loops, _, ham = build(circuit)
    allowed = residual_freedom(coords, keep_harmonic=False)
    # Upper-triangular entries only, so S_1 is unit triangular and invertible.
    off_diagonal = [(i, j) for i in range(allowed.rows) for j in range(allowed.cols) if i < j and allowed[i, j] != 0]
    if not off_diagonal:
        pytest.skip("no allowed mixing between retained modes")
    S = sym.eye(allowed.rows)
    for i, j in off_diagonal:
        S[i, j] = 1
    new = transform(lag, coords, S)
    B = irrotational_B(lag, new, loops)
    ham_new = hamiltonian(lag, new, loops, B, circuit.t)
    old_in_new = ham.H.subs(dict(zip(ham.varphi, S * ham_new.varphi)) | dict(zip(ham.n, S.inv().T * ham_new.n)),
                            simultaneous=True)
    assert same_up_to_constant(ham_new.H, old_in_new, ham)


def test_seed_without_dynamical_modes_is_rejected():
    # The one random circuit build() skips: every mode is free or frozen, so there is nothing to quantize.
    lag = lagrangian(random_circuit(7))
    Z, types = mode_basis(lag)
    assert not set(types) & {"c", "e", "h"}
    with pytest.raises(ValueError, match="no dynamical modes"):
        decouple(lag, Z, types)


@pytest.mark.parametrize("seed", SEEDS)
def test_compact_charge_decoupling_is_a_canonical_relabelling(seed):
    circuit = random_circuit(seed)
    lag, _, coords, loops, _, ham = build(circuit)
    c = [j for j, t in enumerate(coords.types) if t == "c"]
    n = [j for j, t in enumerate(coords.types) if t != "c"]
    S = compact_charge_decoupling(coords)
    if not (c and n):  # nothing to shear
        assert S == sym.eye(len(coords.types))
        return
    new = transform(lag, coords, S)
    assert new.C_tilde.extract(c, n).applyfunc(sym.cancel).is_zero_matrix
    assert (new.Lstar_tilde - coords.Lstar_tilde).applyfunc(sym.cancel).is_zero_matrix
    ham_new = hamiltonian(lag, new, loops, irrotational_B(lag, new, loops), circuit.t)
    old_in_new = ham.H.subs(dict(zip(ham.varphi, S * ham_new.varphi)) | dict(zip(ham.n, S.inv().T * ham_new.n)),
                            simultaneous=True)
    assert same_up_to_constant(ham_new.H, old_in_new, ham)

