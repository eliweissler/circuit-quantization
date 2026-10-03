"""Properties that must hold for any circuit, checked on seeded random circuits."""

import pytest
import sympy as sym

from circuit_quantization import (
    Loops, decouple, default_tree, hamiltonian, irrotational_B,
    lagrangian, mode_basis, residual_freedom, transform,
)
from circuit_quantization.units import Phi0, e, h_GHz
from circuits import harmonium, kite, random_circuit

SEEDS = range(30)


def build(circuit, loops=None):
    """The quantize pipeline, step by step, on the automatic basis."""
    lag = lagrangian(circuit)
    Z, types = mode_basis(lag)
    if not set(types) & {"c", "e", "h"}:
        pytest.skip("no dynamical modes")
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
    # Rebuild H from the branch list alone: T = sum C_k (dPhi_k)^2 / 2 and U over inductors and
    # junctions, with the same B.  Then substitute Phi = Z varphi phi_0 and Q = Z^{-T} 2e n.
    circuit = build_circuit()
    lag, types, coords, loops, B, ham = build(circuit)
    if set(types) - {"c", "e", "h"}:
        pytest.skip("reference covers circuits without non-dynamical modes")
    nodes = circuit.internal_nodes
    Phi = sym.Matrix(sym.symbols(f"Phi_1:{len(nodes) + 1}", real=True))
    velocity = sym.Matrix(sym.symbols(f"v_1:{len(nodes) + 1}", real=True))
    node = dict(zip(nodes, range(len(nodes))))

    def across(x, b):
        return (x[node[b.tail]] if b.tail in node else 0) - (x[node[b.head]] if b.head in node else 0)

    offset = B * loops.Phi_ext
    kinetic = sum(b.C * across(velocity, b) ** 2 for b in circuit.branches) / 2
    C_int = sym.hessian(kinetic, list(velocity))
    Q = sym.Matrix(sym.symbols(f"Q_1:{len(nodes) + 1}", real=True))
    H = (Q.T * C_int.inv() * Q)[0] / 2
    for k, b in enumerate(circuit.branches):
        flux = across(Phi, b) + offset[k]
        if b.type == "L":
            H += flux**2 / (2 * b.L)
        elif b.type == "J":
            H -= b.EJ * h_GHz * sym.cos(2 * sym.pi * flux / Phi0)
    Z = coords.Z
    substitution = dict(zip(Phi, Z * ham.varphi * Phi0 / (2 * sym.pi)))
    substitution.update(zip(Q, Z.inv().T * ham.n * 2 * e))
    reference = H.subs(substitution, simultaneous=True) / h_GHz
    assert same_up_to_constant(ham.H, reference, ham)


@pytest.mark.parametrize("seed", SEEDS)
def test_hamiltonian_is_invariant_under_a_change_of_cycle_basis(seed):
    circuit = random_circuit(seed)
    _, _, _, loops, _, ham = build(circuit)
    G = sym.Matrix(loops.G)
    # One unimodular row operation G_j -> G_j + G_i that keeps every row a physical loop.
    for i in range(G.rows):
        for j in range(G.rows):
            row = G[j, :] + G[i, :]
            if i != j and any(row) and set(row) <= {-1, 0, 1}:
                T = sym.eye(G.rows)
                T[j, i] = 1
                break
        else:
            continue
        break
    else:
        pytest.skip("no single row operation keeps the loops physical")
    rebased = loops.rebase(T * G)  # Phi_ext' = T Phi_ext (eq:G_basis_change)
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
