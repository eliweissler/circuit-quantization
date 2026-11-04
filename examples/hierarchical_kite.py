"""Hierarchical diagonalization of the KITE: does the product-state frame help?

Three single-mode leaf bases are compared in three frames.  ``bare`` keeps
each mode's own quadratic terms only, since every junction cosine spans all
three modes (the leaves of a kron-free tree without mean fields).
``mean-field`` takes the lowest states of the converged mean-field Hamiltonian
of the lowest-energy product state.  ``two-well`` orthonormalizes the union of
the mean-field bases of the two degenerate product-state wells, interleaving
their levels.  The full Hamiltonian is projected onto the product of the
truncated bases (``m`` states per mode) and diagonalized densely; the errors
are against the exact ground doublet and first excited state from
``product_state_demo.py kite``.

Run from the package root::

    python examples/hierarchical_kite.py
"""

import sys
import time

import numpy as np
from scipy.linalg import eigh

sys.path.insert(0, "examples")
from product_state_demo import kite  # noqa: E402

from circuit_quantization.product_state import (  # noqa: E402
    _initial, _mode_hamiltonian, _mode_operators, _scf, default_bases,
    normal_mode_transformation,
)

np.set_printoptions(precision=4, suppress=True, linewidth=120)

ROVERCH = {"E_J": 16.83, "E_CJ": 4.82, "E_L": 1.27, "E_CS": 0.072, "epsilon": 0.03, "delta": 0.0}
S_OPT = np.array([[1.0, 0.2354, 0.2306], [0.0, 0.6962, 0.7339], [0.0, -0.5719, 0.629]])
STARTS = ((0.0, 0.0, 0.0), (np.pi, 0.0, 0.0), (0.0, 0.0, np.pi), (np.pi, 0.0, np.pi), None)
BASIS_KWARGS = dict(charge_cutoff=16, max_points=76)
E_EXACT = np.array([-9.524003, -9.510533, -8.453798])  # 190k-dimensional LOBPCG solve


def describe(label, h):
    names = ["phi_1", "phi_2", "phi_3"]
    print(f"\n=== {label} frame: types {h.types} ===")
    print("E_C =\n", h.E_C)
    print("E_L =\n", h.E_L)
    print("b   =", h.flux_linear, "  n_g =", h.n_g)
    print("H = 4 n^T E_C n + 1/2 phi^T E_L phi + b.phi")
    for E, w, d in h.cosines:
        arg = " ".join(f"{w[i]:+.4f} {names[i]}" for i in range(h.size) if abs(w[i]) > 1e-9)
        print(f"    - {E:.4f} cos({arg} {d:+.4f})")


def configurations(h, bases):
    """Distinct mean-field fixed points reached from ``STARTS``, lowest first."""
    runs = sorted((_scf(h, bases, 1e-11, 300, _initial(h, phi0)) for phi0 in STARTS),
                  key=lambda run: run[0])
    distinct = []
    for energy, states, ex, _, _ in runs:
        def far(other):
            dphi = np.max(np.abs(ex["phi"] - other[2]["phi"]))
            dX = max(np.max(np.abs(a - b)) for a, b in zip(ex["X"], other[2]["X"]))
            return max(dphi, dX) > 1e-3
        if all(far(other) for other in distinct):
            distinct.append((energy, states, ex))
    return distinct


def bare_leaf(h, ops, i):
    op = ops[i]
    H = 4 * h.E_C[i, i] * op["n2"]
    if op["phi"] is not None:
        H = H + 0.5 * h.E_L[i, i] * op["phi"] ** 2 + h.flux_linear[i] * op["phi"]
    for k, (E_k, w, d) in enumerate(h.cosines):
        if op["X"][k] is not None and np.sum(np.abs(w) > 1e-9) == 1:
            X = op["X"][k]
            H = H - 0.5 * E_k * (np.exp(1j * d) * X + np.exp(-1j * d) * X.conj().T)
    return H


def leaf_bases(h, bases, configs, m, kind):
    """``m`` orthonormal vectors per mode from the chosen leaf Hamiltonians."""
    ops = _mode_operators(h, bases)
    U = []
    for i in range(h.size):
        if kind == "bare":
            leaves = [bare_leaf(h, ops, i)]
        else:
            leaves = [_mode_hamiltonian(h, ops, ex, i) for _, _, ex in configs]
        columns = [eigh(H, subset_by_index=(0, m - 1))[1] for H in leaves]
        kept = []
        for level in range(m):
            for vecs in columns:
                v = vecs[:, level].astype(complex)
                for _ in range(2):  # Gram-Schmidt twice
                    for u in kept:
                        v -= u * np.vdot(u, v)
                norm = np.linalg.norm(v)
                if norm > 1e-6 and len(kept) < m:
                    kept.append(v / norm)
        U.append(np.column_stack(kept))
    return U


def projected(h, bases, U):
    """The full Hamiltonian on the product of the truncated bases ``U``."""
    ops = _mode_operators(h, bases)
    N, dims = h.size, [u.shape[1] for u in U]

    def P(i, A):
        return U[i].conj().T @ A @ U[i]

    def embed(factors):
        out = np.ones((1, 1), dtype=complex)
        for i in range(N):
            out = np.kron(out, factors.get(i, np.eye(dims[i])))
        return out

    H = h.constant * embed({})
    for i in range(N):
        H += 4 * h.E_C[i, i] * embed({i: P(i, ops[i]["n2"])})
        if ops[i]["phi"] is not None:
            H += 0.5 * h.E_L[i, i] * embed({i: P(i, ops[i]["phi"] ** 2)})
            H += h.flux_linear[i] * embed({i: P(i, ops[i]["phi"])})
        for j in range(i + 1, N):
            H += 8 * h.E_C[i, j] * embed({i: P(i, ops[i]["n"]), j: P(j, ops[j]["n"])})
            if ops[i]["phi"] is not None and ops[j]["phi"] is not None:
                H += h.E_L[i, j] * embed({i: P(i, ops[i]["phi"]), j: P(j, ops[j]["phi"])})
    for k, (E_k, w, d) in enumerate(h.cosines):
        X = embed({i: P(i, ops[i]["X"][k]) for i in range(N) if ops[i]["X"][k] is not None})
        H -= 0.5 * E_k * (np.exp(1j * d) * X + np.exp(-1j * d) * X.conj().T)
    return H


if __name__ == "__main__":
    ham = kite(ROVERCH)
    frames = {"reference": np.eye(3), "normal modes": normal_mode_transformation(ham),
              "optimized": S_OPT}
    for label, S in frames.items():
        h = ham.transformed(S)
        describe(label, h)
        bases = default_bases(h, **BASIS_KWARGS)
        configs = configurations(h, bases)
        print(f"--- {len(configs)} mean-field configurations; bases {[b.size for b in bases]}")
        for energy, _, ex in configs:
            print(f"    E={energy:.5f}  <phi>={ex['phi']}")
            for k, X in enumerate(ex["X"]):
                print(f"        cosine {k}: |<e^(i w phi)>|={np.abs(X)}  arg/pi={np.angle(X) / np.pi}")
        for kind in ("bare", "mean-field", "two-well"):
            for m in (2, 4, 8, 12, 16):
                t0 = time.time()
                U = leaf_bases(h, bases, configs[:1 if kind == "mean-field" else 2], m, kind)
                H = projected(h, bases, U)
                vals = eigh(H, subset_by_index=(0, 2), eigvals_only=True)
                err = vals - E_EXACT
                print(f"    {kind:10s} m={m:2d} dim={H.shape[0]:5d}  E0 err={err[0]:9.2e}  "
                      f"E1 err={err[1]:9.2e}  splitting={vals[1] - vals[0]:9.2e}  "
                      f"E2 err={err[2]:9.2e}  [{time.time() - t0:.1f}s]", flush=True)
