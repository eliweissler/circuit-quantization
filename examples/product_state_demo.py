"""Product-state coordinates on three circuits, against exact solves.

Each case compares the product ansatz in the reference (node or theta/phi/zeta)
frame, in the relaxed-family normal modes, and in the optimized frame of
``product_state_S1``.  Cases: two capacitively coupled fluxoniums; a transmon
capacitively coupled to a fluxonium at half flux; an asymmetric zero-pi.
"""

import time

import numpy as np
import sympy as sym

import circuit_quantization as cq
from circuit_quantization import Branch, Circuit, Loops, load_spec, quantize, to_spec
from circuit_quantization.product_state import (
    NumericHamiltonian, default_bases, exact_ground_state, infidelity,
    normal_mode_transformation, numeric_hamiltonian, product_state, product_state_S1,
    subspace_infidelity,
)


def coupled_E_C(E_C0, coupling):
    """E_C of two equal shunts with a coupling capacitor ``coupling`` times a shunt."""
    C = np.array([[1 + coupling, -coupling], [-coupling, 1 + coupling]]) / E_C0
    return np.linalg.inv(C)


def fluxonium_pair(coupling=0.2, E_L=0.5, E_J=4.0, flux=np.pi):
    return NumericHamiltonian(("e", "e"), coupled_E_C(1.0, coupling), np.diag([E_L, E_L]), [0, 0],
                              [0, 0], ((E_J, [1, 0], flux), (E_J, [0, 1], flux)))


def transmon_fluxonium(coupling=0.2, E_Jt=10.0, n_g=0.25, E_L=0.5, E_Jf=4.0, flux=np.pi):
    """Transmon (compact, ``E_J/E_C`` about 12) capacitively coupled to a fluxonium."""
    return NumericHamiltonian(("c", "e"), coupled_E_C(1.0, coupling), np.diag([0.0, E_L]), [0, 0],
                              [n_g, 0], ((E_Jt, [1, 0], 0.0), (E_Jf, [0, 1], flux)))


def asymmetric_zero_pi(values):
    """Zero-pi with distinct junctions, shunts and inductors, quantized by the package."""
    ECJ1, ECJ2, EC1, EC2, EL1, EL2, EJ1, EJ2 = sym.symbols(
        "E_CJ1 E_CJ2 E_C1 E_C2 E_L1 E_L2 E_J1 E_J2", positive=True)
    ext = sym.Symbol("varphi_ext", real=True)
    circuit = Circuit([
        Branch("J1", 1, 2, C=cq.ghz_to_farad(ECJ1), EJ=EJ1),
        Branch("J2", 3, 4, C=cq.ghz_to_farad(ECJ2), EJ=EJ2),
        Branch("L1", 2, 3, L=cq.ghz_to_henry(EL1)),
        Branch("L2", 4, 1, L=cq.ghz_to_henry(EL2)),
        Branch("C1", 1, 3, C=cq.ghz_to_farad(EC1)),
        Branch("C2", 2, 4, C=cq.ghz_to_farad(EC2)),
    ])
    loops = Loops.from_tree(circuit, cq.default_tree(circuit), {"L2": cq.phase_to_weber(ext)})
    spec = load_spec(to_spec(quantize(circuit, loops)))
    values = dict(values)
    values.update({p.name: 0.0 for p in spec.parameters
                   if p.name.startswith("Phi_") or p.name.startswith("n_g")})
    return numeric_hamiltonian(spec, values)


def kite(values):
    """The cos 2 phi transmon (KITE) in the source's (varphi, phi_Sigma, phi_Delta) frame.

    The construction follows ``symmetry_derivations/kite.ipynb``: each arm has two
    ``2 E_L`` inductors around its junction, the arm splits are frozen, the loop
    phase ``pi + delta`` is shared equally over the four inductors, and the area
    disorder scales ``E_J`` and ``C_J`` by ``1 +- epsilon``.
    """
    E_J, E_CJ, E_CS, E_L = sym.symbols("E_J E_CJ E_CS E_L", positive=True)
    eps, delta = sym.Symbol("epsilon", real=True), sym.Symbol("delta", real=True)
    circuit = Circuit([
        Branch("L1", 1, 2, L=cq.ghz_to_henry(2 * E_L)),
        Branch("L2", 1, 3, L=cq.ghz_to_henry(2 * E_L)),
        Branch("CS", 1, 6, C=cq.ghz_to_farad(E_CS)),
        Branch("J1", 2, 4, C=cq.ghz_to_farad(E_CJ) * (1 + eps), EJ=E_J * (1 + eps)),
        Branch("J2", 3, 5, C=cq.ghz_to_farad(E_CJ) * (1 - eps), EJ=E_J * (1 - eps)),
        Branch("L3", 5, 6, L=cq.ghz_to_henry(2 * E_L)),
        Branch("L4", 4, 6, L=cq.ghz_to_henry(2 * E_L)),
    ], ground=6)
    cycles = {"kite": {"L1": 1, "J1": 1, "L4": 1, "L2": -1, "J2": -1, "L3": -1},
              "cb": {"CS": 1, "L1": -1, "J1": -1, "L4": -1}}
    loops = Loops.from_branch_dict(circuit, cycles, {"kite": cq.phase_to_weber(sym.pi + delta),
                                                     "cb": cq.phase_to_weber(0)})
    q, h = sym.Rational(1, 4), sym.Rational(1, 2)
    B = cq.B_from_shares(circuit, loops, {"kite": {"L1": q, "L4": q, "L2": -q, "L3": -q, "CS": h},
                                          "cb": {"CS": 1}})
    Z = sym.Matrix([[1, 0, 0, 0, 0], [1, -h, h, 1, 0], [1, -h, -h, 0, 1],
                    [0, h, -h, 1, 0], [0, h, h, 0, 1]])
    res = quantize(circuit, loops, Z, ("c", "e", "e", "fz", "fz"), B=B)
    spec = load_spec(to_spec(res))
    values = dict(values)
    values.update({p.name: 0.0 for p in spec.parameters if p.name.startswith("n_g")})
    return numeric_hamiltonian(spec, values)


def max_kappa(M, indices):
    block = M[np.ix_(indices, indices)]
    if len(indices) < 2:
        return 0.0
    kappa = block / np.sqrt(np.outer(np.diag(block), np.diag(block)))
    return float(np.max(np.abs(kappa - np.diag(np.diag(kappa)))))


def report(label, ham, S, **basis_kwargs):
    h = ham.transformed(S)
    bases = default_bases(h, **basis_kwargs)
    state = product_state(h, bases=bases, adapt=False)
    t0 = time.time()
    energies, vectors = exact_ground_state(h, bases, k=2, guess=state.vector())
    E_exact, psi = energies[0], vectors[:, 0]
    all_modes = list(range(h.size))
    noncompact = [i for i, kind in enumerate(h.types) if kind != "c"]
    junction = max(np.sum(np.abs(w) > 1e-9) for _, w, _ in h.cosines)
    print(f"  {label:13s} E_prod={state.energy:10.6f}  E_exact={E_exact:10.6f}  "
          f"dE={state.energy - E_exact:8.2e}  infidelity={infidelity(state, psi):8.2e}  "
          f"doublet={subspace_infidelity(state, vectors):8.2e} (E1-E0={energies[1] - energies[0]:.2e})  "
          f"max|kappa_C|={max_kappa(h.E_C, all_modes):.3f}  max|kappa_L|={max_kappa(h.E_L, noncompact):.3f}  "
          f"modes/junction={junction}  [exact {time.time() - t0:.0f}s]")


def run(title, ham, basis_kwargs, **opt_kwargs):
    print(title)
    report("reference", ham, np.eye(ham.size), **basis_kwargs)
    report("normal modes", ham, normal_mode_transformation(ham), **basis_kwargs)
    t0 = time.time()
    S, info = product_state_S1(ham, relaxed=True, regularization=1e-4, **opt_kwargs)
    print(f"  optimizer: {info.result.nfev} evaluations, {time.time() - t0:.0f}s")
    report("optimized", ham, S, **basis_kwargs)
    print("  S_opt =", np.array2string(S.round(4), prefix="  S_opt = "))


if __name__ == "__main__":
    import sys
    only = sys.argv[1] if len(sys.argv) > 1 else None
    if only == "kite":
        roverch = {"E_J": 16.83, "E_CJ": 4.82, "E_L": 1.27, "E_CS": 0.072, "epsilon": 0.03, "delta": 0.0}
        run("KITE (cos 2 phi transmon) at half flux, Roverch parameters " + str(roverch),
            kite(roverch), {"charge_cutoff": 16, "max_points": 76},
            starts=((0.0, 0.0, 0.0), (np.pi, 0.0, 0.0), (0.0, 0.0, np.pi), (np.pi, 0.0, np.pi), None))
        sys.exit()
    for coupling, flux in ((0.2, np.pi), (0.5, 0.0)):
        run(f"Fluxonium pair, C_c/C = {coupling}, flux = {flux / np.pi:.0f} pi",
            fluxonium_pair(coupling=coupling, flux=flux), {"max_points": 140})
    run("Transmon + fluxonium at half flux, C_c/C = 0.2, n_g = 0.25",
        transmon_fluxonium(), {"charge_cutoff": 14, "max_points": 140})
    zp = {"E_CJ1": 5.0, "E_CJ2": 4.0, "E_C1": 1.0, "E_C2": 0.8, "E_L1": 0.5, "E_L2": 0.4,
          "E_J1": 5.0, "E_J2": 4.0, "varphi_ext": 0.0}
    run("Asymmetric zero-pi at zero flux " + str(zp),
        asymmetric_zero_pi(zp), {"charge_cutoff": 8, "max_points": 90},
        starts=((0.0, 0.0, 0.0), (0.0, np.pi, 0.0), None))
    roverch = {"E_J": 16.83, "E_CJ": 4.82, "E_L": 1.27, "E_CS": 0.072, "epsilon": 0.03, "delta": 0.0}
    run("KITE (cos 2 phi transmon) at half flux, Roverch parameters " + str(roverch),
        kite(roverch), {"charge_cutoff": 16, "max_points": 76},
        starts=((0.0, 0.0, 0.0), (np.pi, 0.0, 0.0), (0.0, 0.0, np.pi), (np.pi, 0.0, np.pi), None))
