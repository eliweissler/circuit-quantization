import sympy as sp

# Suppose a 1D compact mode. H = (Q - Qg)^2 / 2C - EJ cos(Phi)
# Quantum mechanically, Q is the momentum operator -i \hbar \partial_\Phi.
# If Phi is periodic in [0, 2pi], then the boundary condition is psi(Phi + 2pi) = e^{i theta} psi(Phi)
# The eigenvalues of Q are \hbar (k + theta / 2pi) = 2e (k + n_g_offset)
# If the operator Q has eigenvalues offset by Qg, then (Q - Qg) has eigenvalues 2e * k.
# Then H = (2e*k)^2 / 2C - EJ cos(Phi), which is independent of Qg!
# This contradicts the known physics of the Cooper pair box, where the energy levels depend on the gate charge.

print("If Q has eigenvalues offset by Qg, then Q - Qg is independent of Qg.")
