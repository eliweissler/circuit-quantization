import numpy as np

# Let's simulate a CPB Hamiltonian H = 4 * Ec * (n - ng)^2 - Ej * cos(phi)
Ec = 1.0
Ej = 1.0

# If Q has eigenvalues in Qg + 2e Z, then n = Q / 2e has eigenvalues ng + Z
# So the matrix for (n - ng)^2 is just diag(Z^2)
N_states = 11
Z_vals = np.arange(-5, 6)
n_eigenvals = Z_vals + 0.3  # suppose ng = 0.3, and spectrum is shifted by ng

# If eigenvalues are shifted by ng, then n - ng is just Z_vals
kinetic_terms = 4 * Ec * (Z_vals)**2
H = np.diag(kinetic_terms)

# The cos(phi) operator shifts n by +/- 1
for i in range(N_states):
    if i > 0:
        H[i, i-1] -= Ej / 2
    if i < N_states - 1:
        H[i, i+1] -= Ej / 2

evals_shifted = np.linalg.eigvalsh(H)

# Now what if n had integer eigenvalues (representation sector 0)?
n_eigenvals_0 = Z_vals 
kinetic_terms_0 = 4 * Ec * (n_eigenvals_0 - 0.3)**2
H_0 = np.diag(kinetic_terms_0)
for i in range(N_states):
    if i > 0:
        H_0[i, i-1] -= Ej / 2
    if i < N_states - 1:
        H_0[i, i+1] -= Ej / 2
evals_0 = np.linalg.eigvalsh(H_0)

print("Ground state energy if Q is shifted by Qg (Eq 80):", evals_shifted[0])
print("Ground state energy if Q is integers (standard CPB):", evals_0[0])

