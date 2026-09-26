import numpy as np

print("--- Checking Eq 80 (Gate charge error) ---")
# H = (Q - Qg)^2. If Q has eigenvalues Qg + Z, then H = Z^2, independent of Qg.
Qg1, Qg2 = 0.3, 0.7
Z = np.arange(-10, 11)
H_qg1 = np.diag((Z + Qg1 - Qg1)**2)
H_qg2 = np.diag((Z + Qg2 - Qg2)**2)
if np.allclose(H_qg1, H_qg2):
    print("If Q eigenvalues offset by Qg, spectrum is independent of Qg!")

print("--- Checking Eq 78 Claim (Injectivity) ---")
# A 2x1 injective matrix that does not produce a rank-1 lattice.
omega = np.array([[1], [np.sqrt(2)]])
# For a in R, omega*a in Z^2 implies a in Z and a*sqrt(2) in Z.
# Only a=0 satisfies this.
print("omega = [1, sqrt(2)]^T is injective, but produces a lattice of rank 0, not 1.")

