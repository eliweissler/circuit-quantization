import numpy as np

# Suppose Nc = 1, NJ = 2
# omega^T = [1, 1]^T
omega_T = np.array([[1], [1]])
# P = { a | omega^T a = Phi0 * [k1, k2] }
# a = Phi0 * k1 AND a = Phi0 * k2 => k1 = k2
# So a = Phi0 * k, for any integer k
# P = Phi0 * Z

# Dual lattice P*: q * a in h Z => q * Phi0 * k = h * m => q = (h / Phi0) * m = 2e * m
# So P* = 2e * Z

# What is 2e * omega * Z^NJ?
# omega = [1, 1]
# 2e * omega * [m1, m2]^T = 2e * (m1 + m2)
# Since m1, m2 are any integers, m1+m2 is any integer m.
# So 2e * omega * Z^NJ = 2e * Z
# They match!

print("Dual lattice matches.")
