import numpy as np
# This package is for symmetric matrix eigenvalues
from numpy.linalg import eigvalsh

def compute_beta_i(H_f, H_g, tol=1e-6, max_iter=100):
    
    """
    Finds the smallest beta such that H_f + beta * H_g is positive semidefinite.
    Allows beta to be negative.

    Parameters:
    - H_f: Hessian of f at a single point (n x n symmetric numpy array)
    - H_g: Hessian of g at the same point (n x n symmetric numpy array)
    - tol: tolerance for eigenvalue being >= 0
    - max_iter: number of steps in bisection

    Returns:
    - beta_i: minimum beta such that H_f + beta * H_g is PSD
    """

    # Step 1: Initial bracket for β (searching negative direction)
    beta_low = -1.0
    beta_high = 0.0

    # Step 2: Expand lower bound until H_f + beta * H_g is NOT PSD
    while True:
        H = H_f + beta_low * H_g
        min_eig = np.min(eigvalsh(H))
        if min_eig < 0:
            break
        beta_low *= 2  # go more negative

    # Step 3: Bisection search to find the largest negative β such that H is PSD
    for _ in range(max_iter):
        beta_mid = (beta_low + beta_high) / 2
        H = H_f + beta_mid * H_g
        min_eig = np.min(eigvalsh(H))
        # still PSD; tighten upper bound
        if min_eig >= 0:
            beta_high = beta_mid
        # not PSD; tighten lower bound
        else:
            beta_low = beta_mid
        if beta_high - beta_low < tol:
            break
            
    return beta_high

def compute_beta_star(hessians_f, hessians_g):
    """
    Computes the overall β* across all sample points.

    Parameters:
    - hessians_f: list of Hessians of f at sample points
    - hessians_g: list of Hessians of g at sample points

    Returns:
    - beta_star: the maximum of all local β_i values
    """

    beta_list = []

    for H_f, H_g in zip(hessians_f, hessians_g):
        beta_i = compute_beta_i(H_f, H_g)
        beta_list.append(beta_i)

    return max(beta_list)


# ============================
# Example: 2D Hessians at 3 points
# ============================

Hf_list = [
    np.array([[2.0, 0.0], [0.0, 2.0]]),      # Positive definite
    np.array([[1.0, 0.0], [0.0, 0.5]]),      # Positive definite
    np.array([[0.8, 0.0], [0.0, 0.4]])       # Positive definite
]

Hg_list = [
    np.array([[1.0, 0.0], [0.0, 1.0]]),      # Positive definite
    np.array([[1.0, 0.5], [0.5, 1.0]]),      # Positive definite
    np.array([[2.0, 0.0], [0.0, 2.0]])       # Positive definite
]

# Compute β*
beta_star = compute_beta_star(Hf_list, Hg_list)
print(f"Minimum β*: {beta_star:.6f}")
