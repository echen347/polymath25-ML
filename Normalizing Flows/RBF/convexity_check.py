"""
Convexity check via dual linear program for RBF flows.

Based on the formulation from "Checking Convexity for Adaptive β":
- Sample points (z_i, U(z_i)) are from a convex function iff for each i there exists
  a subgradient g_i such that:
  U(z_j) >= U(z_i) + g_i^T (z_j - z_i) for all j

- By Farkas' Lemma, this fails for point i iff there exist α_j >= 0 such that:
  1. sum_j α_j (z_j - z_i) = 0
  2. sum_j α_j (U(z_j) - U(z_i)) < 0

- Dual LP for each i: minimize sum_j α_j (U(z_j) - U(z_i))
  subject to: α_j >= 0, sum_j α_j (z_j - z_i) = 0

- If the LP is unbounded (optimal value = -inf), convexity fails at point i.
"""

import numpy as np
from typing import Tuple, Optional
from scipy.optimize import linprog


def check_convexity_dual_lp(
    points: np.ndarray,
    values: np.ndarray,
    point_idx: int,
    tol: float = 1e-10,
) -> Tuple[bool, float]:
    """
    Check if point point_idx has a supporting hyperplane (convexity at that point)
    using the dual LP formulation.

    For point i, the primal asks: does there exist subgradient g such that
    U(z_j) >= U(z_i) + g^T(z_j - z_i) for all j?

    The dual LP: minimize sum_j α_j (U(z_j) - U(z_i))
    subject to: α_j >= 0, sum_j α_j (z_j - z_i) = 0

    If unbounded (opt < -tol), convexity fails.

    Args:
        points: Sample points, shape (m, d)
        values: Potential/function values at points, shape (m,)
        point_idx: Index i of the point to check
        tol: Tolerance for numerical comparisons

    Returns:
        (is_convex, optimal_value): True if convexity holds at this point
    """
    m, d = points.shape
    i = point_idx

    # Differences: z_j - z_i for all j
    delta_z = points - points[i]  # (m, d)
    delta_u = values - values[i]  # (m,)

    # LP: minimize c^T @ alpha where c = delta_u
    # subject to: alpha >= 0
    #             sum_j alpha_j * (z_j - z_i)_k = 0 for each dimension k
    # Add sum(alpha) = 1 to bound the LP (otherwise unbounded when min < 0)
    # If min < -tol, convexity fails (Farkas: primal infeasible)

    c = delta_u
    A_eq = delta_z.T  # (d, m) - each row: sum_j alpha_j * (z_j - z_i)_k = 0
    b_eq = np.zeros(d)

    A_eq_full = np.vstack([A_eq, np.ones((1, m))])
    b_eq_full = np.concatenate([b_eq, [1.0]])

    result = linprog(
        c,
        A_eq=A_eq_full,
        b_eq=b_eq_full,
        bounds=(0, None),
        method='highs',
    )

    if not result.success:
        if result.message and 'infeasible' in str(result.message).lower():
            # Infeasible: equality constraints inconsistent - convexity may hold
            return True, np.inf
        return False, result.fun if result.fun is not None else np.nan

    opt_val = result.fun
    # If optimal value < -tol, convexity fails (Farkas: primal infeasible)
    is_convex = opt_val >= -tol
    return is_convex, opt_val


def check_convexity_at_samples(
    points: np.ndarray,
    values: np.ndarray,
    tol: float = 1e-10,
    verbose: bool = False,
) -> Tuple[bool, list]:
    """
    Check if the sample points (z_i, U(z_i)) can be interpolated by a convex function.
    Uses the dual LP formulation for efficiency with many samples.

    Args:
        points: Sample points, shape (m, d)
        values: Potential/function values at points, shape (m,)
        tol: Tolerance for numerical comparisons
        verbose: If True, print which points fail

    Returns:
        (is_convex, failed_indices): True if convex at all sample points
    """
    m = points.shape[0]
    failed_indices = []

    for i in range(m):
        is_convex, opt_val = check_convexity_dual_lp(points, values, i, tol=tol)
        if not is_convex:
            failed_indices.append((i, opt_val))
            if verbose:
                print(f"  Convexity fails at point {i} (LP opt value: {opt_val:.6e})")

    return len(failed_indices) == 0, failed_indices


def reduce_beta_for_convexity(
    flow,
    x_data: np.ndarray,
    beta: float,
    beta_reduction_factor: float = 2.0,
    max_reductions: int = 10,
    n_check_samples: Optional[int] = None,
    tol: float = 1e-8,
    beta_min_floor: Optional[float] = None,
) -> Tuple[float, bool]:
    """
    Reduce beta until convexity holds at sample points (data space with U = -log_prob).

    Uses the dual LP formulation: sample points are (x_i, U_i) where U_i = -log_prob(x_i).
    After adding a new map with beta, we temporarily add it to the flow, compute
    log_prob at data points, and check if (x, -log_prob(x)) forms a convex interpolation.

    Args:
        flow: RadialBasisFlow model with the new map already added (will be reverted)
        x_data: Data points in original space (m, d)
        beta: Proposed beta value (the new map's beta)
        beta_reduction_factor: Factor to divide beta by when convexity fails (default 1.5,
            less aggressive than 2.0 to allow more training)
        max_reductions: Maximum number of beta reductions to try
        n_check_samples: If set, use subset of samples for check (for speed)
        tol: Tolerance for LP optimal value
        beta_min_floor: Minimum |beta| to allow; never reduce below this magnitude.
            Ensures the model can still learn when convexity is hard to achieve.
            E.g. 0.01 means we use at least beta=-0.01. None = no floor.

    Returns:
        (adjusted_beta, convexity_achieved)
    """
    import torch

    m = x_data.shape[0]
    if n_check_samples is not None and m > n_check_samples:
        indices = np.random.choice(m, n_check_samples, replace=False)
        x_check = x_data[indices]
    else:
        x_check = x_data

    # Get the last map (the one we're checking) and its index
    if len(flow.maps) == 0:
        return beta, True

    current_beta = beta
    last_map_idx = len(flow.maps) - 1

    for reduction in range(max_reductions):
        # Update the beta in the last map
        flow.maps[last_map_idx] = {
            **flow.maps[last_map_idx],
            'beta': current_beta
        }

        # Compute -log_prob at data points (potential U)
        with torch.no_grad():
            x_t = torch.tensor(x_check, dtype=torch.float32, device=flow.device)
            log_probs = flow.log_prob(x_t).cpu().numpy()
        U = -log_probs

        # Check convexity at (x_check, U)
        is_convex, failed = check_convexity_at_samples(x_check, U, tol=tol, verbose=False)
        if is_convex:
            return current_beta, True

        current_beta = current_beta / beta_reduction_factor
        # Enforce minimum floor so we don't zero out the update
        if beta_min_floor is not None and abs(current_beta) < beta_min_floor:
            current_beta = -beta_min_floor if beta < 0 else beta_min_floor
        if abs(current_beta) < 1e-12:
            # Restore original beta and give up
            flow.maps[last_map_idx] = {**flow.maps[last_map_idx], 'beta': beta}
            return beta, False

    # Exhausted reductions: use current (most reduced) beta, with floor if set
    if beta_min_floor is not None and abs(current_beta) < beta_min_floor:
        current_beta = -beta_min_floor if beta < 0 else beta_min_floor
    return current_beta, False
