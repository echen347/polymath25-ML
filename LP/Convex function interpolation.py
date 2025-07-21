import numpy as np
import cvxpy as cp


#points are sampled points in R^d, values1 is array of values evaluated at f, values2 is array of values evaluated at f', j is at point j. 
def compute_beta_j_primal(points, values1, values2, j):
    A = np.zeros((N, d + 1))
    b = np.zeros(N)

    for i in range(N):
        A[i, :d] = points[i] - points[j]
        A[i, d] = values2[i] - values2[j]
        b[i] = values1[i] - values1[j]

    c = np.zeros(d + 1)
    c[d] = 1  # Objective: maximize beta

    prob = cp.Problem(cp.Maximize(c @ g), [A @ g <= b])
    prob.solve()

    return prob.value  # This is beta_j


def compute_beta_star_primal(points, values1, values2):
    N = points.shape[0]
    beta_array = np.zeros(N)

    for j in range(N):
        beta_array[j] = compute_beta_j_primal(points, values1, values2, j)

    beta_star = np.min(beta_array)
    return beta_star



def compute_beta_j_dual(points, values1, values2, j):
    N, d = points.shape

    delta_x = points - points[j]         # shape (N, d)
    delta_y = values1 - values1[j]       # shape (N,)
    delta_yprime = values2 - values2[j]  # shape (N,)

    lam = cp.Variable(N, nonneg=True)

    # Constraints:
    constraints = []

    # sum_i λ_i * (x_i - x_j) = 0
    for k in range(d):
        constraints.append(cp.sum(cp.multiply(lam, delta_x[:, k])) == 0)

    # sum_i λ_i * (y'_i - y'_j) = 1
    constraints.append(cp.sum(cp.multiply(lam, delta_yprime)) == 1)

    # Objective: minimize sum_i λ_i * (y_i - y_j)
    objective = cp.Minimize(cp.sum(cp.multiply(lam, delta_y)))

    prob = cp.Problem(objective, constraints)
    prob.solve()

    return prob.value

def compute_beta_star_dual(points, values1, values2):
    N = points.shape[0]
    beta_array = np.zeros(N)

    for j in range(N):
        beta_array[j] = compute_beta_j_dual(points, values1, values2, j)

    beta_star = np.min(beta_array)
    return beta_star
