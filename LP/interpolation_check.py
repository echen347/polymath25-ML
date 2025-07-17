import numpy as np
import cvxpy as cp

# Input: points are array of points, values are array of values where each point evaluates, j is index of points
# Output: true if point j has the tangent plane lays below all the points.  
def check_prime(points, values, j):
    N, d = points.shape
    g = cp.Variable(d)   # the subgradient varaible at point j

    constraints = []
    for i in range(N):
        if i == j:
            continue
        lhs = values[i] - values[j]
        rhs = (points[i] - points[j]) @ g
        constraints.append(lhs >= rhs)

    # Use dummy objective since we only care about feasibility
    prob = cp.Problem(cp.Minimize(0), constraints)
    prob.solve()

    return prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]


# check primal LP each point j. Use this version of check if we have relative small number of points.
# Input: points are array of points, values are array of values where each point evaluates.
# Output: true if points can interpolate the convex function.  
def prime_interpolation_check(points, values):
    return all(check_prime(points, values, j) for j in range(len(points)))


# Input: points are array of points, values are array of values where each point evaluates, j is index of points
# Output: true if point j has the tangent plane lays below all the points. 
def check_dual(points, values, j):
    N = len(points)
    alpha = cp.Variable(N)
    
    constraints = [
        alpha >= 0,
        cp.sum(cp.multiply(alpha[:, None], points - points[j]), axis=0) == 0
    ]
    
    objective = cp.Minimize(cp.sum(cp.multiply(alpha, values - values[j])))

    prob = cp.Problem(objective, constraints)
    prob.solve()

    # If the optimal value is -\infty, dual is feasible, then the primal is infeasible
    return prob.status != cp.UNBOUNDED  # return True means primal is feasible

# check dual LP each point j. Use this version of check if we have large number of points.
# Input: points are array of points, values are array of values where each point evaluates.
# Output: true if points can interpolate the convex function.
def dual_interpolation_check(points, values):
    return all(check_dual(points, values, j) for j in range(len(points)))