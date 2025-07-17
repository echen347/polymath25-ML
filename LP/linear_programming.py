import cvxpy as cp
import numpy as np

# Row value
n = 1000
# Column value
d = 10
# x_i is the i-1th row in the x matrix
x = np.random.uniform(low=-2, high=2, size=(n, 2))

# ALTERNATIVE DISTRIBUTIONS TO GENERATE THE DATA WITH:
# Laplace (heavy-tailed):
# x = np.random.laplace(loc=0, scale=1, size=(n, 2))
# Exponential (positive quadrant only):
# x = np.random.exponential(scale=1.0, size=(n, 2))
# Beta (skewed bounded in [0,1]):
# x = np.random.beta(a=2, b=5, size=(n, 2))
# Binomial (discrete integers):
# x = np.random.binomial(n=10, p=0.5, size=(n, 2))
# Chi-squared (right-skewed):
# x = np.random.chisquare(df=3, size=(n, 2))
# Uniform on a unit disk (polar coords):
# x = np.random.randn(n, 2); x /= np.linalg.norm(x, axis=1, keepdims=True)

# This is a convex function, y = ||x||^2
y = np.apply_along_axis(lambda l: np.dot(l,l), 1, x)
# This is another convex function, l1_norm(x+1)
yprime = np.apply_along_axis(lambda l: np.sum(np.abs(l)), 1, x+1)
# Second column has not been appended yet so that's why we have n x d instead of n x d+1
A = np.zeros((n,n,d))
b = np.zeros((n,n))
# Used to build the rest of matrix A
bprime = np.zeros((n,n))
# Column vector of Beta_i
betaarr = np.zeros(n)

# Building the matrices here
for i in range(n):
    for j in range(n):
        A[i,j] = x[j]-x[i]
        b[i,j] = y[j]-y[i]
        bprime[i,j] = yprime[j]-yprime[i]
A = np.insert(A,d,bprime,axis=2)

for i in range(n):
    # You define a CVXPY optimization variable x that bundles the subgradient and the scalar (d+1)
    x = cp.Variable(d+1)
    # We define the objective vector c to be all zeros except for a 1 in the last position.
    # This ensures that the linear expression c.T @ x simply extracts the last entry of x, which corresponds to the scalar β we are trying to maximize.
    c = np.zeros(d+1)
    # Sets the last entry of the vector c to 1
    c[d] = 1
    prob = cp.Problem(cp.Maximize(c.T @ x),[A[i] @ x <= b[i]])
    prob.solve()
    betaarr[i] = prob.value
    
beta = np.min(betaarr)
print(beta)
