import cvxpy as cp
import numpy as np


n = 1000
d = 10
x = np.random.randn(n,d)
y = np.apply_along_axis(lambda l: np.dot(l,l), 1, x)  #This is a convex function, y = ||x||^2
yprime = np.apply_along_axis(lambda l: np.sum(np.abs(l)), 1, x+1) #This is another convex function, y = ||x+1||_{L_1}. Here x+1 can be interpreted as adding 1 componentwise.
A = np.zeros((n,n,d))
b = np.zeros((n,n))
bprime = np.zeros((n,n))
betaarr = np.zeros(n)
for i in range(n):
    for j in range(n):
        A[i,j] = x[j]-x[i]
        b[i,j] = y[j]-y[i]
        bprime[i,j] = yprime[j]-yprime[i]
A = np.insert(A,d,bprime,axis=2)
for i in range(n):
    x = cp.Variable(d+1)
    c = np.zeros(d+1)
    c[d] = 1
    prob = cp.Problem(cp.Maximize(c.T @ x),[A[i] @ x <= b[i]])
    prob.solve()
    betaarr[i] = prob.value
beta = np.min(betaarr)
print(beta)
