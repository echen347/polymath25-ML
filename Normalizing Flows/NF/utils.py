import torch
import numpy as np

from scipy.optimize import newton
from scipy.optimize import root


np.random.seed(42)
torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

base = torch.distributions.MultivariateNormal(torch.zeros(2, device=device), torch.eye(2, device=device))
S = base.sample((1000,)).cpu().numpy()

# Phi2 = X^4 + X^2 Y^2 + Y^4 + (X + Y) ^ 4 + (X - Y) ^ 4
# T2 = <4 * X^3 + 2 * X * Y^2 + 4 * (X + Y)^3 + 4 * (X - Y)^3, 2 * X^2 * Y + 4 * Y^3 + 4 * (X + Y)^3 - 4 * (X - Y)^3> 


def T(xy):
    x = xy[:, 0]
    y = xy[:, 1]
    Tx = 4 * x**3 + 2 * x * y**2 + 4 * (x + y)**3 + 4 * (x - y)**3
    Ty = 2 * x**2 * y + 4 * y**3 + 4 * (x + y)**3 - 4 * (x - y)**3
    return np.stack([Tx, Ty], axis=1)

def inv_T(tx_ty):
    if tx_ty.ndim == 1:
        tx_ty = tx_ty.reshape(1, -1)
    
    results = []
    for i in range(len(tx_ty)):
        x0 = tx_ty[i].copy()  # initial guess
        def f(xy):
            xy_2d = xy.reshape(1, -1)
            return (T(xy_2d) - tx_ty[i]).ravel()
        sol = root(f, x0)
        results.append(sol.x)
    
    return np.array(results)

def T_3d(xyz):
    # Phi3 = X^4 + Y^4 + Z^4 
    # T3 = <4 * X^3, 4 * Y^3, 4 * Z^3>
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]
    Tx = 4 * x**3
    Ty = 4 * y**3
    Tz = 4 * z**3
    return np.stack([Tx, Ty, Tz], axis=1)

def T_inv_3d(tx_ty_tz):
    if tx_ty_tz.ndim == 1:
        tx_ty_tz = tx_ty_tz.reshape(1, -1)
    
    results = []
    for i in range(len(tx_ty_tz)):
        x0 = tx_ty_tz[i].copy()  # initial guess
        def f(xyz):
            xyz_3d = xyz.reshape(1, -1)
            return (T_3d(xyz_3d) - tx_ty_tz[i]).ravel()
        sol = root(f, x0)
        results.append(sol.x)
    
    return np.array(results)
    
def get_data(n_samples: int = 1000, dim: int = 2, T_inv=None) -> np.ndarray:
    base = torch.distributions.MultivariateNormal(torch.zeros(dim, device=device), torch.eye(dim, device=device))
    S = base.sample((n_samples,)).cpu().numpy()
    X = T_inv(S)
    np.savez('./X_data.npz', X=X)
    return S, X


def sample_data(model, S: np.ndarray, n_samples: int = 1000) -> np.ndarray:
    samples = model.reverse(torch.tensor(S, device=device), max_iter=50_000, lr=1.0, tol=1e-12).cpu().detach().numpy()
    return samples