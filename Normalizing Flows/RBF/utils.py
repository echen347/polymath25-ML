import torch
import numpy as np
from scipy.optimize import root
from typing import Optional


def estimate_forward_KL(X: np.ndarray, flow, log_rho_exact: callable) -> float:
    """
    Estimate forward KL divergence between exact and learned distributions.
    
    Args:
        X: Data points
        flow: Trained flow model
        log_rho_exact: Function to compute exact log density
        
    Returns:
        Estimated KL divergence
    """
    # Compute log-density under the exact model
    log_ex = log_rho_exact(X)

    # Compute log-density under the flow estimate
    if isinstance(X, np.ndarray):
        X_tensor = torch.tensor(X, dtype=torch.float32, device=flow.device)
    else:
        X_tensor = X
    log_est = flow.log_prob(X_tensor).detach().cpu().numpy()

    # Monte Carlo KL
    kl_mc = np.mean(log_ex - log_est)
    return kl_mc


def sample_exact_ring(n_samples: int, r_mean: float = 1.0, r_std: float = 0.1) -> np.ndarray:
    """
    Sample from a ring distribution (used in the notebook example).
    
    Args:
        n_samples: Number of samples
        r_mean: Mean radius
        r_std: Standard deviation of radius
        
    Returns:
        Samples of shape (n_samples, 2)
    """
    theta = np.random.randn(n_samples)
    r = r_mean + r_std * np.random.randn(n_samples)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.stack([x, y], axis=1)


def log_rho_exact_ring(xy: np.ndarray, r_mean: float = 1.0, r_std: float = 0.1) -> np.ndarray:
    """
    Exact log density for ring distribution.
    
    Args:
        xy: Points of shape (n_samples, 2)
        r_mean: Mean radius
        r_std: Standard deviation of radius
        
    Returns:
        Log densities of shape (n_samples,)
    """
    x = xy[:, 0]
    y = xy[:, 1]
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)

    log_p_theta = -0.5*theta**2 - 0.5*np.log(2*np.pi)
    log_p_r = (
        -0.5*((r - r_mean)/r_std)**2
        - 0.5*np.log(2*np.pi*r_std**2)
    )
    log_jac = -np.log(r)

    return log_p_theta + log_p_r + log_jac


def inverse_transform_optimization(
    flow,
    y: torch.Tensor,
    max_iter: int = 1000,
    lr: float = 0.1,
    tol: float = 1e-6,
    x_init: Optional[torch.Tensor] = None,
    use_lbfgs: bool = False,
) -> torch.Tensor:
    """
    Solve inverse transform problem: find x such that forward(x) = y.
    
    Two approaches:
    1. Direct optimization: minimize ||forward(x) - y||^2 (default, uses Adam)
    2. Potential-based: minimize log_prob(x) - (x · y) (if use_lbfgs=True, uses LBFGS)
    
    Args:
        flow: Trained RadialBasisFlow model
        y: Base distribution samples (in base space, after forward transform)
        max_iter: Maximum iterations
        lr: Learning rate
        tol: Tolerance for convergence
        x_init: Initial guess for x (in data space)
        use_lbfgs: If True, use LBFGS with potential-based optimization (more accurate but slower)
        
    Returns:
        Reconstructed x values (in original data space)
    """
    if x_init is None:
        # Better initialization: account for preconditioning
        # The base samples y are in the transformed space
        # We need to invert the preconditioning to get a better initial guess
        if hasattr(flow, 'preconditioning') and len(flow.preconditioning) > 0:
            # Invert preconditioning: x = y * std + mean
            # This gives us a rough initial guess in data space
            mean = flow.preconditioning['mean']
            std = flow.preconditioning['std']
            x_init = y * std + mean
        else:
            x_init = y.clone()
        x_init = x_init.detach().requires_grad_(True)
    else:
        x_init = x_init.clone().detach().requires_grad_(True)
    
    x = x_init
    
    if use_lbfgs:
        # More principled approach: optimize to match forward transform
        # For RBF flows, we directly optimize the forward transform error
        # This is more stable than trying to use log_prob
        def closure():
            optimizer.zero_grad()
            # Forward transform
            y_pred, _ = flow.forward(x)
            # Loss: ||y_pred - y||^2
            loss = torch.mean((y_pred - y)**2)
            loss.backward()
            return loss
        
        optimizer = torch.optim.LBFGS([x], lr=lr, max_iter=max_iter, 
                                     tolerance_grad=tol, tolerance_change=tol,
                                     line_search_fn="strong_wolfe")
        optimizer.step(closure)
    else:
        # Simpler approach: minimize forward transform error
        optimizer = torch.optim.Adam([x], lr=lr)
        
        best_loss = float('inf')
        patience = 50
        no_improve = 0
        
        for i in range(max_iter):
            optimizer.zero_grad()
            
            # Forward transform
            y_pred, _ = flow.forward(x)
            
            # Loss: ||y_pred - y||^2
            loss = torch.mean((y_pred - y)**2)
            
            loss.backward()
            optimizer.step()
            
            # Check convergence with early stopping
            current_loss = loss.item()
            if current_loss < best_loss:
                best_loss = current_loss
                no_improve = 0
            else:
                no_improve += 1
            
            if current_loss < tol:
                break
            if no_improve >= patience:
                break
    
    return x.detach()


def sample_from_flow(
    flow,
    n_samples: int,
    base_distribution: Optional[torch.distributions.Distribution] = None,
    max_iter: int = 1000,
    lr: float = 0.1,
    use_lbfgs: bool = False,
    batch_size: Optional[int] = None,
) -> torch.Tensor:
    """
    Generate samples from the learned distribution.
    
    The sampling process:
    1. Sample y from base distribution (standard Gaussian)
    2. Solve inverse transform: find x such that forward(x) = y
    3. Return x as samples from the learned distribution
    
    Args:
        flow: Trained RadialBasisFlow model
        n_samples: Number of samples to generate
        base_distribution: Base distribution (default: standard Gaussian)
        max_iter: Maximum iterations for inverse transform
        lr: Learning rate for inverse transform
        use_lbfgs: If True, use LBFGS optimization (more accurate but slower)
        batch_size: If provided, process samples in batches (useful for large n_samples)
        
    Returns:
        Generated samples
    """
    if base_distribution is None:
        flat_dim = len(flow.preconditioning['mean'])
        base_distribution = torch.distributions.MultivariateNormal(
            torch.zeros(flat_dim, device=flow.device),
            torch.eye(flat_dim, device=flow.device)
        )
    
    # Step 1: Sample from base distribution
    y = base_distribution.sample((n_samples,))
    
    # Step 2: Apply inverse transform (solve optimization problem)
    # Process in batches if batch_size is specified (helps with memory and convergence)
    if batch_size is not None and n_samples > batch_size:
        x_list = []
        for i in range(0, n_samples, batch_size):
            y_batch = y[i:i+batch_size]
            x_batch = inverse_transform_optimization(
                flow, y_batch, max_iter=max_iter, lr=lr, use_lbfgs=use_lbfgs
            )
            x_list.append(x_batch)
        x = torch.cat(x_list, dim=0)
    else:
        x = inverse_transform_optimization(flow, y, max_iter=max_iter, lr=lr, use_lbfgs=use_lbfgs)
    
    return x


def compute_log_likelihood(
    flow,
    X: torch.Tensor,
) -> torch.Tensor:
    """
    Compute log likelihood of data under the learned distribution.
    
    Args:
        flow: Trained RadialBasisFlow model
        X: Data points
        
    Returns:
        Log likelihoods
    """
    return flow.log_prob(X)


def compute_negative_log_likelihood(
    flow,
    X: torch.Tensor,
) -> float:
    """
    Compute negative log likelihood (NLL) of data.
    
    Args:
        flow: Trained RadialBasisFlow model
        X: Data points
        
    Returns:
        Average NLL
    """
    log_probs = flow.log_prob(X)
    return -log_probs.mean().item()
