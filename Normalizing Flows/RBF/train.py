import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from .model import RadialBasisFlow


def train(
    X: torch.Tensor,
    model: RadialBasisFlow,
    *,
    log_rho_exact: callable = None,
    n_steps: int = 1000,
    print_every: int = 100,
    visualize: bool = False,
    n_viz_samples: int = 1000,
    use_convexity_check: bool = False,
    convexity_check_samples: int = None,
    beta_reduction_factor: float = 2.0,
    beta_min_floor: float = 0.01,
):
    """
    Train a RadialBasisFlow model.
    
    Args:
        X: Training data of shape (n_samples, n_dim)
        model: The RadialBasisFlow model
        log_rho_exact: Optional function to compute exact log density for KL monitoring
        n_steps: Number of training steps
        print_every: Print progress every n steps
        visualize: Whether to visualize during training
        n_viz_samples: Number of samples for visualization
        use_convexity_check: If True, use dual LP to check convexity when beta < 0
        convexity_check_samples: Samples to use for convexity check (None = all)
        beta_reduction_factor: Factor to divide beta by when convexity fails (default 1.5)
        beta_min_floor: Minimum |beta| when reducing; ensures model can still learn (default 0.01)
    """
    if isinstance(X, np.ndarray):
        X = torch.tensor(X, dtype=torch.float32, device=model.device)
    else:
        X = X.to(model.device)
    
    if X.ndim == 1:
        X = X.unsqueeze(1)
    
    print("=" * 80)
    print("Training Radial Basis Flow")
    print("=" * 80)
    print(f"Data shape: {X.shape}")
    print(f"Number of steps: {n_steps}")
    print(f"n_p: {model.n_p}, epsilon: {model.epsilon}")
    if use_convexity_check:
        print("Convexity check: ENABLED (dual LP)")
    print("=" * 80)
    
    # Fit the model
    model.fit(
        X,
        log_rho_exact=log_rho_exact,
        n_steps=n_steps,
        use_convexity_check=use_convexity_check,
        convexity_check_samples=convexity_check_samples,
        beta_reduction_factor=beta_reduction_factor,
        beta_min_floor=beta_min_floor,
    )
    
    if visualize and len(model.maps) > 0:
        visualize_transformations(model, X, n_viz_samples)
    
    print("\nTraining completed!")
    return model


def visualize_transformations(
    model: RadialBasisFlow,
    X: torch.Tensor,
    n_samples: int = 1000,
):
    """
    Visualize the learned transformation.
    
    Args:
        model: Trained RadialBasisFlow model
        X: Original training data
        n_samples: Number of samples to visualize
    """
    model.eval()
    
    # Sample from base distribution
    flat_dim = X.shape[1]
    base = torch.distributions.MultivariateNormal(
        torch.zeros(flat_dim, device=model.device),
        torch.eye(flat_dim, device=model.device)
    )
    S_viz = base.sample((n_samples,))
    
    with torch.no_grad():
        # Transform base samples back to data space (approximate)
        # Note: This is a simplified visualization - proper inverse would require optimization
        try:
            # For visualization, we'll show the forward transform of data
            y, _ = model.forward(X[:n_samples])
            y_np = y.detach().cpu().numpy()
            X_np = X[:n_samples].detach().cpu().numpy()
            
            plt.figure(figsize=(12, 5))
            
            plt.subplot(1, 2, 1)
            plt.scatter(X_np[:, 0], X_np[:, 1], alpha=0.3, label='Original Data X')
            plt.title("Original Data")
            plt.xlabel("X1")
            plt.ylabel("X2")
            plt.legend()
            plt.axis('equal')
            plt.grid(True)
            
            plt.subplot(1, 2, 2)
            plt.scatter(y_np[:, 0], y_np[:, 1], alpha=0.3, label='Transformed to Base')
            plt.title("Transformed to Base Distribution")
            plt.xlabel("Y1")
            plt.ylabel("Y2")
            plt.legend()
            plt.axis('equal')
            plt.grid(True)
            
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Visualization error: {e}")
    
    model.train()


def train_with_validation(
    X_train: torch.Tensor,
    X_val: torch.Tensor,
    model: RadialBasisFlow,
    *,
    log_rho_exact: callable = None,
    n_steps: int = 1000,
    print_every: int = 100,
    val_every: int = 100,
):
    """
    Train with validation monitoring.
    
    Args:
        X_train: Training data
        X_val: Validation data
        model: The RadialBasisFlow model
        log_rho_exact: Optional function to compute exact log density
        n_steps: Number of training steps
        print_every: Print progress every n steps
        val_every: Validate every n steps
    """
    if isinstance(X_train, np.ndarray):
        X_train = torch.tensor(X_train, dtype=torch.float32, device=model.device)
    if isinstance(X_val, np.ndarray):
        X_val = torch.tensor(X_val, dtype=torch.float32, device=model.device)
    
    print("=" * 80)
    print("Training Radial Basis Flow with Validation")
    print("=" * 80)
    
    # Store validation losses
    val_losses = []
    
    # Fit with validation monitoring
    m, n = X_train.shape
    tol = 1e-9
    
    # Preconditioning
    mean = torch.mean(X_train, dim=0)
    z = X_train - mean
    std = torch.sqrt(torch.mean(torch.sum(z**2, dim=1)) / n)
    z = z / std
    z_init = z.clone()
    
    model.preconditioning = {'mean': mean, 'std': std}
    model.maps = []
    
    for i in range(n_steps):
        # Training step (same as in model.fit)
        if torch.rand(1).item() > 0.5:
            idx = torch.randint(0, m, (1,)).item()
            x0 = z[idx].clone()
        else:
            x0 = torch.randn(n, device=model.device)
        
        alpha = model._calculate_alpha(x0, n, m)
        r = torch.norm(z - x0, dim=1)
        f = model._radial_f(r, alpha)
        f_p_over_r = model._radial_f_prime_over_r(r, alpha)
        v = z_init - x0
        v_norm_sq = torch.sum(v**2, dim=1)
        z_dot_v = torch.sum(z * v, dim=1)
        diff_dot_v = torch.sum((z - x0) * v, dim=1)
        C = f_p_over_r * diff_dot_v
        G = -torch.sum(f * z_dot_v) + torch.sum(C)
        H = -torch.sum(f**2 * v_norm_sq) - torch.sum(C**2)
        
        if torch.abs(H) < tol:
            continue
        
        beta = -G / H
        beta = torch.clamp(beta, -model.epsilon, model.epsilon)
        z = z + beta * f.unsqueeze(1) * v
        
        model.maps.append({
            'x0': x0.detach().cpu(),
            'alpha': alpha.detach().cpu(),
            'beta': beta.detach().cpu().item()
        })
        
        # Validation
        if (i+1) % val_every == 0:
            with torch.no_grad():
                val_log_prob = model.log_prob(X_val)
                val_loss = -val_log_prob.mean().item()
                val_losses.append(val_loss)
                
                if (i+1) % print_every == 0:
                    print(f"Step {i+1}/{n_steps}, Validation NLL: {val_loss:.6f}")
        
        # KL monitoring
        if log_rho_exact is not None and (i+1) % print_every == 0:
            kl = model._estimate_forward_KL(X_train, log_rho_exact)
            print(f"Step {i+1}/{n_steps}, KL: {kl:.4f}")
            model.kl_history.append(kl)
    
    return model, val_losses
