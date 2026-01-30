import torch
import torch.nn as nn
import numpy as np
from scipy.special import erf, gamma
from typing import Optional, Dict, List, Tuple
import warnings


class RadialBasisFlow(nn.Module):
    """
    Radial Basis Function Normalizing Flow based on Tabak-Turner paper.
    Modified implementation where the displacement is anchored to the initial point.
    Map: z_new = z + beta * f(|z-x0|) * (z_init - x0)
    
    Converted to PyTorch to match the NF folder structure.
    """

    def __init__(self, n_p: int = 50, epsilon: float = 0.5, device: Optional[torch.device] = None):
        super().__init__()
        self.n_p = n_p
        self.epsilon = epsilon
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # These will be set during fit()
        self.preconditioning: Dict[str, torch.Tensor] = {}
        self.maps: List[Dict[str, torch.Tensor]] = []
        self.kl_history: List[float] = []
        
    def _volume_n_ball(self, n: int) -> float:
        """Volume of n-dimensional ball"""
        return np.pi**(n/2) / gamma(n/2 + 1)

    def _calculate_alpha(self, x0: torch.Tensor, n: int, m: int) -> torch.Tensor:
        """Calculate the scale parameter alpha"""
        omega_n = self._volume_n_ball(n)
        term = (omega_n**-1 * self.n_p / m)**(1/n)
        x0_norm_sq = torch.sum(x0**2).item()
        alpha = (2 * np.pi)**0.5 * term * np.exp(x0_norm_sq / (2*n))
        return torch.tensor(alpha, dtype=torch.float32, device=self.device)

    def _radial_f(self, r: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        """Radial function f(r) = erf(r/alpha) / r"""
        f_vals = torch.zeros_like(r)
        nonzero_r = r > 1e-9
        zero_r = ~nonzero_r

        r_scaled = r[nonzero_r] / alpha
        f_vals[nonzero_r] = torch.erf(r_scaled) / r[nonzero_r]
        f_vals[zero_r] = 2.0 / (alpha * np.sqrt(np.pi))
        return f_vals

    def _radial_f_prime_over_r(self, r: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        """
        Calculates f'(r) / r safely.
        Needed for the Jacobian determinant of the anchored map.
        """
        val = torch.zeros_like(r)
        nonzero_r = r > 1e-9
        zero_r = ~nonzero_r
        
        # Formula: ( (2u/sqrt(pi))*exp(-u^2) - erf(u) ) / r^3
        # where u = r / alpha
        r_nz = r[nonzero_r]
        u = r_nz / alpha
        
        term1 = (2.0 * u / np.sqrt(np.pi)) * torch.exp(-u**2)
        term2 = torch.erf(u)
        
        val[nonzero_r] = (term1 - term2) / (r_nz**3)
        
        # Limit as r->0 is -4 / (3 * sqrt(pi) * alpha^3)
        val[zero_r] = -4.0 / (3.0 * np.sqrt(np.pi) * alpha**3)
        
        return val

    def fit(
        self,
        x: torch.Tensor,
        log_rho_exact: Optional[callable] = None,
        n_steps: int = 1000,
        use_convexity_check: bool = False,
        convexity_check_samples: Optional[int] = None,
        beta_reduction_factor: float = 1.5,
        beta_min_floor: Optional[float] = 0.01,
    ):
        """
        Fit the normalizing flow to data.
        
        Args:
            x: Training data of shape (n_samples, n_dim)
            log_rho_exact: Optional function to compute exact log density for KL monitoring
            n_steps: Number of fitting steps
            use_convexity_check: If True, use dual LP to check convexity when beta < 0;
                reduce beta until convexity holds at sample points
            convexity_check_samples: If set, use this many samples for convexity check
                (for speed with large datasets). None = use all samples.
            beta_reduction_factor: Factor to divide beta by when convexity fails (default 1.5,
                less aggressive than 2.0 to allow more training)
            beta_min_floor: Minimum |beta| when convexity check reduces beta; ensures the
                model can still learn (default 0.01). None = no floor.
        """
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32, device=self.device)
        else:
            x = x.to(self.device)
            
        if x.ndim == 1:
            x = x.unsqueeze(1)
            
        m, n = x.shape
        tol = 1e-9
        
        # Preconditioning
        mean = torch.mean(x, dim=0)
        z = x - mean
        std = torch.sqrt(torch.mean(torch.sum(z**2, dim=1)) / n)
        z = z / std
        
        # CRITICAL: Store the initial whitened coordinates
        z_init = z.clone()
    
        self.preconditioning = {
            'mean': mean,
            'std': std
        }
        self.maps = []
    
        for i in range(n_steps):
            # 1. Select Center
            if torch.rand(1).item() > 0.5:
                idx = torch.randint(0, m, (1,)).item()
                x0 = z[idx].clone()
            else:
                x0 = torch.randn(n, device=self.device)
    
            # 2. Calculate Scale
            alpha = self._calculate_alpha(x0, n, m)
    
            # 3. Optimization Variables
            r = torch.norm(z - x0, dim=1)
            f = self._radial_f(r, alpha)
            f_p_over_r = self._radial_f_prime_over_r(r, alpha)
            
            # The fixed direction vector v = z_init - x0
            v = z_init - x0
            v_norm_sq = torch.sum(v**2, dim=1)
            
            # Projections
            z_dot_v = torch.sum(z * v, dim=1)      # Used for Gradient Energy term
            diff_dot_v = torch.sum((z - x0) * v, dim=1) # Used for Jacobian term
            
            # Jacobian Term C = (f'/r) * ((z-x0) . v)
            C = f_p_over_r * diff_dot_v
            
            # 4. Calculate Gradient (G) and Hessian (H)
            # G = - dE/dbeta + d(logdet)/dbeta
            G = -torch.sum(f * z_dot_v) + torch.sum(C)
            
            # H = - d2E/dbeta2 + d2(logdet)/dbeta2 (approx at beta=0)
            H = -torch.sum(f**2 * v_norm_sq) - torch.sum(C**2)
            
            if torch.abs(H) < tol:
                continue
    
            beta = -G / H
            beta = torch.clamp(beta, -self.epsilon, self.epsilon)
            beta_val = beta.detach().cpu().item()

            # Optional: Convexity check via dual LP (when beta < 0)
            if use_convexity_check and beta_val < 0:
                from .convexity_check import reduce_beta_for_convexity
                # Add map first (required for reduce_beta to compute log_prob)
                self.maps.append({
                    'x0': x0.detach().cpu(),
                    'alpha': alpha.detach().cpu(),
                    'beta': beta_val
                })
                x_data = (z * std + mean).detach().cpu().numpy()
                beta_val, _ = reduce_beta_for_convexity(
                    self,
                    x_data,
                    beta_val,
                    beta_reduction_factor=beta_reduction_factor,
                    max_reductions=10,
                    n_check_samples=convexity_check_samples,
                    beta_min_floor=beta_min_floor,
                )
                # Update map and beta with adjusted value
                self.maps[-1]['beta'] = beta_val
                beta = torch.tensor(beta_val, device=self.device)
            else:
                self.maps.append({
                    'x0': x0.detach().cpu(),
                    'alpha': alpha.detach().cpu(),
                    'beta': beta_val
                })

            # 5. Update Data
            z = z + beta * f.unsqueeze(1) * v

            if log_rho_exact is not None and (i+1) % 100 == 0:
                kl = self._estimate_forward_KL(x, log_rho_exact)
                print(f"Step {i+1}/{n_steps}, KL: {kl:.4f}")
                self.kl_history.append(kl)

    def _estimate_forward_KL(self, X: torch.Tensor, log_rho_exact: callable) -> float:
        """Estimate forward KL divergence"""
        # Compute log-density under the exact model
        if isinstance(X, torch.Tensor):
            X_np = X.detach().cpu().numpy()
        else:
            X_np = X
        log_ex = log_rho_exact(X_np)

        # Compute log-density under the flow estimate
        log_est = self.log_prob(X).detach().cpu().numpy()

        # Monte Carlo KL
        kl_mc = np.mean(log_ex - log_est)
        return kl_mc

    def _transform(self, x_new: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Applies the sequence of maps to new data."""
        if isinstance(x_new, np.ndarray):
            x_new = torch.tensor(x_new, dtype=torch.float32, device=self.device)
        else:
            x_new = x_new.to(self.device)
            
        if x_new.ndim == 1:
            x_new = x_new.unsqueeze(0)
        m, n = x_new.shape
        
        # Preconditioning
        z = (x_new - self.preconditioning['mean']) / self.preconditioning['std']
        z_init = z.clone()  # Anchor for the displacement
        
        log_J = torch.full((m,), -n * torch.log(self.preconditioning['std']), device=self.device)
    
        for p in self.maps:
            x0 = p['x0'].to(self.device)
            alpha = p['alpha'].to(self.device)
            beta = p['beta']
            
            r = torch.norm(z - x0, dim=1)
            f = self._radial_f(r, alpha)
            f_p_over_r = self._radial_f_prime_over_r(r, alpha)
            
            v = z_init - x0
            diff_dot_v = torch.sum((z - x0) * v, dim=1)
            
            # Determinant of (I + beta * f_p_over_r * v * (z-x0)^T)
            # Using Matrix Determinant Lemma: 1 + v^T u
            C = f_p_over_r * diff_dot_v
            term = 1 + beta * C
            
            # Update Log Jacobian
            valid = term > 0
            log_J_update = torch.zeros(m, device=self.device)
            log_J_update[valid] = torch.log(term[valid])
            log_J += log_J_update

            # Apply Map
            z = z + beta * f.unsqueeze(1) * v
            
        return z, log_J

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward transform: maps data to base distribution.
        Returns transformed data and log determinant.
        """
        y, logdet = self._transform(x)
        return y, logdet

    def log_prob(self, x_new: torch.Tensor) -> torch.Tensor:
        """
        Compute log probability of data under the learned distribution.
        
        Args:
            x_new: Data points of shape (n_samples, n_dim)
            
        Returns:
            Log probabilities of shape (n_samples,)
        """
        m, n = x_new.shape
        y, log_J = self._transform(x_new)
        log_prob_gaussian = -0.5 * torch.sum(y**2, dim=1) - 0.5 * n * np.log(2 * np.pi)
        return log_prob_gaussian + log_J

    def sample(self, n_samples: int, base_distribution: Optional[torch.distributions.Distribution] = None) -> torch.Tensor:
        """
        Generate samples from the learned distribution by sampling from base
        and applying inverse transform.
        
        Note: This requires solving an optimization problem for each sample.
        For efficiency, consider using a different sampling method.
        """
        if base_distribution is None:
            # Default to standard Gaussian
            flat_dim = len(self.preconditioning['mean'])
            base_distribution = torch.distributions.MultivariateNormal(
                torch.zeros(flat_dim, device=self.device),
                torch.eye(flat_dim, device=self.device)
            )
        
        # Sample from base
        y = base_distribution.sample((n_samples,))
        
        # Apply inverse transform (requires optimization)
        # This is a simplified version - in practice, you'd need to solve
        # the inverse problem for each sample
        warnings.warn("Sample method requires inverse transform optimization. "
                     "Consider using a more efficient sampling method.")
        
        # For now, return a placeholder - proper implementation would require
        # solving the inverse problem
        return y


def create_model(
    n_p: int = 50,
    epsilon: float = 0.5,
    device: Optional[torch.device] = None,
) -> RadialBasisFlow:
    """
    Factory function to create a RadialBasisFlow model.
    
    Args:
        n_p: Number of points parameter
        epsilon: Clipping parameter for beta
        device: Device to run on
        
    Returns:
        RadialBasisFlow model and model info
    """
    model = RadialBasisFlow(n_p=n_p, epsilon=epsilon, device=device)
    model_info = {
        "model_type": "RadialBasisFlow",
        "n_p": n_p,
        "epsilon": epsilon,
    }
    return model, model_info
