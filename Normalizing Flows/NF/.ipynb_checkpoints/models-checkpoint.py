import copy
import gc
import warnings
from typing import Optional, Iterable

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.func import vmap, hessian

class ICNN(nn.Module):
    def __init__(self, n_dim, hidden_dims):
        super(ICNN, self).__init__()
        self.n_dim = n_dim
        self.hidden_dims = hidden_dims
        self.alpha = 1.0  

        self.Wx_layers = nn.ModuleList()
        self.Wz_layers = nn.ModuleList()
        self.b = nn.ParameterList()


        prev_dim = 0
        for idx, hidden_dim in enumerate(hidden_dims):
            wx = nn.Linear(n_dim, hidden_dim)
            wz = nn.Linear(prev_dim, hidden_dim)

            if idx==0:
                with torch.no_grad():
                    wz.weight.zero_()

            self.Wx_layers.append(wx)
            self.Wz_layers.append(wz)
            self.b.append(nn.Parameter(torch.zeros(hidden_dim)))

            prev_dim = hidden_dim
        self.output_layer = nn.Linear(hidden_dims[-1], 1, bias=False)

    def forward(self, x):
        z = None

        for i in range(len(self.hidden_dims)):
            wx_x = self.Wx_layers[i](x)

            if i==0:
                linear_sum = wx_x + self.b[i]
            else:
                wz_z = self.Wz_layers[i](z)
                linear_sum = wx_x + wz_z + self.b[i]
            z = F.softplus(linear_sum)

        output = self.output_layer(z)
        return output

    def enforce_constraints(self):
        """
        Enforces the non-negativity constraint on W^(z) weights and output layer.
        This should be called after the optimizer.step() during training.
        """
        with torch.no_grad():
            for i in range(1, len(self.Wz_layers)):
                self.Wz_layers[i].weight.copy_(torch.abs(self.Wz_layers[i].weight))
            
            # Ensure output layer weights are also non-negative
            self.output_layer.weight.copy_(torch.abs(self.output_layer.weight))


class DeepConvexFlow(nn.Module):
    """
    Deep convex potential flow with REDUCED quadratic regularization
    """

    def __init__(
        self,
        icnn: nn.Module,
        dim: int,
        n_icnns: int = 3,
        atol: float = 1e-3,
        bias_w1: float = 0.0,
        w0_scale: float = 0.1,  # CHANGED: Much smaller quadratic weight
    ):
        super().__init__()
        self.dim = dim
        self.n_icnns = int(n_icnns)
        self.atol = atol
        self._active = n_icnns

        # ---- ICNNs and beta weights ----
        if self.n_icnns > 1:
            self.icnn = nn.ModuleList([copy.deepcopy(icnn) for _ in range(self.n_icnns)])
            self.beta = nn.ParameterList([nn.Parameter(torch.tensor(bias_w1, dtype=torch.float32))
                                          for _ in range(self.n_icnns)])
        else:
            self.icnn = icnn
            self.beta = nn.ParameterList([nn.Parameter(torch.tensor(bias_w1, dtype=torch.float32))])

        # CHANGED: Small fixed quadratic term (0.1 instead of 1.0)
        self.w0 = torch.nn.Parameter(torch.log(torch.exp(torch.tensor(1)) - 1))

    @torch.no_grad()
    def set_active(self, count: int):
        self._active = max(0, min(int(count), self.n_icnns))

    def get_potential(self, x: torch.Tensor) -> torch.Tensor:
        n = x.shape[0]
        # CHANGED: Smaller quadratic regularization
        quad = self.w0 * (x.reshape(n, -1).pow(2).sum(dim=1, keepdim=True) / 2.0)

        if self.n_icnns > 1:
            icnn_weighted_sum = 0.0
            n_active = getattr(self, "_active", self.n_icnns)
            for i in range(n_active):
                icnn_out = self.icnn[i](x)
                icnn_weighted_sum = icnn_weighted_sum + F.softplus(self.beta[i]) * icnn_out
        else:
            icnn_weighted_sum = F.softplus(self.beta[0]) * self.icnn(x)

        return icnn_weighted_sum + quad

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.enable_grad():
            x = x.clone().requires_grad_(True)
            Phi = self.get_potential(x)
            f = torch.autograd.grad(Phi.sum(), x, create_graph=True)[0]
        return f

    def reverse(self, y: torch.Tensor, max_iter: int = 100_000, lr: float = 1.0,
                tol: float = 1e-12, x: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        if x is None:
            x = y.clone().detach().requires_grad_(True)

        def closure():
            Fval = self.get_potential(x)
            loss = Fval.sum() - (x * y).sum()
            (grad_x,) = torch.autograd.grad(loss, (x,), retain_graph=False)
            x.grad = grad_x.detach()
            return loss

        opt = torch.optim.LBFGS([x], lr=lr, line_search_fn="strong_wolfe",
                                max_iter=max_iter, tolerance_grad=tol, tolerance_change=tol)
        opt.step(closure)

        if x.is_cuda:
            torch.cuda.empty_cache()
        gc.collect()
        return x

    def forward_transform(self, x: torch.Tensor, logdet: torch.Tensor | float = 0.0, extra=None):
        return self.forward_transform_bruteforce(x, logdet)

    def _batch_logdet_psd(self, H: torch.Tensor, jitter: float) -> torch.Tensor:
        B, D, _ = H.shape
        I = torch.eye(D, dtype=H.dtype, device=H.device).expand(B, D, D)
        Hj = H + jitter * I

        L, info = torch.linalg.cholesky_ex(Hj)
        chol_logdet = 2.0 * torch.sum(torch.log(torch.diagonal(L, dim1=-2, dim2=-1)), dim=-1)

        sign, slog = torch.linalg.slogdet(Hj)
        ok = (info == 0) & (sign > 0)
        return torch.where(ok, chol_logdet, slog)

    def forward_transform_bruteforce(self, x: torch.Tensor, logdet: torch.Tensor | float = 0.0):
        warnings.warn("brute force", UserWarning)

        bsz = x.shape[0]
        input_shape = x.shape[1:]

        with torch.enable_grad():
            x = x.clone().requires_grad_(True)
            Phi = self.get_potential(x)
            f = torch.autograd.grad(Phi.sum(), x, create_graph=True)[0]

        x_flat = x.reshape(bsz, -1)

        def Phi_scalar_on_flat(x_row_flat: torch.Tensor) -> torch.Tensor:
            x_row = x_row_flat.reshape(1, *input_shape)
            return self.get_potential(x_row).sum()

        H = vmap(hessian(Phi_scalar_on_flat))(x_flat)
        logdet_H = self._batch_logdet_psd(H, jitter=self.atol)

        if isinstance(logdet, (int, float)):
            logdet = torch.as_tensor(logdet, dtype=logdet_H.dtype, device=logdet_H.device)
        if logdet.ndim == 0:
            logdet = logdet.expand_as(logdet_H)

        f = f.reshape(bsz, *input_shape)
        return f, logdet + logdet_H


def create_model(
    n_dim: int,
    hidden_dims: Iterable[int],
    n_icnns: int = 3,
    w0_scale: float = 0.1,
    bias_w1: float = 0,
) -> DeepConvexFlow:
    icnn = ICNN(n_dim=n_dim, hidden_dims=list(hidden_dims))
    model = DeepConvexFlow(
        icnn=icnn,
        dim=n_dim,
        n_icnns=n_icnns,
        w0_scale=w0_scale,
        bias_w1=bias_w1,
    )
    betas = [b for b in model.beta]
    model_info = {
        "model_type": "DeepConvexFlow",
        "n_dim": n_dim,
        "hidden_dims": hidden_dims,
        "n_icnns": n_icnns,
        "w0_scale": w0_scale,
        "bias_w1": bias_w1,
        "betas_init": betas,
    }
    return model, model_info