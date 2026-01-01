import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from .models import DeepConvexFlow

def _set_requires_grad(params, requires_grad: bool):
    for p in params:
        p.requires_grad = requires_grad

def _enforce_icnn_constraints(model):
    if hasattr(model, 'n_icnns') and model.n_icnns > 1:
        for icnn in model.icnn:
            icnn.enforce_constraints()
    else:
        model.icnn.enforce_constraints()

def train_stepwise(
    X,
    model: DeepConvexFlow,
    *,
    batch_size: int = 500,
    lr: float = 2e-2,
    epochs_per_step: int = 60,
    finetune_epochs: int = 100,
    print_every: int = 10,
    beta_min: float = 0.0,
    visualize: bool = True,
    n_viz_samples: int = 1000,
    train_prev: bool = False,
):
    """
    Stepwise training of DeepConvexFlow.
    
    Args:
        X: Training data.
        model: The DeepConvexFlow model.
        batch_size: Batch size for training.
        lr: Learning rate.
        epochs_per_step: Number of epochs per stage.
        finetune_epochs: Number of epochs for final fine-tuning.
        print_every: Print loss every n epochs.
        beta_min: Minimum value for beta.
        visualize: Whether to visualize reconstructions.
        n_viz_samples: Number of samples for visualization.
        train_prev: If True, train all previous steps along with the new one. 
                    If False, train only the new step.
    """
    device = next(model.parameters()).device
    X_tensor = torch.as_tensor(X, dtype=torch.float32, device=device)
    if X_tensor.ndim == 1:
        X_tensor = X_tensor.unsqueeze(1)

    dataset = TensorDataset(X_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    flat_dim = X_tensor[0].numel()
    base = torch.distributions.MultivariateNormal(
        torch.zeros(flat_dim, device=device),
        torch.eye(flat_dim, device=device),
    )

    # Sample base distribution for visualization
    S_viz = None
    if visualize:
        S_viz = base.sample((n_viz_samples,))

    def make_optimizer(parameters, lr_val):
        return torch.optim.AdamW(parameters, lr=lr_val, weight_decay=1e-5)

    def visualize_transformations(stage_name: str):
        """Visualize forward and inverse transformations"""
        if S_viz is None:
            return
            
        model.eval()
        with torch.no_grad():
            x_recon = model.reverse(S_viz)
            x_recon = x_recon.detach().cpu().numpy()
            X_np = X_tensor.cpu().numpy()
            
            plt.figure(figsize=(12, 5))
            plt.scatter(X_np[:, 0], X_np[:, 1], alpha=0.3, label='Data X')
            plt.scatter(x_recon[:, 0], x_recon[:, 1], alpha=0.3, label='Reconstructed X from S')
            plt.title(f"Reconstruction after {stage_name}")
            plt.legend()
            plt.show()
        
        model.train()

    def run_epochs(epochs, stage_name, parameters):
        model.train()
        optimizer = make_optimizer(parameters, lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        for epoch in range(1, epochs + 1):
            total_loss, total_count = 0.0, 0
            for (xb,) in loader:
                y, logdet = model.forward_transform(xb, logdet=0.0)
                logp0 = base.log_prob(y.reshape(y.shape[0], -1))
                loss = -(logp0 + logdet).mean()

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                
                optimizer.step()
                _enforce_icnn_constraints(model)

                # Enforce beta floor
                if model.n_icnns > 1 and beta_min > 0:
                    with torch.no_grad():
                        min_logit = torch.log(torch.exp(torch.tensor(beta_min, device=device)) - 1)
                        for i in range(model.n_icnns):
                            if model.beta[i].requires_grad:
                                model.beta[i].data.clamp_(min=min_logit.item())

                total_loss += loss.item() * xb.size(0)
                total_count += xb.size(0)

            scheduler.step()
            
            if epoch % print_every == 0:
                avg_loss = total_loss / total_count
                print(f"[{stage_name}] Epoch {epoch:03d}/{epochs:03d} - NLL = {avg_loss:.6f}")

    print("=" * 80)
    print("Stepwise Training")
    print("=" * 80)

    for k in range(model.n_icnns):
        print(f"\nStage {k+1}: Training ICNN {k}")
        
        model.set_active(k + 1)
        _set_requires_grad(model.parameters(), False)
        
        # Determine which parameters to train
        params_to_train = []
        
        if train_prev:
            # Train all active ICNNs (0 to k)
            if model.n_icnns > 1:
                for i in range(k + 1):
                    _set_requires_grad(model.icnn[i].parameters(), True)
                    model.beta[i].requires_grad_(True)
                    params_to_train.extend(model.icnn[i].parameters())
                    params_to_train.append(model.beta[i])
            else:
                _set_requires_grad(model.icnn.parameters(), True)
                model.beta[0].requires_grad_(True)
                params_to_train.extend(model.icnn.parameters())
                params_to_train.append(model.beta[0])
            
            stage_desc = f"icnn_0-{k}_joint"
        else:
            # Train only the new ICNN (k)
            if model.n_icnns > 1:
                _set_requires_grad(model.icnn[k].parameters(), True)
                model.beta[k].requires_grad_(True)
                params_to_train.extend(model.icnn[k].parameters())
                params_to_train.append(model.beta[k])
            else:
                _set_requires_grad(model.icnn.parameters(), True)
                model.beta[0].requires_grad_(True)
                params_to_train.extend(model.icnn.parameters())
                params_to_train.append(model.beta[0])
            
            stage_desc = f"icnn_{k}_solo"

        run_epochs(epochs_per_step, stage_desc, params_to_train)

        if visualize:
            visualize_transformations(f"After Stage {k+1}")

    # Final joint fine-tuning
    if finetune_epochs > 0:
        print("\n" + "=" * 80)
        print("Final Joint Fine-tuning")
        print("=" * 80)
        model.set_active(model.n_icnns)
        _set_requires_grad(model.parameters(), True)
        
        # Collect all parameters
        all_params = [p for p in model.parameters() if p.requires_grad]
        
        run_epochs(finetune_epochs, "final_finetune", all_params)

        if visualize:
            visualize_transformations("Final Model")