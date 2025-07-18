import torch
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model import NormalizingFlow
from distributions import get_base_distribution, get_target_distribution_samples

def train_and_evaluate():
    # --- Hyperparameters ---
    dim = 2  # Dimension of the data
    n_layers = 8  # Number of coupling layers in the flow
    n_samples = 2048  # Number of samples to use for training and plotting
    n_epochs = 10000  # Number of training epochs
    learning_rate = 1e-4  # Learning rate for the optimizer

    # --- Model and Optimizer ---
    model = NormalizingFlow(dim=dim, n_layers=n_layers)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # --- Base Distribution ---
    # We use a simple distribution (a standard multivariate normal) as our base.
    # The model will learn to transform samples from this distribution to the target distribution.
    base_dist = torch.distributions.MultivariateNormal(torch.zeros(dim), torch.eye(dim))

    print("Training...")
    # --- Training Loop ---
    for epoch in range(n_epochs + 1):
        optimizer.zero_grad()
        
        # 1. Get samples from the target distribution
        x = get_target_distribution_samples(n_samples)
        
        # 2. Pass samples through the model (forward pass)
        # This gives us the transformed samples 'z' and the log-determinant of the Jacobian.
        z, log_det_jacobian = model(x)
        
        # 3. Calculate the loss
        # The loss is the negative log-likelihood of the samples.
        # By the change of variables formula, log p(x) = log p(z) + log|det J|.
        # We want to maximize log p(x), which is equivalent to minimizing -log p(x).
        log_prob = base_dist.log_prob(z) + log_det_jacobian
        loss = -log_prob.mean()
        
        # 4. Backpropagation and optimization
        loss.backward()
        optimizer.step()
        
        # Print loss periodically to make training verbose
        if epoch % 500 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

    print("Training finished.")

    # --- Evaluation ---
    # Generate new samples by passing base distribution samples through the inverse of the model.
    print("Generating and plotting samples...")
    with torch.no_grad():
        z = get_base_distribution(n_samples, dim)
        # The inverse function of the model maps from the base distribution to the target distribution.
        # It does not return a log-determinant, which was the source of the error.
        generated_samples = model.inverse(z)

    # --- Plotting ---
    target_samples = get_target_distribution_samples(n_samples)
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(target_samples[:, 0], target_samples[:, 1], alpha=0.5, s=10)
    plt.title("Target Distribution")
    plt.xlabel("x1")
    plt.ylabel("x2")

    plt.subplot(1, 2, 2)
    plt.scatter(generated_samples[:, 0], generated_samples[:, 1], alpha=0.5, c='r', s=10)
    plt.title("Generated Samples")
    plt.xlabel("x1")
    plt.ylabel("x2")

    plt.tight_layout()
    plt.savefig("nf_results.png")
    print("Plot saved to nf_results.png")

if __name__ == "__main__":
    train_and_evaluate() 