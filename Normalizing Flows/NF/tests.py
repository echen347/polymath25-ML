import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_samples(samples: np.ndarray, true_samples: np.ndarray, title: str = "Samples vs True"):
    plt.figure(figsize=(6,6))
    plt.scatter(samples[:,0], samples[:,1], alpha=0.5)
    plt.scatter(true_samples[:,0], true_samples[:,1], alpha=0.5)
    plt.title(title)
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.axis('equal')
    plt.grid()
    plt.show()

def l2_test(true_samples: np.ndarray, generated_samples: np.ndarray) -> float:
    """
    Computes the L2 distance between the true samples and generated samples.
    Both inputs should be of shape (n_samples, n_dim).
    """
    diff = true_samples - generated_samples
    l2_distance = np.mean(np.sum(np.abs(diff), axis=1))
    return l2_distance

def plot_kl(kl_history: list[float], title: str = "KL Divergence over Epochs"):
    plt.figure(figsize=(8,5))
    plt.plot(kl_history)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("KL Divergence")
    plt.grid()
    plt.show()

def plot_l2(l2_history: list[float], title: str = "L2 Distance over Epochs"):
    plt.figure(figsize=(8,5))
    plt.plot(l2_history)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("L2 Distance")
    plt.grid()
    plt.show()