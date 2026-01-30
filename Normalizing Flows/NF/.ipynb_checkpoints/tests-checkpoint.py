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
    return np.mean(np.linalg.norm(true_samples - generated_samples, axis=1))