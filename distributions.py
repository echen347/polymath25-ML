import torch
import numpy as np

def get_base_distribution(n_samples, dim):
    """
    Returns samples from the base distribution, which is a simple,
    easy-to-sample-from distribution. Here, we use a standard
    multivariate normal (Gaussian) distribution.
    """
    return torch.randn(n_samples, dim)

def get_target_distribution_samples(n_samples):
    """
    Returns samples from the target distribution. This is the complex
    distribution that we want our model to learn. For this example,
    we use a 2D "banana"-shaped distribution, which is a common
    benchmark for generative models.
    """
    x = torch.randn(n_samples, 2)
    # This transformation creates the non-linear "banana" shape.
    x[:, 1] = x[:, 1] * 0.5 + 0.5 * x[:, 0]**2
    return x 