"""
Radial Basis Function Normalizing Flow

This package implements a Radial Basis Function (RBF) based normalizing flow
based on the Tabak-Turner paper. The implementation uses PyTorch and follows
a similar structure to the ICNN-based flows in the NF folder.
"""

from .model import RadialBasisFlow, create_model
from .train import train, train_with_validation, visualize_transformations
from .convexity_check import (
    check_convexity_dual_lp,
    check_convexity_at_samples,
    reduce_beta_for_convexity,
)
from .utils import (
    estimate_forward_KL,
    sample_exact_ring,
    log_rho_exact_ring,
    inverse_transform_optimization,
    sample_from_flow,
    compute_log_likelihood,
    compute_negative_log_likelihood,
)
from .tests import run_all_tests

__all__ = [
    'RadialBasisFlow',
    'create_model',
    'check_convexity_dual_lp',
    'check_convexity_at_samples',
    'reduce_beta_for_convexity',
    'train',
    'train_with_validation',
    'visualize_transformations',
    'estimate_forward_KL',
    'sample_exact_ring',
    'log_rho_exact_ring',
    'inverse_transform_optimization',
    'sample_from_flow',
    'compute_log_likelihood',
    'compute_negative_log_likelihood',
    'run_all_tests',
]

__version__ = '0.1.0'
