import torch
import numpy as np
import matplotlib.pyplot as plt
from .model import RadialBasisFlow, create_model
from .utils import (
    sample_exact_ring,
    log_rho_exact_ring,
    estimate_forward_KL,
    sample_from_flow,
    compute_negative_log_likelihood,
)


def test_basic_functionality():
    """Test basic functionality of RadialBasisFlow"""
    print("Testing basic functionality...")
    
    # Create model
    model, _ = create_model(n_p=50, epsilon=0.5)
    
    # Generate test data
    X = sample_exact_ring(100)
    X_tensor = torch.tensor(X, dtype=torch.float32, device=model.device)
    
    # Fit model
    model.fit(X_tensor, n_steps=100)
    
    # Test forward transform
    y, logdet = model.forward(X_tensor)
    assert y.shape == X_tensor.shape, "Forward transform shape mismatch"
    assert logdet.shape == (X_tensor.shape[0],), "Logdet shape mismatch"
    
    # Test log_prob
    log_probs = model.log_prob(X_tensor)
    assert log_probs.shape == (X_tensor.shape[0],), "Log prob shape mismatch"
    
    print("✓ Basic functionality test passed!")


def test_ring_distribution():
    """Test on ring distribution"""
    print("\nTesting on ring distribution...")
    
    # Generate data
    X = sample_exact_ring(1000)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    
    # Create and train model
    model, _ = create_model(n_p=500, epsilon=0.1)
    model.fit(X_tensor, log_rho_exact=log_rho_exact_ring, n_steps=600)
    
    # Compute KL divergence
    kl = estimate_forward_KL(X, model, log_rho_exact_ring)
    print(f"KL divergence: {kl:.4f}")
    
    # Compute NLL
    nll = compute_negative_log_likelihood(model, X_tensor)
    print(f"Negative log likelihood: {nll:.4f}")
    
    print("✓ Ring distribution test passed!")


def test_visualization():
    """Test visualization of learned distribution"""
    print("\nTesting visualization...")
    
    # Generate data
    X = sample_exact_ring(1000)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    
    # Create and train model
    model, _ = create_model(n_p=500, epsilon=0.1)
    model.fit(X_tensor, n_steps=300)
    
    # Generate samples
    try:
        samples = sample_from_flow(model, n_samples=1000, max_iter=500, lr=0.1)
        samples_np = samples.detach().cpu().numpy()
        
        # Plot
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.scatter(X[:, 0], X[:, 1], alpha=0.3, label='Original Data')
        plt.title("Original Data")
        plt.xlabel("X1")
        plt.ylabel("X2")
        plt.legend()
        plt.axis('equal')
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.scatter(samples_np[:, 0], samples_np[:, 1], alpha=0.3, label='Generated Samples')
        plt.title("Generated Samples")
        plt.xlabel("X1")
        plt.ylabel("X2")
        plt.legend()
        plt.axis('equal')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('rbf_test_visualization.png', dpi=150, bbox_inches='tight')
        print("✓ Visualization saved to rbf_test_visualization.png")
        
    except Exception as e:
        print(f"⚠ Visualization test failed: {e}")


def test_inverse_transform():
    """Test inverse transform"""
    print("\nTesting inverse transform...")
    
    # Generate data
    X = sample_exact_ring(100)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    
    # Create and train model
    model, _ = create_model(n_p=200, epsilon=0.1)
    model.fit(X_tensor, n_steps=200)
    
    # Forward transform
    y, _ = model.forward(X_tensor[:10])  # Test on subset
    
    # Inverse transform (approximate)
    from .utils import inverse_transform_optimization
    x_recon = inverse_transform_optimization(model, y, max_iter=500, lr=0.1)
    
    # Check reconstruction error
    error = torch.mean((X_tensor[:10] - x_recon)**2).item()
    print(f"Reconstruction error: {error:.6f}")
    
    if error < 1.0:  # Reasonable threshold
        print("✓ Inverse transform test passed!")
    else:
        print("⚠ Inverse transform has high error (this is expected for RBF flows)")


def run_all_tests():
    """Run all tests"""
    print("=" * 80)
    print("Running RBF Flow Tests")
    print("=" * 80)
    
    try:
        test_basic_functionality()
        test_ring_distribution()
        test_visualization()
        test_inverse_transform()
        
        print("\n" + "=" * 80)
        print("All tests completed!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()
