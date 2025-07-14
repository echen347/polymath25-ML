import torch
import torch.nn as nn
from .layers import AffineCouplingLayer

class NormalizingFlow(nn.Module):
    """
    A Normalizing Flow model, constructed by chaining together multiple AffineCouplingLayers.
    This composition of simple, invertible transformations allows the model to learn
    a complex, high-dimensional probability distribution.
    """
    def __init__(self, dim, n_layers):
        super().__init__()
        # We create a sequence of coupling layers.
        # Note that the order matters. To make the model more expressive, one could
        # alternate the part of the input that gets transformed (e.g., by permuting the dimensions
        # between layers), but we keep it simple here.
        self.layers = nn.ModuleList([AffineCouplingLayer(dim) for _ in range(n_layers)])

    def forward(self, x):
        """
        The forward pass of the entire flow. It passes the input through all layers sequentially,
        accumulating the log-determinants of the Jacobians from each layer.
        """
        log_det_jacobian_total = 0
        for layer in self.layers:
            x, log_det_jacobian = layer(x)
            log_det_jacobian_total += log_det_jacobian
        return x, log_det_jacobian_total

    def inverse(self, y):
        """
        The inverse pass of the entire flow. It passes the input through all layers in reverse order.
        This is used to generate new samples by transforming points from the base distribution.
        """
        for layer in reversed(self.layers):
            y = layer.inverse(y)
        return y 