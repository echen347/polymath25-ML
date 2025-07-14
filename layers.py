import torch
import torch.nn as nn

class AffineCouplingLayer(nn.Module):
    """
    An affine coupling layer, a core component of many normalizing flows (e.g., RealNVP).
    It splits the input into two halves. One half is used to compute the parameters (scale and shift)
    for an affine transformation that is applied to the other half.
    This design makes the transformation easily invertible and the Jacobian determinant easy to compute.
    """
    def __init__(self, dim, hidden_dim=256):
        super().__init__()
        self.dim = dim
        # A simple neural network to compute the scale and shift parameters.
        # It takes one half of the input and outputs parameters for the other half.
        self.net = nn.Sequential(
            nn.Linear(dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            # The output size is 'dim' because we need a scale and a shift parameter
            # for each dimension in the half of the input that gets transformed.
            nn.Linear(hidden_dim, dim) 
        )

    def forward(self, x):
        """
        Defines the forward pass of the layer. This is the direction used during training
        to transform data from the complex target distribution to the simple base distribution.
        """
        # Split the input into two halves, x_a and x_b.
        x_a, x_b = x.chunk(2, dim=1)
        
        # x_a is passed through the network to get scale (s) and shift (t) parameters.
        # x_a remains unchanged (identity transformation).
        log_s, t = self.net(x_a).chunk(2, dim=1)
        
        # The scale parameter 's' must be positive. We use a sigmoid function to ensure this.
        s = torch.sigmoid(log_s + 2)
        
        # The other half, x_b, is transformed using the computed scale and shift.
        y_b = (x_b + t) * s
        
        # The output of the forward pass is the concatenation of the unchanged part and the transformed part.
        y_a = x_a
        
        # The log-determinant of the Jacobian for this transformation is simply the sum of the logs of the scale factors.
        # This is efficient to compute because the Jacobian matrix is triangular.
        log_det_jacobian = s.log().sum(dim=1)
        
        return torch.cat([y_a, y_b], dim=1), log_det_jacobian

    def inverse(self, y):
        """
        Defines the inverse pass of the layer. This is used for generation (sampling),
        transforming data from the simple base distribution to the complex target distribution.
        """
        # Split the input 'y' into its two halves.
        y_a, y_b = y.chunk(2, dim=1)
        
        # The parameters for the inverse transformation are computed in the same way, from y_a.
        log_s, t = self.net(y_a).chunk(2, dim=1)
        s = torch.sigmoid(log_s + 2)
        
        # The inverse transformation is applied to y_b.
        x_b = y_b / s - t
        
        # The other half, x_a, is the same as y_a.
        x_a = y_a
        
        return torch.cat([x_a, x_b], dim=1) 