import torch
from simplepinn.core.grad import grad


class HeatEquation:
    """
    Heat equation:
        u_t = alpha * u_xx

    Residual form:
        u_t - alpha * u_xx = 0
    """

    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def residual(self, model, coords):
        x = coords["x"]
        t = coords["t"]

        inputs = torch.cat([x, t], dim=1)
        u = model(inputs)

        u_t = grad(u, t)
        u_xx = grad(u, x, order=2)

        return u_t - self.alpha * u_xx
