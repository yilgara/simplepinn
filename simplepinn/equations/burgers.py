import torch
from simplepinn.core.grad import grad
from simplepinn.equations.base import BaseEquation


class BurgersEquation(BaseEquation):
    """
    Burgers' equation:
        u_t + u * u_x = nu * u_xx

    Residual form:
        u_t + u * u_x - nu * u_xx = 0
    """

    def __init__(self, nu=0.01):
        self.nu = nu

    def residual(self, model, coords):
        x = coords["x"]
        t = coords["t"]

        inputs = torch.cat([x, t], dim=1)
        u = model(inputs)

        u_t = grad(u, t)
        u_x = grad(u, x)
        u_xx = grad(u, x, order=2)

        return u_t + u * u_x - self.nu * u_xx
