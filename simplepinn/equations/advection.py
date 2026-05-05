import torch
from simplepinn.core.grad import grad
from simplepinn.equations.base import BaseEquation


class AdvectionEquation(BaseEquation):
    """
    1D advection equation:
        u_t + c * u_x = 0

    Residual form:
        u_t + c * u_x = 0
    """

    def __init__(self, c=1.0):
        self.c = c

    def residual(self, model, coords):
        x = coords["x"]
        t = coords["t"]

        inputs = torch.cat([x, t], dim=1)
        u = model(inputs)

        u_t = grad(u, t)
        u_x = grad(u, x)

        return u_t + self.c * u_x
