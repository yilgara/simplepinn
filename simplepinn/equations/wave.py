import torch
from simplepinn.core.grad import grad
from simplepinn.equations.base import BaseEquation


class WaveEquation(BaseEquation):
    """
    Wave equation:
        u_tt = c^2 * u_xx

    Residual form:
        u_tt - c^2 * u_xx = 0
    """

    def __init__(self, c=1.0):
        self.c = c

    def residual(self, model, coords):
        x = coords["x"]
        t = coords["t"]

        inputs = torch.cat([x, t], dim=1)
        u = model(inputs)

        u_tt = grad(u, t, order=2)
        u_xx = grad(u, x, order=2)

        return u_tt - (self.c ** 2) * u_xx
