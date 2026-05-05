from simplepinn.core.grad import grad
from simplepinn.equations.base import BaseEquation


class LaplaceEquation(BaseEquation):
    """
    1D Laplace equation:
        u_xx = 0

    Residual form:
        u_xx = 0
    """

    def residual(self, model, coords):
        x = coords["x"]
        inputs = x
        u = model(inputs)
        u_xx = grad(u, x, order=2)
        return u_xx
