from simplepinn.core.grad import grad
from simplepinn.equations.base import BaseEquation


class PoissonEquation(BaseEquation):
    """
    1D Poisson equation:
        u_xx = f(x)

    Residual form:
        u_xx - f(x) = 0
    """

    def __init__(self, source):
        self.source = source

    def residual(self, model, coords):
        x = coords["x"]
        inputs = x
        u = model(inputs)
        u_xx = grad(u, x, order=2)
        f = self.source(x)
        return u_xx - f
