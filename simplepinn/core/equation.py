import inspect
import torch

from simplepinn.equations.base import BaseEquation


class FunctionEquation(BaseEquation):
    """
    Wrap a user-defined residual function as a SimplePINN equation.

    The function should accept u first, then variables by name:

        @sp.equation
        def my_pde(u, x, t):
            ...
    """

    def __init__(self, fn):
        self.fn = fn
        self.name = fn.__name__
        self.signature = inspect.signature(fn)

    def residual(self, model, coords):
        arg_names = list(self.signature.parameters)
        var_names = arg_names[1:]
        inputs = torch.cat([coords[name] for name in var_names], dim=1)
        u = model(inputs)
        values = [coords[name] for name in var_names]
        return self.fn(u, *values)


def equation(fn):
    return FunctionEquation(fn)
