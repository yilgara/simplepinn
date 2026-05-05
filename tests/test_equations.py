import torch
import simplepinn as sp


def _coords_xt(n=8):
    x = torch.rand(n, 1, requires_grad=True)
    t = torch.rand(n, 1, requires_grad=True)
    return {"x": x, "t": t}


def _coords_x(n=8):
    x = torch.rand(n, 1, requires_grad=True)
    return {"x": x}


def test_time_dependent_equation_residual_shapes():
    model = sp.core.MLP(input_dim=2, output_dim=1)
    coords = _coords_xt()

    equations = [
        sp.equations.HeatEquation(),
        sp.equations.WaveEquation(),
        sp.equations.BurgersEquation(),
        sp.equations.AdvectionEquation(),
    ]

    for equation in equations:
        residual = equation.residual(model, coords)
        assert residual.shape == (8, 1)


def test_static_equation_residual_shapes():
    model = sp.core.MLP(input_dim=1, output_dim=1)
    coords = _coords_x()

    equations = [
        sp.equations.PoissonEquation(lambda x: torch.sin(torch.pi * x)),
        sp.equations.LaplaceEquation(),
    ]

    for equation in equations:
        residual = equation.residual(model, coords)
        assert residual.shape == (8, 1)
