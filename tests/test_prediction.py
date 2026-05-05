import torch
import simplepinn as sp


def test_predict_supports_static_1d_problem():
    problem = sp.Problem(domain=[(0, 1)], vars=["x"])
    problem.add_pde(sp.equations.LaplaceEquation())

    model = sp.PINN(problem)
    x = torch.linspace(0, 1, 5)
    prediction = model.predict(x)

    assert prediction.shape == (5, 1)
