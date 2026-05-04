import math
import torch

from simplepinn.core.problem import Problem
from simplepinn.core.pinn import PINN
from simplepinn.equations import HeatEquation
from simplepinn.boundaries import Dirichlet, Function


def solve_heat(
    alpha=0.01,
    domain=(0, 1),
    time=(0, 1),
    initial="sin",
    boundary="zero",
    hidden_layers=4,
    hidden_units=64,
    epochs=2000,
    lr=1e-3,
    n_pde=1000,
    compute_error=True,
):
    problem = Problem(
        domain=[domain, time],
        vars=["x", "t"]
    )

    problem.add_pde(HeatEquation(alpha=alpha))

    if boundary == "zero":
        problem.add_boundary(
            Dirichlet(edges=["left", "right"], value=0.0)
        )
    else:
        raise ValueError("Only boundary='zero' is currently supported.")

    if initial == "sin":
        problem.add_initial(
            Function(lambda x: torch.sin(torch.pi * x))
        )
    elif callable(initial):
        problem.add_initial(Function(initial))
    else:
        raise ValueError("initial must be 'sin' or a callable function.")

    model = PINN(
        problem,
        hidden_layers=hidden_layers,
        hidden_units=hidden_units
    )

    model.summary()
    model.fit(epochs=epochs, lr=lr, n_pde=n_pde)

    if compute_error and initial == "sin" and boundary == "zero":
        x = torch.linspace(domain[0], domain[1], 200).reshape(-1, 1)
        t_value = 0.5

        u_pred = model.predict(x, t=t_value)

        u_exact = math.exp(-alpha * math.pi**2 * t_value) * torch.sin(math.pi * x)

        error = torch.mean(torch.abs(u_pred - u_exact))

        print(f"Mean absolute error at t={t_value}: {error.item():.6f}")

    return model


def solve(
    equation,
    **kwargs
):
    if equation == "heat":
        return solve_heat(**kwargs)

    elif equation == "burgers":
        from simplepinn.equations import BurgersEquation
        raise NotImplementedError("Burgers preset not added yet")

    else:
        raise ValueError(f"Unknown equation: {equation}")
