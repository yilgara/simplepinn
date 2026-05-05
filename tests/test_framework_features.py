import torch
import simplepinn as sp


def test_problem_defaults_match_readme_style_api():
    problem = sp.Problem()
    assert problem.domain == [(0, 1), (0, 1)]
    assert problem.vars == ["x", "t"]


def test_readme_style_training_runs_one_epoch():
    problem = sp.Problem()
    problem.add_pde(sp.equations.HeatEquation(alpha=0.01))
    problem.add_boundary(sp.boundaries.Dirichlet(value=0.0))
    problem.add_initial(sp.boundaries.Sinusoidal())

    model = sp.PINN(problem)
    model.fit(epochs=1, n_pde=8)

    assert len(model.history["total"]) == 1


def test_custom_equation_decorator_runs_one_epoch():
    @sp.equation
    def stationary(u, x, t):
        return sp.grad(u, t)

    problem = sp.Problem()
    problem.add_pde(stationary)

    model = sp.PINN(problem)
    model.fit(epochs=1, n_pde=8)

    assert len(model.history["pde"]) == 1


def test_data_constraints_contribute_to_history():
    problem = sp.Problem()
    problem.add_pde(sp.equations.HeatEquation(alpha=0.01))

    x = torch.linspace(0, 1, 4).reshape(-1, 1)
    t = torch.zeros_like(x)
    u = torch.sin(torch.pi * x)
    problem.add_data(x, t, u)

    model = sp.PINN(problem)
    model.fit(epochs=1, n_pde=8)

    assert model.history["data"][0] >= 0.0


def test_latin_hypercube_sampler_runs_one_epoch():
    problem = sp.Problem(sampler="latin_hypercube")
    problem.add_pde(sp.equations.HeatEquation(alpha=0.01))
    problem.add_boundary(sp.boundaries.Dirichlet(value=0.0))
    problem.add_initial(sp.boundaries.Sinusoidal())

    model = sp.PINN(problem)
    model.fit(epochs=1, n_pde=8, auto_weight=True)

    assert len(model.history["total"]) == 1


def test_heat_preset_smoke():
    model = sp.solve_heat(epochs=1, n_pde=8, compute_error=False)
    assert isinstance(model, sp.PINN)


def test_learnable_equation_parameter_is_optimized():
    alpha = torch.nn.Parameter(torch.tensor(0.05))

    problem = sp.Problem()
    problem.add_pde(sp.equations.HeatEquation(alpha=alpha))
    problem.add_boundary(sp.boundaries.Dirichlet(value=0.0))
    problem.add_initial(sp.boundaries.Sinusoidal())

    model = sp.PINN(problem)
    trainable_ids = {id(param) for param in model._trainable_parameters()}

    assert id(alpha) in trainable_ids
