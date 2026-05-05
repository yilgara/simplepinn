import simplepinn as sp


def test_public_api_exports_core_objects():
    assert sp.Problem is not None
    assert sp.PINN is not None
    assert sp.grad is not None
    assert sp.equation is not None
    assert sp.solve is not None
    assert sp.solve_heat is not None
    assert sp.solve_burgers is not None


def test_public_api_exports_equations_and_boundaries():
    assert sp.equations.HeatEquation is not None
    assert sp.equations.WaveEquation is not None
    assert sp.equations.BurgersEquation is not None
    assert sp.equations.PoissonEquation is not None
    assert sp.equations.LaplaceEquation is not None
    assert sp.equations.AdvectionEquation is not None
    assert sp.boundaries.Dirichlet is not None
    assert sp.boundaries.Neumann is not None
    assert sp.boundaries.Function is not None
    assert sp.boundaries.Sinusoidal is not None
