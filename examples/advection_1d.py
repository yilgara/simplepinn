import torch
import simplepinn as sp


problem = sp.Problem(
    domain=[(0, 1), (0, 1)],
    vars=["x", "t"]
)

problem.add_pde(
    sp.equations.AdvectionEquation(c=1.0)
)

problem.add_boundary(
    sp.boundaries.Dirichlet(edges=["left"], value=0.0)
)

problem.add_initial(
    sp.boundaries.Function(lambda x: torch.sin(torch.pi * x))
)

model = sp.PINN(problem, hidden_layers=4, hidden_units=64)
model.fit(epochs=2000, lr=1e-3, n_pde=1000)
model.plot(t=0.5)
