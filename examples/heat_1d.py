import torch
import simplepinn as sp


# Define problem
problem = sp.Problem(
    domain=[(0, 1), (0, 1)],  # x ∈ [0,1], t ∈ [0,1]
    vars=["x", "t"]
)

# PDE: heat equation
problem.add_pde(sp.equations.HeatEquation(alpha=0.01))

# Boundary: u(0,t) = u(1,t) = 0
problem.add_boundary(
    sp.boundaries.Dirichlet(edges=["left", "right"], value=0.0)
)

# Initial: u(x,0) = sin(pi x)
problem.add_initial(
    sp.boundaries.Function(lambda x: torch.sin(torch.pi * x))
)

# Model
model = sp.PINN(problem, hidden_layers=4, hidden_units=64)

# Train
model.fit(epochs=2000, lr=1e-3)

print("Training finished.")

model.plot(t=0.5)
