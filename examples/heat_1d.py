import numpy as np
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

# --- Compare with analytical solution ---
def exact_solution(x, t, alpha=0.01):
    return torch.exp(torch.tensor(-np.pi**2 * alpha * t)) * torch.sin(torch.pi * x)


# Generate test points
x = torch.linspace(0, 1, 200).reshape(-1, 1)

u_pred = model.predict(x, t=0.5)
u_exact = exact_solution(x, t=0.5)

error = torch.mean(torch.abs(u_pred - u_exact))
print("Error:", error.item())

model.plot(t=0.5)
