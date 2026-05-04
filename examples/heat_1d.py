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
    return np.exp(-np.pi**2 * alpha * t) * np.sin(np.pi * x)


# Generate test points
x = np.linspace(0, 1, 200).reshape(-1, 1)
t = np.full_like(x, 0.5)

inputs = torch.tensor(np.hstack([x, t]), dtype=torch.float32)

with torch.no_grad():
    u_pred = model.model(inputs).cpu().numpy()

u_exact = exact_solution(x, 0.5)

# Compute error
error = np.mean(np.abs(u_pred - u_exact))

print(f"\nMean absolute error vs analytical solution: {error:.6f}")

model.plot(t=0.5)
