import torch
import matplotlib.pyplot as plt
import simplepinn as sp


true_alpha = 0.01
learned_alpha = torch.nn.Parameter(torch.tensor(0.05))


def exact_solution(x, t, alpha=true_alpha):
    return torch.exp(-alpha * torch.pi**2 * t) * torch.sin(torch.pi * x)


problem = sp.Problem(
    domain=[(0, 1), (0, 1)],
    vars=["x", "t"]
)

problem.add_pde(
    sp.equations.HeatEquation(alpha=learned_alpha)
)

problem.add_boundary(
    sp.boundaries.Dirichlet(edges=["left", "right"], value=0.0)
)

problem.add_initial(
    sp.boundaries.Sinusoidal()
)

# Sparse observations from the true physics.
x_data = torch.linspace(0.1, 0.9, 9).reshape(-1, 1)
t_data = torch.full_like(x_data, 0.5)
u_data = exact_solution(x_data, t_data)

problem.add_data(x_data, t_data, u_data)

model = sp.PINN(problem, hidden_layers=4, hidden_units=64)
model.lambda_data = 10.0

print(f"True alpha: {true_alpha:.4f}")
print(f"Initial alpha guess: {learned_alpha.item():.4f}")

model.fit(epochs=3000, lr=1e-3, n_pde=1000)

print(f"Recovered alpha: {learned_alpha.item():.4f}")

x = torch.linspace(0, 1, 200).reshape(-1, 1)
t_plot = 0.5
u_pred = model.predict(x, t=t_plot).detach()
u_exact = exact_solution(x, torch.full_like(x, t_plot))

error = torch.mean(torch.abs(u_pred - u_exact)).item()
print(f"Mean absolute error at t={t_plot}: {error:.6f}")

plt.plot(x.numpy(), u_pred.numpy(), label="PINN")
plt.plot(x.numpy(), u_exact.numpy(), "--", label="Exact")
plt.scatter(x_data.numpy(), u_data.numpy(), color="red", label="Sparse observations")
plt.xlabel("x")
plt.ylabel("u(x,t)")
plt.title(f"Inverse heat problem: recovered alpha={learned_alpha.item():.4f}")
plt.grid(True)
plt.legend()
plt.show()
