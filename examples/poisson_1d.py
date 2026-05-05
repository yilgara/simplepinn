import torch
import matplotlib.pyplot as plt
import simplepinn as sp


problem = sp.Problem(
    domain=[(0, 1)],
    vars=["x"]
)

problem.add_pde(
    sp.equations.PoissonEquation(
        source=lambda x: -(torch.pi ** 2) * torch.sin(torch.pi * x)
    )
)

problem.add_data(
    torch.tensor([[0.0], [1.0]]),
    torch.tensor([[0.0], [0.0]])
)

model = sp.PINN(problem, hidden_layers=4, hidden_units=64)
model.fit(epochs=2000, lr=1e-3, n_pde=1000)

x = torch.linspace(0, 1, 200).reshape(-1, 1)
u_pred = model.predict(x).detach()
u_exact = torch.sin(torch.pi * x)

error = torch.mean(torch.abs(u_pred - u_exact)).item()
print(f"Mean absolute error: {error:.6f}")

plt.plot(x.numpy(), u_pred.numpy(), label="PINN")
plt.plot(x.numpy(), u_exact.numpy(), "--", label="Exact")
plt.xlabel("x")
plt.ylabel("u(x)")
plt.title("Poisson equation solution")
plt.grid(True)
plt.legend()
plt.show()
