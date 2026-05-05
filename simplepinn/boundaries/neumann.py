import torch
from simplepinn.core.grad import grad


class Neumann:
    """
    Neumann boundary condition:
        du/dx = value
    """

    def __init__(self, edges=None, value=0.0):
        self.edges = edges or []
        self.value = value
        if not self.edges:
            raise ValueError("Neumann boundary requires edges=['left', 'right'].")

    def loss(self, model, problem, n_points=100):
        total_loss = 0.0
        x_low, x_high = problem.domain[0]
        t_low, t_high = problem.domain[1]

        for edge in self.edges:
            t = torch.rand(n_points, 1) * (t_high - t_low) + t_low
            t.requires_grad_(True)

            if edge == "left":
                x = torch.full((n_points, 1), float(x_low))
            elif edge == "right":
                x = torch.full((n_points, 1), float(x_high))
            else:
                raise ValueError(f"Unknown edge: {edge}")

            x.requires_grad_(True)
            inputs = torch.cat([x, t], dim=1)
            u = model(inputs)
            u_x = grad(u, x)

            target = torch.full_like(u_x, float(self.value))
            total_loss += torch.mean((u_x - target) ** 2)

        return total_loss
