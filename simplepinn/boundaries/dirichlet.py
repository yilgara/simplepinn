import torch


class Dirichlet:
    """
    Dirichlet boundary condition:
        u = value
    """

    def __init__(self, edges=None, value=0.0):
        if edges is None:
            edges = ["left", "right"]

        if not edges:
            raise ValueError("Dirichlet boundary requires edges=['left', 'right']")

        self.edges = edges
        self.value = value

    def loss(self, model, problem, n_points=100):
        total_loss = 0.0

        # For now supports 1D space + time: vars=["x", "t"]
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
            u_pred = model(inputs)

            target = torch.full_like(u_pred, float(self.value))
            total_loss += torch.mean((u_pred - target) ** 2)

        return total_loss
