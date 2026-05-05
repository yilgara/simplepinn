import torch


class Sinusoidal:
    """
    Sinusoidal initial condition:
        u(x, 0) = amplitude * sin(frequency * pi * x)
    """

    def __init__(self, amplitude=1.0, frequency=1.0):
        self.amplitude = amplitude
        self.frequency = frequency

    def loss(self, model, problem, n_points=100):
        x_low, x_high = problem.domain[0]
        t_low, _ = problem.domain[1]

        x = torch.rand(n_points, 1) * (x_high - x_low) + x_low
        x.requires_grad_(True)

        t = torch.full((n_points, 1), float(t_low))
        t.requires_grad_(True)

        inputs = torch.cat([x, t], dim=1)
        u_pred = model(inputs)
        u_true = self.amplitude * torch.sin(self.frequency * torch.pi * x)

        return torch.mean((u_pred - u_true) ** 2)
