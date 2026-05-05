import torch
import torch.nn as nn

from simplepinn.viz.plot import plot_1d

from .network import MLP


class PINN:
    """
    Physics-Informed Neural Network model.

    Handles:
    - neural network
    - sampling
    - PDE loss
    - training loop
    """

    def __init__(self, problem, hidden_layers=4, hidden_units=64):
        self.problem = problem
        self.hidden_layers = hidden_layers
        self.hidden_units = hidden_units

        # Input dimension = number of variables (e.g. x, t → 2)
        input_dim = len(problem.vars)

        self.model = MLP(
            input_dim=input_dim,
            output_dim=1,
            hidden_layers=hidden_layers,
            hidden_units=hidden_units
        )

        self.optimizer = None
        self.lambda_pde = 1.0
        self.lambda_bc = 1.0
        self.lambda_ic = 1.0

    def _format_domain(self):
        return " x ".join(f"[{low},{high}]" for low, high in self.problem.domain)

    def _class_names(self, items):
        if not items:
            return "None"

        return ", ".join(item.__class__.__name__ for item in items)

    def _equation_name(self):
        if not self.problem.pdes:
            return "None"

        return self.problem.pdes[0].__class__.__name__

    def summary(self):
        print("PINN Summary:")
        print(f"- Equation: {self._equation_name()}")
        print(f"- Domain: {self.problem.domain}")
        print(f"- Variables: {self.problem.vars}")
        print(f"- Network: {self.hidden_layers} layers, {self.hidden_units} units")

        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"- Parameters: {total_params}")

    # --------------------------------------------------
    # SAMPLING
    # --------------------------------------------------

    def _sample_interior(self, n_points):
        """
        Uniform sampling inside the domain.
        Returns dict: {"x": tensor, "t": tensor}
        """
        coords = {}

        for i, (low, high) in enumerate(self.problem.domain):
            var_name = self.problem.vars[i]

            x = torch.rand(n_points, 1) * (high - low) + low
            x.requires_grad_(True)

            coords[var_name] = x

        return coords

    # --------------------------------------------------
    # PDE LOSS
    # --------------------------------------------------

    def _compute_pde_loss(self, coords):
        """
        Computes total PDE loss across all equations.
        """
        loss = 0.0

        for pde in self.problem.pdes:
            residual = pde.residual(self.model, coords)
            loss += torch.mean(residual ** 2)

        return loss

    def _compute_boundary_loss(self):
        loss = 0.0

        for boundary in self.problem.boundaries:
            loss += boundary.loss(self.model, self.problem)

        return loss

    def _compute_initial_loss(self):
        loss = 0.0

        for initial in self.problem.initials:
            loss += initial.loss(self.model, self.problem)

        return loss

    # --------------------------------------------------
    # TRAINING
    # --------------------------------------------------

    def fit(self, epochs=1000, lr=1e-3, n_pde=1000):
        """
        Train the PINN model.
        """
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        print("\n=== Training PINN ===")
        print(f"PDE: {self._equation_name()}")
        print(f"Domain: {self.problem.domain}")
        print(f"Samples: {n_pde}")
        print("====================\n")

        for epoch in range(epochs):
            self.optimizer.zero_grad()

            # Sample interior points
            coords = self._sample_interior(n_pde)

            # Compute PDE loss
            loss_pde = self._compute_pde_loss(coords)

            loss_bc = self._compute_boundary_loss()
            loss_ic = self._compute_initial_loss()

            loss = (
                self.lambda_pde * loss_pde +
                self.lambda_bc * loss_bc +
                self.lambda_ic * loss_ic
            )

            # Backpropagation
            loss.backward()
            self.optimizer.step()

            # Logging
            if epoch % 100 == 0:
                print(
                    f"Epoch {epoch}: "
                    f"total={loss.item():.6f} | "
                    f"pde={loss_pde.item():.6f} (lambda={self.lambda_pde}) | "
                    f"bc={loss_bc.item():.6f} (lambda={self.lambda_bc}) | "
                    f"ic={loss_ic.item():.6f} (lambda={self.lambda_ic})"
                )

    def plot(self, t=0.0):
        plot_1d(self.model, self.problem, t_value=t)

    def predict(self, x, t=0.0):
        x = torch.as_tensor(x, dtype=torch.float32).reshape(-1, 1)
        t = torch.full_like(x, float(t))
        inputs = torch.cat([x, t], dim=1)

        was_training = self.model.training
        self.model.eval()

        with torch.no_grad():
            u = self.model(inputs)

        if was_training:
            self.model.train()

        return u
