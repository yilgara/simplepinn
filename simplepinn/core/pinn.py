import torch
import torch.nn as nn

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

        # Input dimension = number of variables (e.g. x, t → 2)
        input_dim = len(problem.vars)

        self.model = MLP(
            input_dim=input_dim,
            output_dim=1,
            hidden_layers=hidden_layers,
            hidden_units=hidden_units
        )

        self.optimizer = None

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

        for epoch in range(epochs):
            self.optimizer.zero_grad()

            # Sample interior points
            coords = self._sample_interior(n_pde)

            # Compute PDE loss
            loss_pde = self._compute_pde_loss(coords)

            loss_bc = self._compute_boundary_loss()
            loss_ic = self._compute_initial_loss()

            loss = loss_pde + loss_bc + loss_ic

            # Backpropagation
            loss.backward()
            self.optimizer.step()

            # Logging
            if epoch % 100 == 0:
                print(f"Epoch {epoch}: loss = {loss.item():.6f}")
