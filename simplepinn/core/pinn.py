import torch
import torch.nn as nn

from simplepinn.samplers import AdaptiveSampler, LatinHypercubeSampler, UniformSampler
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
        self.lambda_data = 1.0
        self.history = {
            "epoch": [],
            "total": [],
            "pde": [],
            "bc": [],
            "ic": [],
            "data": [],
        }

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
        physics_params = sum(p.numel() for p in self._physics_parameters())
        print(f"- Network parameters: {total_params}")
        print(f"- Physics parameters: {physics_params}")

    def _physics_parameters(self):
        params = []

        for item in [*self.problem.pdes, *self.problem.boundaries, *self.problem.initials]:
            for value in vars(item).values():
                if isinstance(value, nn.Parameter) and value.requires_grad:
                    params.append(value)

        return params

    def _trainable_parameters(self):
        return [*self.model.parameters(), *self._physics_parameters()]

    # --------------------------------------------------
    # SAMPLING
    # --------------------------------------------------

    def _sample_interior(self, n_points):
        """
        Sampling inside the domain.
        Returns dict: {"x": tensor, "t": tensor}
        """
        samplers = {
            "uniform": UniformSampler,
            "latin_hypercube": LatinHypercubeSampler,
            "adaptive": AdaptiveSampler,
        }

        sampler = self.problem.sampler

        if isinstance(sampler, str):
            if sampler not in samplers:
                raise ValueError(
                    "Unknown sampler. Choose 'uniform', 'latin_hypercube', or 'adaptive'."
                )

            sampler = samplers[sampler]()

        return sampler.sample(self.problem.domain, self.problem.vars, n_points)

    def _build_inputs(self, *values):
        tensors = [
            torch.as_tensor(value, dtype=torch.float32).reshape(-1, 1)
            for value in values
        ]
        return torch.cat(tensors, dim=1)


    # --------------------------------------------------
    # PDE LOSS
    # --------------------------------------------------

    def _compute_pde_loss(self, coords):
        """
        Computes total PDE loss across all equations.
        """
        loss = torch.tensor(0.0)

        for pde in self.problem.pdes:
            residual = pde.residual(self.model, coords)
            loss += torch.mean(residual ** 2)

        return loss

    def _compute_boundary_loss(self):
        loss = torch.tensor(0.0)

        for boundary in self.problem.boundaries:
            loss += boundary.loss(self.model, self.problem)

        return loss

    def _compute_initial_loss(self):
        loss = torch.tensor(0.0)

        for initial in self.problem.initials:
            loss += initial.loss(self.model, self.problem)

        return loss

    def _compute_data_loss(self):
        loss = torch.tensor(0.0)

        for data in self.problem.data:
            if len(data) == 2:
                inputs, target = data
                inputs = torch.as_tensor(inputs, dtype=torch.float32)
                target = torch.as_tensor(target, dtype=torch.float32).reshape(-1, 1)
            elif len(data) == 3:
                x, t, target = data
                inputs = self._build_inputs(x, t)
                target = torch.as_tensor(target, dtype=torch.float32).reshape(-1, 1)
            else:
                raise ValueError("data constraints must be (inputs, target) or (x, t, target)")

            prediction = self.model(inputs)
            loss += torch.mean((prediction - target) ** 2)

        return loss

    def _loss_weights(self, loss_pde, loss_bc, loss_ic, loss_data, auto_weight):
        if not auto_weight:
            return (
                self.lambda_pde,
                self.lambda_bc,
                self.lambda_ic,
                self.lambda_data,
            )

        losses = [loss_pde, loss_bc, loss_ic, loss_data]
        active = [loss.detach() for loss in losses if loss.detach().item() > 0.0]

        if not active:
            return 1.0, 1.0, 1.0, 1.0

        mean_loss = torch.stack(active).mean()

        weights = []
        for loss in losses:
            if loss.detach().item() == 0.0:
                weights.append(0.0)
            else:
                weights.append((mean_loss / (loss.detach() + 1e-8)).item())

        return tuple(weights)

    # --------------------------------------------------
    # TRAINING
    # --------------------------------------------------

    def fit(
        self,
        epochs=1000,
        lr=1e-3,
        n_pde=1000,
        progress_callback=None,
        auto_weight=False,
    ):
        """
        Train the PINN model.
        """
        self.optimizer = torch.optim.Adam(self._trainable_parameters(), lr=lr)
        self.history = {
            "epoch": [],
            "total": [],
            "pde": [],
            "bc": [],
            "ic": [],
            "data": [],
        }

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
            loss_data = self._compute_data_loss()
            weight_pde, weight_bc, weight_ic, weight_data = self._loss_weights(
                loss_pde,
                loss_bc,
                loss_ic,
                loss_data,
                auto_weight,
            )

            loss = (
                weight_pde * loss_pde +
                weight_bc * loss_bc +
                weight_ic * loss_ic +
                weight_data * loss_data
            )

            # Backpropagation
            loss.backward()
            self.optimizer.step()

            self.history["epoch"].append(epoch)
            self.history["total"].append(loss.item())
            self.history["pde"].append(loss_pde.item())
            self.history["bc"].append(loss_bc.item())
            self.history["ic"].append(loss_ic.item())
            self.history["data"].append(loss_data.item())

            if progress_callback and epoch % max(1, epochs // 100) == 0:
                progress_callback(epoch, epochs, self.history)

            # Logging
            if epoch % 100 == 0:
                print(
                    f"Epoch {epoch}: "
                    f"total={loss.item():.6f} | "
                    f"pde={loss_pde.item():.6f} (lambda={weight_pde}) | "
                    f"bc={loss_bc.item():.6f} (lambda={weight_bc}) | "
                    f"ic={loss_ic.item():.6f} (lambda={weight_ic}) | "
                    f"data={loss_data.item():.6f} (lambda={weight_data})"
                )

    def plot(self, t=0.0):
        plot_1d(self.model, self.problem, t_value=t)

    def predict(self, x, t=0.0):
        x = torch.as_tensor(x, dtype=torch.float32).reshape(-1, 1)

        if len(self.problem.vars) == 1:
            inputs = x
        else:
            t = torch.full_like(x, float(t))
            inputs = torch.cat([x, t], dim=1)

        was_training = self.model.training
        self.model.eval()

        with torch.no_grad():
            u = self.model(inputs)

        if was_training:
            self.model.train()

        return u
