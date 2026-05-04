import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    Simple fully connected neural network for PINNs.

    Approximates u(x, t; θ)
    """

    def __init__(self, input_dim, output_dim=1, hidden_layers=4, hidden_units=64):
        super().__init__()

        layers = []

        # Input layer
        layers.append(nn.Linear(input_dim, hidden_units))
        layers.append(nn.Tanh())

        # Hidden layers
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_units, hidden_units))
            layers.append(nn.Tanh())

        # Output layer
        layers.append(nn.Linear(hidden_units, output_dim))

        self.model = nn.Sequential(*layers)

        self._initialize_weights()

    def forward(self, x):
        return self.model(x)

    def _initialize_weights(self):
        """
        Xavier initialization for stable training.
        """
        for m in self.model:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
