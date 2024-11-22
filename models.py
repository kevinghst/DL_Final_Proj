from typing import List
import numpy as np
from torch import nn
from torch.nn import functional as F
import torch


def build_mlp(layers_dims: List[int]):
    layers = []
    for i in range(len(layers_dims) - 2):
        layers.append(nn.Linear(layers_dims[i], layers_dims[i + 1]))
        layers.append(nn.BatchNorm1d(layers_dims[i + 1]))
        layers.append(nn.ReLU(True))
    layers.append(nn.Linear(layers_dims[-2], layers_dims[-1]))
    return nn.Sequential(*layers)


class MockModel(torch.nn.Module):
    """
    Does nothing. Just for testing.
    """

    def __init__(self, device="cuda", bs=64, n_steps=17, output_dim=256):
        super().__init__()
        self.device = device
        self.bs = bs
        self.n_steps = n_steps
        self.repr_dim = 256

    def forward(self, states, actions):
        """
        Args:
            states: [B, T, Ch, H, W]
            actions: [B, T-1, 2]

        Output:
            predictions: [B, T, D]
        """
        return torch.randn((self.bs, self.n_steps, self.repr_dim)).to(self.device)


class Prober(torch.nn.Module):
    def __init__(
        self,
        embedding: int,
        arch: str,
        output_shape: List[int],
    ):
        super().__init__()
        self.output_dim = np.prod(output_shape)
        self.output_shape = output_shape
        self.arch = arch

        arch_list = list(map(int, arch.split("-"))) if arch != "" else []
        f = [embedding] + arch_list + [self.output_dim]
        layers = []
        for i in range(len(f) - 2):
            layers.append(torch.nn.Linear(f[i], f[i + 1]))
            layers.append(torch.nn.ReLU(True))
        layers.append(torch.nn.Linear(f[-2], f[-1]))
        self.prober = torch.nn.Sequential(*layers)

    def forward(self, e):
        output = self.prober(e)
        return output

class LowEnergyOneModel(nn.Module):
    """
    First try
    """

    def __init__(self, device="cuda", bs=64, n_steps=17, output_dim=256, repr_dim=256, margin=1.0):
        super().__init__()
        self.device = device
        self.bs = bs
        self.n_steps = n_steps
        self.repr_dim = repr_dim
        self.action_dim = 2
        self.state_dim = (2, 64, 64)
        self.output_dim = output_dim
        self.margin = margin

        self.state_encoder = nn.Sequential(
            nn.Conv2d(in_channels=self.state_dim[0], out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(128 * (self.state_dim[1] // 4) * (self.state_dim[2] // 4), self.repr_dim)
        )

        self.action_encoder = nn.Sequential(
            nn.Linear(self.action_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.repr_dim)
        )

        self.combiner = nn.Sequential(
            nn.Linear(self.repr_dim * 2, self.repr_dim),
            nn.ReLU(),
            nn.Linear(self.repr_dim, self.output_dim)
        )

        self.energy_predictor = nn.Sequential(
            nn.Linear(self.output_dim, 1)
        )

    def forward(self, states, actions):
        """
        Args:
            states: [B, T, Ch, H, W]
            actions: [B, T-1, 2]

        Output:
            energy_scores: [B, T-1, 1]
        """
        B, T, Ch, H, W = states.size()

        states_encoded = self.state_encoder(states.view(B * T, Ch, H, W)).view(B, T, self.repr_dim)
        actions_encoded = self.action_encoder(actions.view(B * (T - 1), self.action_dim)).view(B, T - 1, self.repr_dim)
        combined = torch.cat((states_encoded[:, :-1, :], actions_encoded), dim=2)  # [B, T-1, repr_dim * 2]
        predictions = self.combiner(combined.view(B * (T - 1), self.repr_dim * 2)).view(B, T - 1, self.output_dim)
        energy_scores = self.energy_predictor(predictions.view(B * (T - 1), self.output_dim)).view(B, T - 1, 1)

        return energy_scores

    def loss(self, positive_energy, negative_energy):
        """
        Args:
            positive_energy: [B, T-1, 1] - Energy for positive pairs (compatible state-action).
            negative_energy: [B, T-1, 1] - Energy for negative pairs (incompatible state-action).

        Returns:
            contrastive_loss: Scalar - Energy-based loss.
        """
        return F.relu(positive_energy - negative_energy + self.margin).mean()

