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
        print(f'state {states.size()}')
        print(f'actions {actions.size()}')
        r = torch.randn((self.bs, self.n_steps, self.repr_dim)).to(self.device)
        print(f'return {r.size()}')
        return r


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
    Modified to output predictions [B, T, D]
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

    def forward(self, states, actions):
        """
        Args:
            states: [B, T, Ch, H, W]
            actions: [B, T-1, 2]

        Output:
            predictions: [B, T, D]
        """
        B, T, Ch, H, W = states.size()

        states_encoded = self.state_encoder(states.view(B * T, Ch, H, W)).view(B, T, self.repr_dim)
        actions_encoded = self.action_encoder(actions.view(B * (T - 1), self.action_dim)).view(B, T - 1, self.repr_dim)
        combined = torch.cat((states_encoded[:, :-1, :], actions_encoded), dim=2) 
        predictions = self.combiner(combined.view(B * (T - 1), self.repr_dim * 2)).view(B, T - 1, self.output_dim)
        last_state_pred = self.combiner(
            torch.cat((states_encoded[:, -1, :], torch.zeros_like(actions_encoded[:, -1, :])), dim=1)
        )
        predictions = torch.cat((predictions, last_state_pred.unsqueeze(1)), dim=1)

        return predictions

class LowEnergyTwoModel(nn.Module):
    # TODO: simplify this
    def __init__(self, device="cuda", bs=64, n_steps=17, output_dim=256, repr_dim=256):
        super().__init__()
        self.encoder = Encoder(input_shape=(3, 64, 64), repr_dim=repr_dim)
        self.predictor = Predictor(repr_dim=repr_dim, action_dim=4)
        self.target_encoder = TargetEncoder(input_shape=(3, 64, 64), repr_dim=repr_dim)
    
    def forward(self, observations, actions):
        states = self.encoder(observations[:, :-1])  # skip last observation
        target_states = self.target_encoder(observations[:, 1:])  # skip first observation

        predicted_states = []
        for t in range(actions.size(1)):
            predicted_state = self.predictor(states[:, t], actions[:, t])
            predicted_states.append(predicted_state)
            states[:, t + 1] = predicted_state  # teacher forcing

        predicted_states = torch.stack(predicted_states, dim=1)

        return predicted_states, target_states

    def loss(predicted_states, target_states):
        mse_loss = F.mse_loss(predicted_states, target_states)
        variance = target_states.var(dim=0).mean()
        var_loss = F.relu(1e-2 - variance).mean()
        cov = torch.cov(target_states.T)
        cov_loss = (cov.fill_diagonal_(0).pow(2).sum() / cov.size(0))

        return mse_loss + var_loss + cov_loss


class Encoder(nn.Module):
    def __init__(self, input_shape, repr_dim=256):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear((input_shape[1] // 4) * (input_shape[2] // 4) * 64, repr_dim)
        )
    
    def forward(self, x):
        return self.cnn(x)

class Predictor(nn.Module):
    def __init__(self, repr_dim=256, action_dim=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(repr_dim + action_dim, repr_dim),
            nn.ReLU(),
            nn.Linear(repr_dim, repr_dim)
        )
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.fc(x)

class TargetEncoder(Encoder):
    def __init__(self, input_shape, repr_dim=256):
        super().__init__(input_shape, repr_dim)

