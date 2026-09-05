"""
The policy and value networks for skrl's PPO: one small CNN each over the environment's
(6, 60, 80) observation. skrl hands models a flattened batch; `unflatten` puts the image
back. Both are yours to change; the environment is the contract.
"""

from __future__ import annotations

import torch
from skrl.models.torch import CategoricalMixin, DeterministicMixin, Model
from skrl.utils.spaces.torch import unflatten_tensorized_space
from torch import nn


def cnn(channels: int, height: int, width: int, out: int = 256) -> nn.Sequential:
    """Nature-DQN-shaped: three convolutions, one linear layer, ReLU throughout."""
    features = nn.Sequential(
        nn.Conv2d(channels, 32, 8, stride=4),
        nn.ReLU(),
        nn.Conv2d(32, 64, 4, stride=2),
        nn.ReLU(),
        nn.Conv2d(64, 64, 3, stride=1),
        nn.ReLU(),
        nn.Flatten(),
    )
    with torch.no_grad():
        flat = features(torch.zeros(1, channels, height, width)).shape[1]
    return nn.Sequential(features, nn.Linear(flat, out), nn.ReLU())


class Policy(CategoricalMixin, Model):
    """Logits over the four movements."""

    def __init__(self, observation_space, action_space, device=None) -> None:
        Model.__init__(
            self, observation_space=observation_space, action_space=action_space, device=device
        )
        CategoricalMixin.__init__(self, unnormalized_log_prob=True)
        channels, height, width = observation_space.shape
        self.features = cnn(channels, height, width)
        self.logits = nn.Linear(256, self.num_actions)

    def compute(self, inputs, role=""):
        image = unflatten_tensorized_space(self.observation_space, inputs["observations"])
        return self.logits(self.features(image.float() / 255.0)), {}


class Value(DeterministicMixin, Model):
    """The state value, one number."""

    def __init__(self, observation_space, action_space, device=None) -> None:
        Model.__init__(
            self, observation_space=observation_space, action_space=action_space, device=device
        )
        DeterministicMixin.__init__(self)
        channels, height, width = observation_space.shape
        self.features = cnn(channels, height, width)
        self.value = nn.Linear(256, 1)

    def compute(self, inputs, role=""):
        image = unflatten_tensorized_space(self.observation_space, inputs["observations"])
        return self.value(self.features(image.float() / 255.0)), {}
