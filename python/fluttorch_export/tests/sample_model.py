"""A model for exercising the exporter, addressed the way a real one would be.

Kept here rather than inside a test so the CLI can resolve it as
``tests.sample_model:build``, which is the same path a caller's own package
takes — a fixture the CLI cannot address does not test the CLI.
"""

from __future__ import annotations

import torch

IN_FEATURES, HIDDEN, OUT_FEATURES = 4, 8, 3
SEED = 20260730


class TwoLayer(torch.nn.Module):
    """The smallest model with an interior worth attributing drift to."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(IN_FEATURES, HIDDEN)
        self.act = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(HIDDEN, OUT_FEATURES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


def build() -> TwoLayer:
    """A model with fixed weights, so an artifact is reproducible."""
    torch.manual_seed(SEED)
    return TwoLayer().eval()


def example_inputs() -> torch.Tensor:
    torch.manual_seed(SEED + 1)
    return torch.randn(1, IN_FEATURES)


def golden_cases() -> list[torch.Tensor]:
    """Four inputs spanning the ranges the model is expected to see.

    Not random draws: each one is chosen for what it exercises, because goldens
    sampled from a distribution nobody chose describe the sampler rather than the
    model.
    """
    return [
        torch.zeros(1, IN_FEATURES),  # every ReLU on the boundary
        torch.ones(1, IN_FEATURES),  # uniformly positive
        -torch.ones(1, IN_FEATURES),  # uniformly negative, most units dead
        torch.tensor([[1e3, -1e3, 1e-3, 0.0]]),  # wide dynamic range
    ]


LABELS = ["low", "mid", "high"]
