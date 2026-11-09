from dataclasses import dataclass

import torch
from torch import Tensor

from .ppo import PPOBatch


@dataclass(frozen=True)
class RolloutConfig:
    discount_factor: float = 0.99
    gae_lambda: float = 0.95


class RolloutBuffer:
    """Stores transitions and converts them into a PPO batch."""

    def __init__(self, config: RolloutConfig | None = None) -> None:
        self.config = config or RolloutConfig()
        self.transitions: list[dict[str, object]] = []

    def add(self, transition: dict[str, object]) -> None:
        self.transitions.append(transition)

    def __len__(self) -> int:
        return len(self.transitions)

    def clear(self) -> None:
        self.transitions.clear()

    def to_batch(self, last_value: float | Tensor = 0.0) -> PPOBatch:
        if not self.transitions:
            raise ValueError("cannot build a PPO batch from an empty rollout")

        rewards = torch.tensor(
            [float(transition["reward"]) for transition in self.transitions],
            dtype=torch.float32,
        )
        values = torch.cat(
            [self._as_tensor(transition["old_values"]).reshape(-1) for transition in self.transitions]
        ).to(torch.float32)
        dones = torch.tensor(
            [bool(transition["done"]) for transition in self.transitions],
            dtype=torch.float32,
        )
        bootstrap_value = torch.as_tensor(last_value, dtype=torch.float32).reshape(()).item()
        advantages = torch.zeros_like(rewards)
        next_advantage = 0.0

        for index in range(len(self.transitions) - 1, -1, -1):
            next_value = bootstrap_value if index == len(self.transitions) - 1 else values[index + 1].item()
            non_terminal = 1.0 - dones[index]
            temporal_difference = rewards[index] + self.config.discount_factor * next_value * non_terminal - values[index]
            next_advantage = temporal_difference + self.config.discount_factor * self.config.gae_lambda * non_terminal * next_advantage
            advantages[index] = next_advantage

        returns = advantages + values
        return PPOBatch(
            grid=self._stack("grid"),
            player_features=self._stack("player_features"),
            weapon_features=self._stack("weapon_features"),
            enemy_type_ids=self._stack("enemy_type_ids").to(torch.long),
            enemy_features=self._stack("enemy_features"),
            actions=self._stack("actions"),
            old_log_probabilities=self._stack("old_log_probabilities").reshape(-1),
            returns=returns,
            advantages=advantages,
            old_values=values,
        )

    def _stack(self, name: str) -> Tensor:
        return torch.cat(
            [self._as_tensor(transition[name]).detach().cpu() for transition in self.transitions],
            dim=0,
        )

    @staticmethod
    def _as_tensor(value: object) -> Tensor:
        if not isinstance(value, Tensor):
            raise TypeError("rollout fields must be torch tensors")
        return value
