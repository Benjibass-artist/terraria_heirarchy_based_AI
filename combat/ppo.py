from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .model import MeleeCombatPolicy


@dataclass(frozen=True)
class PPOConfig:
    clip_ratio: float = 0.2
    value_loss_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    learning_rate: float = 3e-4
    update_epochs: int = 4
    max_gradient_norm: float = 0.5


@dataclass(frozen=True)
class PPOBatch:
    grid: Tensor
    player_features: Tensor
    weapon_features: Tensor
    actions: Tensor
    old_log_probabilities: Tensor
    returns: Tensor
    advantages: Tensor
    old_values: Tensor
    enemy_type_ids: Tensor | None = None
    enemy_features: Tensor | None = None
    weapon_type_id: Tensor | None = None


class PPOTrainer:
    """PPO optimizer for the melee combat actor-critic."""

    def __init__(
        self,
        policy: MeleeCombatPolicy,
        config: PPOConfig | None = None,
    ) -> None:
        self.policy = policy
        self.config = config or PPOConfig()
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=self.config.learning_rate,
        )

    def update(self, batch: PPOBatch) -> dict[str, float]:
        advantages = batch.advantages
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        metrics: dict[str, float] = {}

        for _ in range(self.config.update_epochs):
            logits, values, _ = self.policy(
                batch.grid,
                batch.player_features,
                batch.weapon_features,
                enemy_type_ids=batch.enemy_type_ids,
                enemy_features=batch.enemy_features,
                weapon_type_id=batch.weapon_type_id,
            )
            log_probabilities, entropy = self.policy.evaluate_actions(logits, batch.actions)
            probability_ratio = (log_probabilities - batch.old_log_probabilities).exp()
            unclipped_objective = probability_ratio * advantages
            clipped_ratio = probability_ratio.clamp(
                1.0 - self.config.clip_ratio,
                1.0 + self.config.clip_ratio,
            )
            clipped_objective = clipped_ratio * advantages
            policy_loss = -torch.minimum(unclipped_objective, clipped_objective).mean()

            clipped_values = batch.old_values + (
                values - batch.old_values
            ).clamp(-self.config.clip_ratio, self.config.clip_ratio)
            unclipped_value_loss = (values - batch.returns).square()
            clipped_value_loss = (clipped_values - batch.returns).square()
            value_loss = 0.5 * torch.maximum(
                unclipped_value_loss,
                clipped_value_loss,
            ).mean()
            entropy_bonus = entropy.mean()
            total_loss = (
                policy_loss
                + self.config.value_loss_coefficient * value_loss
                - self.config.entropy_coefficient * entropy_bonus
            )

            self.optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            gradient_norm = nn.utils.clip_grad_norm_(
                self.policy.parameters(),
                self.config.max_gradient_norm,
            )
            self.optimizer.step()

            metrics = {
                "loss": float(total_loss.detach()),
                "policy_loss": float(policy_loss.detach()),
                "value_loss": float(value_loss.detach()),
                "entropy": float(entropy_bonus.detach()),
                "gradient_norm": float(gradient_norm.detach()),
                "mean_ratio": float(probability_ratio.detach().mean()),
            }

        return metrics
