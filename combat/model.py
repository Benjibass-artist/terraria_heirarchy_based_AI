from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.distributions import Bernoulli


@dataclass(frozen=True)
class CombatObservationSpec:
    grid_channels: int = 8
    grid_height: int = 45
    grid_width: int = 80
    enemy_type_count: int = 512
    enemy_embedding_size: int = 8
    player_features: int = 16
    weapon_features: int = 136
    weapon_type_count: int = 512
    weapon_embedding_size: int = 8
    enemy_slots: int = 32
    enemy_features: int = 13
    hidden_size: int = 256
    gru_layers: int = 1
    movement_action_count: int = 5
    aim_vector_size: int = 2
    auto_aim_action_count: int = 1
    target_action_count: int = 32


class MeleeCombatPolicy(nn.Module):
    """CNN-GRU actor-critic for melee combat.

    Actions are independent binary controls in this order:
    left, right, jump, down, attack.
    """

    def __init__(self, spec: CombatObservationSpec | None = None) -> None:
        super().__init__()
        self.spec = spec or CombatObservationSpec()

        self.enemy_type_embedding = nn.Embedding(
            self.spec.enemy_type_count,
            self.spec.enemy_embedding_size,
            padding_idx=0,
        )
        self.weapon_type_embedding = nn.Embedding(
            self.spec.weapon_type_count,
            self.spec.weapon_embedding_size,
            padding_idx=0,
        )

        self.grid_encoder = nn.Sequential(
            nn.Conv2d(
                self.spec.grid_channels + self.spec.enemy_embedding_size,
                32,
                kernel_size=5,
                stride=2,
                padding=2,
            ),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 96, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

        self.vector_encoder = nn.Sequential(
            nn.Linear(
                self.spec.player_features
                + self.spec.weapon_features
                + self.spec.weapon_embedding_size,
                128,
            ),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )

        self.enemy_encoder = nn.Sequential(
            nn.Linear(self.spec.enemy_slots * self.spec.enemy_features, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )

        self.recurrent = nn.GRU(
            input_size=96 + 128 + 128,
            hidden_size=self.spec.hidden_size,
            num_layers=self.spec.gru_layers,
            batch_first=True,
        )
        self.policy_head = nn.Linear(self.spec.hidden_size, self.spec.movement_action_count)
        self.aim_head = nn.Linear(self.spec.hidden_size, self.spec.aim_vector_size)
        self.aim_log_std = nn.Parameter(torch.full((self.spec.aim_vector_size,), -1.0))
        self.auto_aim_head = nn.Linear(self.spec.hidden_size, self.spec.auto_aim_action_count)
        self.target_head = nn.Linear(self.spec.hidden_size, self.spec.target_action_count)
        self.value_head = nn.Linear(self.spec.hidden_size, 1)

    def forward(
        self,
        grid: Tensor,
        player_features: Tensor,
        weapon_features: Tensor,
        hidden_state: Tensor | None = None,
        enemy_type_ids: Tensor | None = None,
        enemy_features: Tensor | None = None,
        weapon_type_id: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return action logits, state values, and the next GRU state.

        Inputs may be one timestep ([batch, ...]) or a sequence
        ([batch, time, ...]).
        """
        grid, player_features, weapon_features, enemy_type_ids, enemy_features, weapon_type_id, was_single_step = self._add_time_axis(
            grid, player_features, weapon_features, enemy_type_ids, enemy_features, weapon_type_id
        )
        batch_size, time_steps = grid.shape[:2]

        embedded_enemy_types = self.enemy_type_embedding(enemy_type_ids)
        embedded_enemy_types = embedded_enemy_types.permute(0, 1, 4, 2, 3)
        encoded_grid = torch.cat((grid, embedded_enemy_types), dim=2)
        grid_features = self.grid_encoder(
            encoded_grid.reshape(batch_size * time_steps, *encoded_grid.shape[2:])
        )
        vector = torch.cat((player_features, weapon_features), dim=-1)
        weapon_embedding = self.weapon_type_embedding(weapon_type_id)
        vector = torch.cat((vector, weapon_embedding), dim=-1)
        vector_features = self.vector_encoder(vector.reshape(batch_size * time_steps, -1))
        enemy_features = enemy_features.reshape(batch_size * time_steps, -1)
        enemy_features = self.enemy_encoder(enemy_features)
        encoded = torch.cat((grid_features, vector_features, enemy_features), dim=-1)
        encoded = encoded.reshape(batch_size, time_steps, -1)

        recurrent_features, next_hidden = self.recurrent(encoded, hidden_state)
        movement_logits = self.policy_head(recurrent_features)
        aim_logits = self.aim_head(recurrent_features)
        auto_aim_logits = self.auto_aim_head(recurrent_features)
        target_logits = self.target_head(recurrent_features)
        logits = torch.cat((movement_logits, aim_logits, auto_aim_logits, target_logits), dim=-1)
        values = self.value_head(recurrent_features).squeeze(-1)

        if was_single_step:
            return logits[:, 0], values[:, 0], next_hidden
        return logits, values, next_hidden

    def sample_action(
        self,
        grid: Tensor,
        player_features: Tensor,
        weapon_features: Tensor,
        hidden_state: Tensor | None = None,
        enemy_type_ids: Tensor | None = None,
        enemy_features: Tensor | None = None,
        weapon_type_id: Tensor | None = None,
        deterministic: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        logits, values, next_hidden = self(
            grid,
            player_features,
            weapon_features,
            hidden_state,
            enemy_type_ids,
            enemy_features,
            weapon_type_id,
        )
        movement_logits = logits[..., :self.spec.movement_action_count]
        aim_start = self.spec.movement_action_count
        aim_end = aim_start + self.spec.aim_vector_size
        aim_logits = logits[..., aim_start:aim_end]
        auto_aim_logits = logits[..., aim_end:]
        auto_aim_end = aim_end + self.spec.auto_aim_action_count
        auto_aim_logits = logits[..., aim_end:auto_aim_end]
        target_end = auto_aim_end + self.spec.target_action_count
        target_logits = logits[..., auto_aim_end:target_end]
        movement_distribution = Bernoulli(logits=movement_logits)
        aim_distribution = torch.distributions.Normal(aim_logits, self.aim_log_std.exp())
        auto_aim_distribution = Bernoulli(logits=auto_aim_logits)
        target_distribution = torch.distributions.Categorical(logits=target_logits)
        movement_actions = (
            (movement_logits >= 0).to(dtype=torch.float32)
            if deterministic
            else movement_distribution.sample()
        )
        raw_aim_actions = aim_logits if deterministic else aim_distribution.sample()
        aim_actions = torch.tanh(raw_aim_actions)
        aim_log_probability = self._squashed_log_probability(
            aim_distribution, raw_aim_actions, aim_actions
        )
        auto_aim_actions = (
            (auto_aim_logits >= 0).to(dtype=torch.float32)
            if deterministic
            else auto_aim_distribution.sample()
        )
        target_index = target_logits.argmax(dim=-1) if deterministic else target_distribution.sample()
        target_actions = torch.nn.functional.one_hot(
            target_index, num_classes=self.spec.target_action_count
        ).to(dtype=torch.float32)
        actions = torch.cat((movement_actions, aim_actions, auto_aim_actions, target_actions), dim=-1)
        log_probabilities = (
            movement_distribution.log_prob(movement_actions).sum(dim=-1)
            + aim_log_probability
            + auto_aim_distribution.log_prob(auto_aim_actions).sum(dim=-1)
            + target_distribution.log_prob(target_index)
        )
        return actions, log_probabilities, values, next_hidden

    def evaluate_actions(self, logits: Tensor, actions: Tensor) -> tuple[Tensor, Tensor]:
        """Return PPO log probabilities and entropy for sampled actions."""
        movement_logits = logits[..., :self.spec.movement_action_count]
        aim_start = self.spec.movement_action_count
        aim_end = aim_start + self.spec.aim_vector_size
        aim_logits = logits[..., aim_start:aim_end]
        auto_aim_end = aim_end + self.spec.auto_aim_action_count
        auto_aim_logits = logits[..., aim_end:auto_aim_end]
        target_end = auto_aim_end + self.spec.target_action_count
        target_logits = logits[..., auto_aim_end:target_end]
        movement_actions = actions[..., :self.spec.movement_action_count]
        aim_actions = actions[..., aim_start:aim_end]
        auto_aim_actions = actions[..., aim_end:auto_aim_end]
        target_actions = actions[..., auto_aim_end:target_end]
        movement_distribution = Bernoulli(logits=movement_logits)
        aim_distribution = torch.distributions.Normal(aim_logits, self.aim_log_std.exp())
        auto_aim_distribution = Bernoulli(logits=auto_aim_logits)
        target_distribution = torch.distributions.Categorical(logits=target_logits)
        aim_actions = actions[..., aim_start:aim_end].clamp(-0.999999, 0.999999)
        raw_aim_actions = torch.atanh(aim_actions)
        aim_log_probability = self._squashed_log_probability(
            aim_distribution, raw_aim_actions, aim_actions
        )
        log_probabilities = (
            movement_distribution.log_prob(movement_actions).sum(dim=-1)
            + aim_log_probability
            + auto_aim_distribution.log_prob(auto_aim_actions).sum(dim=-1)
            + target_distribution.log_prob(target_actions.argmax(dim=-1))
        )
        entropy = (
            movement_distribution.entropy().sum(dim=-1)
            + aim_distribution.entropy().sum(dim=-1)
            + auto_aim_distribution.entropy().sum(dim=-1)
            + target_distribution.entropy()
        )
        return log_probabilities, entropy

    @staticmethod
    def _squashed_log_probability(
        distribution: torch.distributions.Normal,
        raw_actions: Tensor,
        actions: Tensor,
    ) -> Tensor:
        correction = torch.log(1.0 - actions.square() + 1e-6)
        return (distribution.log_prob(raw_actions) - correction).sum(dim=-1)

    def _add_time_axis(
        self,
        grid: Tensor,
        player_features: Tensor,
        weapon_features: Tensor,
        enemy_type_ids: Tensor | None,
        enemy_features: Tensor | None,
        weapon_type_id: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, bool]:
        if grid.ndim == 4:
            if enemy_type_ids is None:
                enemy_type_ids = torch.zeros(
                    grid.shape[0], grid.shape[2], grid.shape[3],
                    dtype=torch.long, device=grid.device
                )
            if enemy_features is None:
                enemy_features = torch.zeros(
                    grid.shape[0], self.spec.enemy_slots, self.spec.enemy_features,
                    dtype=grid.dtype, device=grid.device
                )
            if weapon_type_id is None:
                weapon_type_id = torch.zeros(grid.shape[0], dtype=torch.long, device=grid.device)
            return (
                grid.unsqueeze(1),
                player_features.unsqueeze(1),
                weapon_features.unsqueeze(1),
                enemy_type_ids.unsqueeze(1),
                enemy_features.unsqueeze(1),
                weapon_type_id.unsqueeze(1),
                True,
            )
        if grid.ndim == 5:
            if enemy_type_ids is None:
                enemy_type_ids = torch.zeros(
                    grid.shape[0], grid.shape[1], grid.shape[3], grid.shape[4],
                    dtype=torch.long, device=grid.device
                )
            if enemy_features is None:
                enemy_features = torch.zeros(
                    grid.shape[0], grid.shape[1], self.spec.enemy_slots, self.spec.enemy_features,
                    dtype=grid.dtype, device=grid.device
                )
            if weapon_type_id is None:
                weapon_type_id = torch.zeros(grid.shape[0], grid.shape[1], dtype=torch.long, device=grid.device)
            return grid, player_features, weapon_features, enemy_type_ids, enemy_features, weapon_type_id, False
        raise ValueError("grid must have shape [batch, channels, height, width] or [batch, time, channels, height, width]")
