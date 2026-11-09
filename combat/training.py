from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import torch
from torch.optim import Optimizer

from .checkpoints import WeaponCheckpointStore
from .ppo import PPOBatch, PPOTrainer
from .scheduler import WeaponTrainingScheduler


@dataclass
class TrainingState:
    global_steps: int = 0
    active_weapon_id: int | None = None
    weapon_steps: dict[int, int] = field(default_factory=dict)


class WeaponTrainingManager:
    """Coordinates shared-policy training across changing weapons.

    The policy remains shared. Checkpoints preserve optimizer/model state and
    per-weapon progress so a stopped run can resume without losing history.
    """

    def __init__(
        self,
        trainer: PPOTrainer,
        checkpoint_directory: str | Path = "checkpoints",
        scheduler: WeaponTrainingScheduler | None = None,
    ) -> None:
        self.trainer = trainer
        self.scheduler = scheduler or WeaponTrainingScheduler()
        self.checkpoints = WeaponCheckpointStore(checkpoint_directory)
        self.state = TrainingState()

    def observe_weapon(self, weapon_id: int, steps: int = 1) -> bool:
        """Record the equipped weapon and return whether it changed."""
        weapon_id = int(weapon_id)
        changed = self.state.active_weapon_id != weapon_id
        self.state.active_weapon_id = weapon_id
        self.state.weapon_steps.setdefault(weapon_id, 0)
        self.advance_steps(steps)
        return changed

    def advance_steps(self, steps: int) -> None:
        if steps < 0:
            raise ValueError("steps cannot be negative")
        self.state.global_steps += steps
        if self.state.active_weapon_id is not None:
            self.state.weapon_steps[self.state.active_weapon_id] += steps

    def update(self, batch: PPOBatch) -> dict[str, float]:
        metrics = self.trainer.update(batch)
        return metrics

    def save(self, metrics: dict[str, float] | None = None) -> Path:
        if self.state.active_weapon_id is None:
            raise ValueError("cannot save before observing a weapon")
        return self.checkpoints.save(
            self.trainer.policy,
            weapon_id=self.state.active_weapon_id,
            training_steps=self.state.global_steps,
            optimizer=self.trainer.optimizer,
            metrics={
                **(metrics or {}),
                "global_steps": float(self.state.global_steps),
                "active_weapon_id": float(self.state.active_weapon_id),
            },
        )

    def switch_if_due(self, steps: int) -> Path | None:
        """Save at a scheduler boundary; hotbar selection remains external."""
        switch = self.scheduler.advance(steps)
        if switch is None:
            return None
        return self.save({"switched_from_slot": float(switch.previous_slot), "switched_to_slot": float(switch.next_slot)})

    def load_latest_weapon(self, weapon_id: int) -> dict[str, Any] | None:
        payload = self.checkpoints.load_latest(
            self.trainer.policy,
            weapon_id,
            optimizer=self.trainer.optimizer,
        )
        if payload is not None:
            self.state.active_weapon_id = weapon_id
            self.state.global_steps = max(
                self.state.global_steps,
                int(payload.get("training_steps", 0)),
            )
            self.state.weapon_steps[weapon_id] = int(payload.get("training_steps", 0))
        return payload
