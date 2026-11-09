from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import torch
from torch import nn


class WeaponCheckpointStore:
    """Append-only checkpoint storage keyed by weapon ID.

    Every save creates a new file. Existing checkpoints are never replaced or
    removed, so training can stop and resume without losing prior progress.
    """

    def __init__(self, directory: str | Path = "checkpoints") -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        policy: nn.Module,
        weapon_id: int,
        training_steps: int,
        optimizer: torch.optim.Optimizer | None = None,
        metrics: dict[str, float] | None = None,
    ) -> Path:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        path = self.directory / f"weapon_{weapon_id}_{timestamp}_{uuid4().hex[:8]}.pt"
        payload: dict[str, Any] = {
            "weapon_id": int(weapon_id),
            "training_steps": int(training_steps),
            "saved_at_utc": timestamp,
            "policy_state": policy.state_dict(),
            "metrics": metrics or {},
        }
        if optimizer is not None:
            payload["optimizer_state"] = optimizer.state_dict()

        temporary_path = path.with_suffix(".tmp")
        torch.save(payload, temporary_path)
        temporary_path.replace(path)
        return path

    def latest(self, weapon_id: int) -> Path | None:
        matches = sorted(self.directory.glob(f"weapon_{int(weapon_id)}_*.pt"))
        return matches[-1] if matches else None

    def load_latest(
        self,
        policy: nn.Module,
        weapon_id: int,
        optimizer: torch.optim.Optimizer | None = None,
        device: str | torch.device = "cpu",
    ) -> dict[str, Any] | None:
        path = self.latest(weapon_id)
        if path is None:
            return None

        payload = torch.load(path, map_location=device, weights_only=False)
        policy.load_state_dict(payload["policy_state"])
        if optimizer is not None and "optimizer_state" in payload:
            optimizer.load_state_dict(payload["optimizer_state"])
        return payload
