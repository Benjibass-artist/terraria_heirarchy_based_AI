from collections import deque
import random
from typing import Any


class ReplayBuffer:
    """Bounded replay storage for retaining experience from older weapons."""

    def __init__(self, capacity: int = 1_000_000) -> None:
        self.items: deque[dict[str, Any]] = deque(maxlen=capacity)

    def add(self, transition: dict[str, Any]) -> None:
        self.items.append(transition)

    def extend(self, transitions: list[dict[str, Any]]) -> None:
        self.items.extend(transitions)

    def sample(self, count: int) -> list[dict[str, Any]]:
        if count < 0 or count > len(self.items):
            raise ValueError("sample count must be between zero and the buffer size")
        return random.sample(list(self.items), count)

    def __len__(self) -> int:
        return len(self.items)


class WeaponReplayMixer:
    """Mix recent and historical weapon experience to reduce forgetting."""

    def __init__(self, capacity: int = 1_000_000, old_fraction: float = 0.3) -> None:
        if not 0.0 <= old_fraction <= 1.0:
            raise ValueError("old_fraction must be between zero and one")
        self.old_fraction = old_fraction
        self.old_experience = ReplayBuffer(capacity)

    def add_completed_rollout(self, transitions: list[dict[str, Any]]) -> None:
        self.old_experience.extend(transitions)

    def mix(
        self,
        recent_transitions: list[dict[str, Any]],
        batch_size: int,
    ) -> list[dict[str, Any]]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        old_count = min(int(batch_size * self.old_fraction), len(self.old_experience))
        recent_count = batch_size - old_count
        if recent_count > len(recent_transitions):
            recent_count = len(recent_transitions)
            old_count = min(batch_size - recent_count, len(self.old_experience))
        mixed = random.sample(recent_transitions, recent_count)
        mixed.extend(self.old_experience.sample(old_count))
        random.shuffle(mixed)
        return mixed
