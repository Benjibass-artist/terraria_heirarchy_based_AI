from dataclasses import dataclass


@dataclass(frozen=True)
class WeaponScheduleConfig:
    hotbar_slots: tuple[int, ...] = tuple(range(10))
    switch_interval_steps: int = 324_000


@dataclass(frozen=True)
class WeaponSwitch:
    previous_slot: int
    next_slot: int
    step: int


class WeaponTrainingScheduler:
    """External weapon rotation for training; it is not part of the policy."""

    def __init__(self, config: WeaponScheduleConfig | None = None) -> None:
        self.config = config or WeaponScheduleConfig()
        if not self.config.hotbar_slots:
            raise ValueError("at least one hotbar slot is required")
        if self.config.switch_interval_steps <= 0:
            raise ValueError("switch_interval_steps must be positive")
        self.slot_index = 0
        self.step = 0

    @property
    def current_slot(self) -> int:
        return self.config.hotbar_slots[self.slot_index]

    def advance(self, steps: int = 1) -> WeaponSwitch | None:
        if steps < 0:
            raise ValueError("steps cannot be negative")
        previous_step = self.step
        self.step += steps
        if self.step // self.config.switch_interval_steps == previous_step // self.config.switch_interval_steps:
            return None

        previous_slot = self.current_slot
        self.slot_index = (self.slot_index + 1) % len(self.config.hotbar_slots)
        return WeaponSwitch(previous_slot, self.current_slot, self.step)

    def reset(self, step: int = 0) -> None:
        if step < 0:
            raise ValueError("step cannot be negative")
        self.step = step
