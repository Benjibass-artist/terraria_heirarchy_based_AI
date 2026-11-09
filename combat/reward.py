from dataclasses import dataclass


@dataclass(frozen=True)
class CombatRewardConfig:
    attack_miss: float = -10.0
    attack_hit: float = 5.0
    attack_not_ready: float = -5.0
    damage_taken_scale: float = 0.5
    enemy_kill: float = 25.0
    death: float = -20.0


@dataclass(frozen=True)
class CombatRewardEvent:
    attack_attempted: bool = False
    attack_hit: bool = False
    attack_not_ready: bool = False
    damage_taken: float = 0.0
    enemies_killed: int = 0
    was_alive: bool = True
    is_alive: bool = True


def compute_combat_reward(
    event: CombatRewardEvent,
    config: CombatRewardConfig | None = None,
) -> float:
    """Compute one transition reward from events resolved during that tick.

    Death is detected only on an alive-to-dead transition, so respawning with
    positive health does not apply the death penalty a second time.
    """
    config = config or CombatRewardConfig()
    reward = 0.0

    if event.attack_attempted:
        reward += config.attack_hit if event.attack_hit else config.attack_miss
    if event.attack_not_ready:
        reward += config.attack_not_ready
    reward -= max(event.damage_taken, 0.0) * config.damage_taken_scale
    reward += max(event.enemies_killed, 0) * config.enemy_kill
    if event.was_alive and not event.is_alive:
        reward += config.death
    return reward