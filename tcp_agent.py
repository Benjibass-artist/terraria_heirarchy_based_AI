import argparse
import json
import math
import socketserver
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from combat import (
    CombatObservationSpec,
    CombatRewardEvent,
    MeleeCombatPolicy,
    RolloutBuffer,
    compute_combat_reward,
)


@dataclass
class PendingTransition:
    grid: torch.Tensor
    player_features: torch.Tensor
    weapon_features: torch.Tensor
    enemy_type_ids: torch.Tensor
    enemy_features: torch.Tensor
    weapon_type_id: torch.Tensor
    actions: torch.Tensor
    log_probability: torch.Tensor
    value: torch.Tensor


class MeleeAgentSession:
    """One model session for one Terraria player connection."""

    def __init__(
        self,
        checkpoint: str | Path | None = None,
        device: str = "cpu",
        scripted: bool = False,
        scripted_phase_ticks: int = 300,
    ) -> None:
        self.spec = CombatObservationSpec()
        self.device = torch.device(device)
        self.policy = MeleeCombatPolicy(self.spec).to(self.device)
        if checkpoint is not None:
            state = torch.load(checkpoint, map_location=self.device, weights_only=True)
            self.policy.load_state_dict(state)
        self.policy.eval()
        self.scripted = scripted
        self.scripted_phase_ticks = scripted_phase_ticks
        self.scripted_tick = 0
        self.hidden_state: torch.Tensor | None = None
        self.pending: PendingTransition | None = None
        self.rollout_buffer = RolloutBuffer()

    def process_observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        if self.scripted:
            action = self._scripted_action()
            self.scripted_tick += 1
            return action

        grid, player_features, weapon_features, enemy_type_ids, enemy_features, weapon_type_id = self._encode_observation(observation)
        reward = compute_combat_reward(
            CombatRewardEvent(
                attack_attempted=observation.get("attack_attempted", False),
                attack_hit=observation.get("attack_hit", False),
                damage_taken=float(observation.get("damage_taken", 0.0)),
                enemies_killed=int(observation.get("enemies_killed", 0)),
                was_alive=observation.get("was_alive", True),
                is_alive=observation.get("is_alive", True),
            )
        )
        is_alive = bool(observation.get("is_alive", True))
        just_respawned = not bool(observation.get("was_alive", True)) and is_alive

        if just_respawned:
            self.hidden_state = None

        if self.pending is not None:
            self.rollout_buffer.add({
                "grid": self.pending.grid,
                "player_features": self.pending.player_features,
                "weapon_features": self.pending.weapon_features,
                "enemy_type_ids": self.pending.enemy_type_ids,
                "enemy_features": self.pending.enemy_features,
                "weapon_type_id": self.pending.weapon_type_id,
                "actions": self.pending.actions,
                "old_log_probabilities": self.pending.log_probability,
                "old_values": self.pending.value,
                "reward": reward,
                "done": not is_alive,
            })
            self.pending = None

        if not is_alive:
            self.hidden_state = None
            return self._no_op()

        with torch.no_grad():
            actions, log_probability, value, self.hidden_state = self.policy.sample_action(
                grid,
                player_features,
                weapon_features,
                hidden_state=self.hidden_state,
                enemy_type_ids=enemy_type_ids,
                enemy_features=enemy_features,
                weapon_type_id=weapon_type_id,
            )

        self.pending = PendingTransition(
            grid=grid,
            player_features=player_features,
            weapon_features=weapon_features,
            enemy_type_ids=enemy_type_ids,
            enemy_features=enemy_features,
            weapon_type_id=weapon_type_id,
            actions=actions,
            log_probability=log_probability,
            value=value,
        )
        return self._decode_action(actions[0])

    def _scripted_action(self) -> dict[str, Any]:
        cycle = self.scripted_tick // self.scripted_phase_ticks
        phase = cycle % 5
        aim_angle = (cycle % 8) * math.pi / 4.0
        aim_x = math.cos(aim_angle)
        aim_y = math.sin(aim_angle)
        auto_aim = cycle % 2 == 0
        action = {
            "left": False,
            "right": False,
            "jump": False,
            "down": False,
            "attack": True,
            "aim_x": aim_x,
            "aim_y": aim_y,
            "auto_aim": auto_aim,
            "target_slot": 0,
            "hotbar_slot": 0,
        }
        if phase == 0:
            action["right"] = True
            return action
        if phase == 1:
            action["right"] = True
            action["jump"] = True
            return action
        if phase == 2:
            action["left"] = True
            action["jump"] = True
            return action
        if phase == 3:
            action["left"] = True
            action["down"] = True
        return action

    def build_ppo_batch(self, last_value: float = 0.0):
        return self.rollout_buffer.to_batch(last_value=last_value)

    def clear_rollout(self) -> None:
        self.rollout_buffer.clear()

    def _encode_observation(
        self, observation: dict[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        flags = torch.tensor(
            observation["grid_flags"], dtype=torch.int64, device=self.device
        ).reshape(self.spec.grid_height, self.spec.grid_width)
        grid = torch.stack(
            [((flags & bit) != 0).to(torch.float32) for bit in (1, 2, 4, 8)],
            dim=0,
        )
        grid = torch.cat(
            (grid, torch.zeros(4, self.spec.grid_height, self.spec.grid_width, device=self.device)),
            dim=0,
        ).unsqueeze(0)
        player_features = torch.tensor(
            observation["player_features"], dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        weapon_features = torch.tensor(
            observation["weapon_features"], dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        enemy_type_ids = torch.tensor(
            observation["enemy_type_ids"], dtype=torch.long, device=self.device
        ).reshape(1, self.spec.grid_height, self.spec.grid_width)
        enemy_features = torch.zeros(
            1, self.spec.enemy_slots, self.spec.enemy_features,
            dtype=torch.float32, device=self.device,
        )
        enemies = sorted(
            observation.get("enemies", []),
            key=lambda enemy: enemy.get("distance", float("inf")),
        )
        for index, enemy in enumerate(enemies[:self.spec.enemy_slots]):
            enemy_features[0, index] = torch.tensor(
                [
                    min(max(float(enemy.get("type_id", 0)) / 512.0, 0.0), 1.0),
                    float(enemy.get("relative_x", 0.0)) / 40.0,
                    float(enemy.get("relative_y", 0.0)) / 22.5,
                    float(enemy.get("velocity_x", 0.0)) / 10.0,
                    float(enemy.get("velocity_y", 0.0)) / 10.0,
                    float(enemy.get("health_fraction", 0.0)),
                    float(enemy.get("line_of_sight", 0.0)),
                    float(enemy.get("threat", 0.0)),
                    float(enemy.get("distance", 0.0)) / 40.0,
                ],
                dtype=torch.float32,
                device=self.device,
            )
        weapon_type_id = torch.tensor(
            [min(max(int(observation.get("weapon_type_id", 0)), 0), self.spec.weapon_type_count - 1)],
            dtype=torch.long,
            device=self.device,
        )
        return grid, player_features, weapon_features, enemy_type_ids, enemy_features, weapon_type_id

    def _decode_action(self, action: torch.Tensor) -> dict[str, Any]:
        movement = action[:5].to(torch.bool).tolist()
        aim = action[5:7].clamp(-1.0, 1.0)
        auto_aim_index = 7
        target_start = auto_aim_index + 1
        target_end = target_start + self.spec.target_action_count
        hotbar_end = target_end + self.spec.hotbar_action_count
        return {
            "left": movement[0],
            "right": movement[1],
            "jump": movement[2],
            "down": movement[3],
            "attack": movement[4],
            "aim_x": float(aim[0].item()),
            "aim_y": float(aim[1].item()),
            "auto_aim": bool(action[auto_aim_index].item() >= 0.5),
            "target_slot": int(action[target_start:target_end].argmax().item()),
            "hotbar_slot": int(action[target_end:hotbar_end].argmax().item()),
        }

    @staticmethod
    def _no_op() -> dict[str, Any]:
        return {
            "left": False,
            "right": False,
            "jump": False,
            "down": False,
            "attack": False,
            "aim_x": 1.0,
            "aim_y": 0.0,
            "auto_aim": False,
            "target_slot": 0,
            "hotbar_slot": 0,
        }


class AgentRequestHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        session: MeleeAgentSession = self.server.session  # type: ignore[attr-defined]
        for raw_line in self.rfile:
            if not raw_line.strip():
                continue
            observation = json.loads(raw_line)
            action = session.process_observation(observation)
            self.wfile.write((json.dumps(action) + "\n").encode("utf-8"))
            self.wfile.flush()


def run_server(
    host: str,
    port: int,
    checkpoint: str | None,
    device: str,
    scripted: bool,
    scripted_phase_ticks: int,
) -> None:
    server = socketserver.ThreadingTCPServer((host, port), AgentRequestHandler)
    server.daemon_threads = True
    server.session = MeleeAgentSession(
        checkpoint=checkpoint,
        device=device,
        scripted=scripted,
        scripted_phase_ticks=scripted_phase_ticks,
    )  # type: ignore[attr-defined]
    mode = "scripted input test" if scripted else "policy inference"
    print(f"Melee agent listening on {host}:{port} ({mode})", flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--checkpoint")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--scripted", action="store_true")
    parser.add_argument("--scripted-phase-ticks", type=int, default=300)
    arguments = parser.parse_args()
    run_server(
        arguments.host,
        arguments.port,
        arguments.checkpoint,
        arguments.device,
        arguments.scripted,
        arguments.scripted_phase_ticks,
    )
