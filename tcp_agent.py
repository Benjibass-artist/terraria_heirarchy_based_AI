import argparse
import json
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
    ) -> None:
        self.spec = CombatObservationSpec()
        self.device = torch.device(device)
        self.policy = MeleeCombatPolicy(self.spec).to(self.device)
        if checkpoint is not None:
            state = torch.load(checkpoint, map_location=self.device, weights_only=True)
            self.policy.load_state_dict(state)
        self.policy.eval()
        self.scripted = scripted
        self.scripted_tick = 0
        self.hidden_state: torch.Tensor | None = None
        self.pending: PendingTransition | None = None
        self.rollout_buffer = RolloutBuffer()

    def process_observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        if self.scripted:
            action = self._scripted_action()
            self.scripted_tick += 1
            return action

        grid, player_features, weapon_features, enemy_type_ids = self._encode_observation(observation)
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
            )

        self.pending = PendingTransition(
            grid=grid,
            player_features=player_features,
            weapon_features=weapon_features,
            enemy_type_ids=enemy_type_ids,
            actions=actions,
            log_probability=log_probability,
            value=value,
        )
        return self._decode_action(actions[0])

    def _scripted_action(self) -> dict[str, Any]:
        phase = (self.scripted_tick // 60) % 5
        if phase == 0:
            return {"left": False, "right": True, "jump": False, "down": False, "attack": False, "aim_direction": 0}
        if phase == 1:
            return {"left": False, "right": True, "jump": True, "down": False, "attack": False, "aim_direction": 0}
        if phase == 2:
            return {"left": True, "right": False, "jump": True, "down": False, "attack": False, "aim_direction": 4}
        if phase == 3:
            return {"left": True, "right": False, "jump": False, "down": True, "attack": False, "aim_direction": 4}
        return {"left": False, "right": False, "jump": False, "down": False, "attack": True, "aim_direction": 0}

    def build_ppo_batch(self, last_value: float = 0.0):
        return self.rollout_buffer.to_batch(last_value=last_value)

    def clear_rollout(self) -> None:
        self.rollout_buffer.clear()

    def _encode_observation(
        self, observation: dict[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
        return grid, player_features, weapon_features, enemy_type_ids

    @staticmethod
    def _decode_action(action: torch.Tensor) -> dict[str, Any]:
        movement = action[:5].to(torch.bool).tolist()
        return {
            "left": movement[0],
            "right": movement[1],
            "jump": movement[2],
            "down": movement[3],
            "attack": movement[4],
            "aim_direction": int(action[5:].argmax().item()),
        }

    @staticmethod
    def _no_op() -> dict[str, Any]:
        return {
            "left": False,
            "right": False,
            "jump": False,
            "down": False,
            "attack": False,
            "aim_direction": 0,
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


def run_server(host: str, port: int, checkpoint: str | None, device: str, scripted: bool) -> None:
    server = socketserver.ThreadingTCPServer((host, port), AgentRequestHandler)
    server.daemon_threads = True
    server.session = MeleeAgentSession(checkpoint=checkpoint, device=device, scripted=scripted)  # type: ignore[attr-defined]
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
    arguments = parser.parse_args()
    run_server(arguments.host, arguments.port, arguments.checkpoint, arguments.device, arguments.scripted)
