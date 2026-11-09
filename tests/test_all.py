import json
import socket
import socketserver
import threading

import torch

from combat import (
    CombatObservationSpec,
    CombatRewardEvent,
    MeleeCombatPolicy,
    PPOBatch,
    PPOConfig,
    PPOTrainer,
    compute_combat_reward,
)
from combat.rollout import RolloutBuffer
from tcp_agent import AgentRequestHandler, MeleeAgentSession


def test_model_and_rewards() -> None:
    spec = CombatObservationSpec()
    policy = MeleeCombatPolicy(spec)
    grid = torch.zeros(2, spec.grid_channels, spec.grid_height, spec.grid_width)
    player = torch.zeros(2, spec.player_features)
    weapon = torch.zeros(2, spec.weapon_features)
    enemy_types = torch.zeros(2, spec.grid_height, spec.grid_width, dtype=torch.long)
    enemy_types[:, 20, 40] = 7

    actions, log_probs, values, hidden = policy.sample_action(
        grid, player, weapon, enemy_type_ids=enemy_types
    )
    assert actions.shape == (
        2,
        spec.movement_action_count + spec.aim_vector_size + spec.auto_aim_action_count + spec.target_action_count + spec.hotbar_action_count,
    )
    assert log_probs.shape == (2,)
    assert values.shape == (2,)
    assert hidden.shape == (spec.gru_layers, 2, spec.hidden_size)
    assert compute_combat_reward(
        CombatRewardEvent(attack_attempted=True, attack_hit=True, damage_taken=4, enemies_killed=1, was_alive=True, is_alive=False)
    ) == 8.0
    assert compute_combat_reward(CombatRewardEvent(was_alive=False, is_alive=True)) == 0.0


def test_ppo_and_rollout() -> None:
    torch.manual_seed(7)
    spec = CombatObservationSpec()
    policy = MeleeCombatPolicy(spec)
    trainer = PPOTrainer(policy, PPOConfig(update_epochs=2))
    buffer = RolloutBuffer()

    for _ in range(8):
        grid = torch.randn(1, spec.grid_channels, spec.grid_height, spec.grid_width)
        player = torch.randn(1, spec.player_features)
        weapon = torch.randn(1, spec.weapon_features)
        enemy_types = torch.randint(0, spec.enemy_type_count, (1, spec.grid_height, spec.grid_width))
        with torch.no_grad():
            actions, old_log_probs, values, _ = policy.sample_action(
                grid, player, weapon, enemy_type_ids=enemy_types
            )
        buffer.add({
            "grid": grid,
            "player_features": player,
            "weapon_features": weapon,
            "enemy_type_ids": enemy_types,
            "enemy_features": torch.randn(1, spec.enemy_slots, spec.enemy_features),
            "weapon_type_id": torch.tensor([7], dtype=torch.long),
            "actions": actions,
            "old_log_probabilities": old_log_probs,
            "old_values": values,
            "reward": 1.0,
            "done": False,
        })

    batch = buffer.to_batch()
    before = next(policy.parameters()).detach().clone()
    metrics = trainer.update(batch)
    after = next(policy.parameters()).detach()
    assert torch.isfinite(torch.tensor(list(metrics.values()))).all()
    assert not torch.equal(before, after)


def test_tcp_protocol() -> None:
    class TestServer(socketserver.ThreadingTCPServer):
        allow_reuse_address = True

    session = MeleeAgentSession()
    server = TestServer(("127.0.0.1", 0), AgentRequestHandler)
    server.session = session
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    grid_size = 80 * 45
    observation = {
        "tick": 1,
        "grid_flags": [0] * grid_size,
        "enemy_type_ids": [0] * grid_size,
        "player_features": [0.0] * 16,
        "weapon_features": [0.0] * 16,
        "was_alive": True,
        "is_alive": True,
    }
    observation["grid_flags"][40 * 80 + 45] = 8
    observation["enemy_type_ids"][40 * 80 + 45] = 7

    with socket.create_connection(server.server_address) as connection:
        connection_file = connection.makefile("rwb")
        connection_file.write((json.dumps(observation) + "\n").encode("utf-8"))
        connection_file.flush()
        connection.shutdown(socket.SHUT_WR)
        action = json.loads(connection_file.readline())

    server.shutdown()
    server.server_close()
    thread.join(timeout=1)
    assert set(action) == {
        "left", "right", "jump", "down", "attack", "aim_x", "aim_y",
        "auto_aim", "target_slot", "hotbar_slot",
    }
    assert -1 <= action["aim_x"] <= 1
    assert -1 <= action["aim_y"] <= 1


if __name__ == "__main__":
    test_model_and_rewards()
    test_ppo_and_rollout()
    test_tcp_protocol()
    print("all tests passed")
