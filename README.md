# Terraria Agent Tree Model

Initial melee-combat foundation for a Terraria AI written in PyTorch.

## Current model

`combat.MeleeCombatPolicy` is a CNN-GRU actor-critic intended for PPO training:

- CNN: encodes the 80x45 local tile grid.
- Enemy embedding: converts each enemy type ID into a learned spatial feature; ID `0` means no enemy.
- MLP: encodes player and weapon features.
- GRU: remembers recent movement, enemy, and projectile behavior.
- Actor head: independently controls left, right, jump, and attack, plus an 8-way aim direction and auto-aim choice for shortswords.
- Critic head: estimates the value of the current state for PPO.

`combat.PPOTrainer` provides the clipped PPO update step. It consumes rollout batches from an environment; the Terraria TCP rollout collector is the next integration layer.

## Python TCP bridge

Start the inference bridge with:

```powershell
python tcp_agent.py
```

It listens on `127.0.0.1:8765`, decodes the mod's JSON observations, preserves the GRU state, and returns movement, attack, and aim actions. Add `--checkpoint path\to\policy.pt` after a trained checkpoint exists. The bridge records transitions in memory; converting completed rollouts into PPO batches is the next training integration step.

For the first in-game control test, use scripted actions instead of the untrained policy:

```powershell
python tcp_agent.py --scripted
```

The scripted cycle tests right, right+jump, left+jump, down+left, and attack. It attacks during every phase so the test is visible sooner. Each phase lasts five seconds by default. Use `--scripted-phase-ticks 600` for ten-second phases. Scripted actions alternate `auto_aim` on and off, testing both closest-enemy targeting and learned aim directions. The observation also includes the closest enemy's normalized direction and distance so PPO can learn the same target relationship. The mod pauses AI controls and stops sending observations while the Terraria window is unfocused.

The reward module uses the requested values:

- `-10` for a completed melee swing with no hit.
- `+5` for a completed melee swing that hits.
- `-damage_taken / 2` for damage received.
- `+25` per enemy killed.
- `-20` only when the player transitions from alive to dead.

Respawning does not apply another death penalty because it is not an alive-to-dead transition.

## tModLoader bridge

The `mod/` directory contains the initial source to copy into a tModLoader mod source folder. It sends newline-delimited JSON observations to `127.0.0.1:8765` and reads movement, attack, and aim actions from the same connection. It includes:

- an 80x45 local tile grid
- platform and liquid flags
- enemy type IDs
- player and held-item features
- melee hit, damage, kill, and death-transition events

The C# source needs to be built by tModLoader, where the Terraria and XNA reference assemblies are available. It cannot be fully compiled as a standalone C# project from this workspace yet.

The model is deliberately shared-policy friendly. Future agents can reuse the observation and encoder contract while adding their own policy heads. Ranged weapons and cursor aiming are outside the first agent's scope. Later melee weapons that emit straight projectiles can be added as weapon descriptors without changing the movement policy.

## Quick check

```powershell
python -m tests.test_all
```

PyTorch must be installed in the selected Python environment.
