# Terraria Agent Tree Model

Initial melee-combat foundation for a Terraria AI written in PyTorch.

## Current model

`combat.MeleeCombatPolicy` is a CNN-GRU actor-critic intended for PPO training:

- CNN: encodes the 80x45 local tile grid.
- Enemy embedding: converts each enemy type ID into a learned spatial feature; ID `0` means no enemy.
- MLP: encodes player and weapon features.
- GRU: remembers recent movement, enemy, and projectile behavior.
- Actor head: independently controls left, right, jump, and attack, plus continuous aim, auto-aim choice, and target selection. Hotbar changes are controlled only by the external scheduler.
- Attack head: masked until Terraria reports the held weapon is ready; the observation includes readiness, remaining item cooldown, and animation fraction.
- Critic head: estimates the value of the current state for PPO.

The external `WeaponTrainingScheduler` rotates all ten hotbar slots by training steps. At 60 observations per second, its default 324,000-step interval is approximately 1.5 hours. It wraps from slot 9 back to slot 0. It is orchestration code, not part of the neural network, so normal gameplay can use only one to three weapons without changing the policy.

`TrainingMetricsLogger` appends ten-minute weapon and overall point periods to `training_metrics.csv` and refreshes the combined `training_progress.png` plus one named graph per weapon in `training_progress_weapons/`. Each weapon graph has period reward and cumulative reward panels. Re-training a weapon adds new points without deleting earlier history. You can manually regenerate the graphs with `python plot_training.py`.

Periodic checkpoints default to every 60 PPO updates, plus saves on weapon changes and clean shutdown. Older random-named checkpoints are preserved; new checkpoints include the weapon name.

`WeaponTrainingManager` keeps the policy shared while tracking the active weapon and per-weapon training progress. It saves before scheduler transitions and loads the newest weapon checkpoint when resuming. Checkpoint files are append-only, so stopping and restarting training does not erase previous progress.

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

The scripted cycle tests right, right+jump, left+jump, down+left, and pulsed attacks. Use `--cycle-hotbar` to rotate through all ten hotbar slots. `--weapon-phase-ticks 1800` changes weapons every 30 seconds for testing; use `162000` for approximately 45 minutes at 60 observations per second. The observation includes explicit enemy records, and PPO can choose a target slot and hotbar slot. The mod pauses AI controls and stops sending observations while the Terraria window is unfocused.

The reward module uses the requested values:

- `-10` for a completed melee swing with no hit.
- `+5` for a completed melee swing that hits.
- `-damage_taken / 2` for damage received.
- `+25` per enemy killed.
- `-20` only when the player transitions from alive to dead.

Respawning does not apply another death penalty because it is not an alive-to-dead transition.

Misses are scored per completed weapon swing, not per frame while the attack button is held. A held attack therefore cannot accumulate `-10` every tick; Terraria's animation completion resolves one hit or miss event.

## tModLoader bridge

The `mod/` directory contains the initial source to copy into a tModLoader mod source folder. It sends newline-delimited JSON observations to `127.0.0.1:8765` and reads movement, attack, and aim actions from the same connection. It includes:

- an 80x45 local tile grid
- platform and liquid flags
- enemy type IDs
- player and held-item features
- enemy current HP, max HP, damage, defense, threat, velocity, line of sight, and distance
- melee hit, damage, kill, and death-transition events

The C# source needs to be built by tModLoader, where the Terraria and XNA reference assemblies are available. It cannot be fully compiled as a standalone C# project from this workspace yet.

The model is deliberately shared-policy friendly. Future agents can reuse the observation and encoder contract while adding their own policy heads. Ranged weapons and cursor aiming are outside the first agent's scope. Later melee weapons that emit straight projectiles can be added as weapon descriptors without changing the movement policy.

## Quick check

```powershell
python -m tests.test_all
```

PyTorch must be installed in the selected Python environment.
