# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This project trains reinforcement learning agents to play Pokemon Red using PyBoy (Game Boy emulator), Gymnasium, and Stable Baselines 3 (PPO). The agent learns by reading Game Boy RAM directly for reward signals.

**Important prerequisite:** A legally-obtained `PokemonRed.gb` ROM (SHA1: `ea9bcae617fdf159b045185467ae58b2e4a48b9a`) must be placed in the repository root. All scripts expect `../PokemonRed.gb` relative to their working directory (`v2/`).

## Active Version: `v2/`

**We are using `v2/`.** All development, training, and experimentation should target the `v2/` directory. The `baselines/` directory is the original version kept for reference only — do not modify or run it unless explicitly asked.

Key differences from `baselines/`:
- Uses coordinate-based exploration reward instead of KNN on screen frames (no hnswlib dependency)
- Richer Dict observation space: `screens`, `health`, `level` (Fourier-encoded), `badges`, `events`, `map`, `recent_actions`
- Requires PyBoy 2.4.0+ with updated API (`pyboy.screen.ndarray`, `pyboy.memory[addr]`)
- Streams agent coordinates to the global live map by default via `StreamWrapper`
- Checkpoints saved to `v2/runs/`

The two versions are **not interchangeable** — PyBoy API differences mean environment code cannot be freely swapped between them.

## Setup

Scripts must be run from within `v2/`, not from the repo root.

```bash
cd v2
pip install -r requirements.txt
# macOS: pip install -r macos_requirements.txt
```

## Key Commands

**Run pretrained model interactively:**
```bash
cd v2 && python run_pretrained_interactive.py
```
Toggle AI input at runtime by editing `agent_enabled.txt` (first line must start with "yes" to enable).

**Train from scratch:**
```bash
cd v2 && python baseline_fast_v2.py
```
Pass a checkpoint path via stdin to resume:
```bash
echo "runs/poke_26214400_steps" | python baseline_fast_v2.py
```

**Train continuously, always resuming the latest checkpoint:**
```bash
cd v2 && ./go_forever.sh
```

**Monitor training with TensorBoard:**
```bash
cd v2/runs && tensorboard --logdir .
# then open localhost:6006
```

## Architecture

### Gym Environment (`v2/red_gym_env_v2.py`)
Implements `gymnasium.Env`. The environment:
1. Loads a Game Boy save state (`.state` file) via PyBoy on `reset()`
2. Translates 7 discrete actions (down/left/right/up/A/B/start) to GameBoy button presses/releases
3. Reads RAM addresses directly for reward computation and observations

**Observation space** (Dict):
- `screens`: stacked grayscale frames `(72, 80, 3)`
- `health`: HP fraction `[0, 1]`
- `level`: Fourier-encoded level sum
- `badges`: 8-bit MultiBinary
- `events`: all event flag bits `(0xD747–0xD87E)`
- `map`: local explore map centered on agent `(48, 48, 1)`
- `recent_actions`: last 3 actions taken

**Reward components**: event flags set (`×4`), badges earned (`×10`), healing received (`×10`), unique coordinates explored (`×0.1`), stuck penalty (`−0.05` when revisiting a coord >600 times).

### RAM Reading
Addresses are inlined as hex literals in `red_gym_env_v2.py`. Key ones:
- `0xD35E` — map ID, `0xD361/0xD362` — Y/X position, `0xD356` — badges bitmask
- `0xD747–0xD87E` — event flags range, `0xD163` — party size
- `0xD18C, 0xD1B8, 0xD1E4, 0xD210, 0xD23C, 0xD268` — party Pokémon levels

### Exploration Strategy
Coordinate dict keyed by `"x:{x} y:{y} m:{map}"`. Reward is proportional to the number of unique coordinates visited. Coordinates are only tracked outside of battle (`0xD057 == 0`). The explore baseline resets once party level sum ≥ 22.

### Training Scripts
PPO via Stable Baselines 3 with `SubprocVecEnv` for parallelism. Key config:
- `num_cpu`: number of parallel environments (typically 64)
- `ep_length`: steps per episode (`2048 * 80`)
- `action_freq`: emulator ticks per agent action (default 24)
- Checkpoints saved to `v2/runs/poke_NNNN_steps.zip`

### Supporting Components
- **`stream_agent_wrapper.py`**: Gymnasium wrapper that streams `(x, y, map)` coordinates to `wss://transdimensional.xyz/broadcast` every 300 steps for the live map visualization
- **`tensorboard_callback.py`**: Custom SB3 callback logging per-episode stats, histograms, exploration maps, and event flags to TensorBoard
- **`global_map.py`** (v2): Converts local `(row, col, map_id)` coordinates to a unified global coordinate using `map_data.json` offsets
- **`events.json`**: Map of RAM address+bit keys to human-readable event flag names

### Save States
Pre-made `.state` files in the repo root control episode starting conditions:
- `init.state` — game start
- `has_pokedex.state` — after receiving Pokédex
- `has_pokedex_nballs.state` — after receiving Pokédex + Poké Balls (used by baselines interactive/parallel scripts)
- `fast_text_start.state` — fast text speed set

### Visualization
`visualization/` contains Jupyter notebooks and scripts for post-hoc analysis of agent trajectories. These are standalone and do not affect training.
