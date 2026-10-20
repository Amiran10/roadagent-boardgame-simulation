# RoadAgent Boardgame Simulation

A partially observable multi-agent game environment and reinforcement learning project built around a custom board game. The project trains a recurrent MAPPO policy for three-player team play using legal action masks, private observations, and centralized training with decentralized execution.

## Overview

This repository combines:
- a custom rules engine for a multi-agent dice game
- mission variants with different win conditions
- a recurrent MAPPO training loop
- evaluation and benchmarking scripts
- a terminal interface for human play
- smoke tests for validation

The emphasis is on learning under partial observability, action legality, and team coordination in a turn-based board-game setting.

## Why this project is interesting

- Multi-agent reinforcement learning in a structured game environment
- Hidden information and asymmetric player observations
- Centralized training, decentralized execution (CTDE)
- Legal action masking during policy optimization
- Mission-conditioned gameplay and team objectives
- Fully custom environment design instead of a generic toy benchmark

## Core project idea

Each player has a private goal, partial information about the board state, and a limited action space. The trained system learns to coordinate in a hidden-information setting while respecting game rules and mission constraints.

## Key files

- `road_agent.py` — rules engine, game mechanics, observations, and legality checks
- `missions.py` — mission definitions and custom win conditions
- `mappo.py` — recurrent MAPPO implementation, training, evaluation, model saving/loading
- `train_mappo.py` — training script for one or more mission-specific policies
- `benchmark.py` — comparison between MAPPO and a legal baseline strategy
- `play_game.py` — terminal-based human gameplay
- `test_smoke.py` — smoke tests for correctness and basic environment validation
- `LICENSE` — repository license

## Repository structure

```text
roadagent-boardgame-simulation/
├── README.md
├── LICENSE
├── requirements.txt
├── pyproject.toml
├── .gitignore
├── road_agent.py
├── missions.py
├── mappo.py
├── train_mappo.py
├── benchmark.py
├── play_game.py
├── test_smoke.py
├── models/                # generated during training
├── reports/               # generated during benchmarking
└── .venv/                 # local virtual environment
```

## Installation

```bash
git clone https://github.com/Amiran10/roadagent-boardgame-simulation.git
cd roadagent-boardgame-simulation
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Requirements

The project uses:
- Python 3.10+
- NumPy
- PyTorch

## Quick start

### Run the smoke test

```bash
python -m unittest -v test_smoke.py
```

### Play a human game in the terminal

```bash
python play_game.py
```

### Train a model

```bash
python train_mappo.py --mission standard --updates 120 --episodes-per-update 64 --imitation-episodes 256
```

### Benchmark MAPPO against a baseline

```bash
python benchmark.py --games 2000 --models-dir models --output reports/mappo_comparison.md
```

## Missions

The repository currently includes three mission rules:

1. `standard` — all secret goals must be met
2. `minimum_total` — each player must lock at least 15 total points, in addition to their goal
3. `team_sixes` — the team must lock at least three 6s, in addition to all goals

These mission definitions are implemented in `missions.py` and can be extended easily.

## Training details

The reinforcement learning policy uses:
- recurrent actor and critic networks
- action masks for legal moves only
- hidden-state temporal reasoning
- imitation warm-start from legal gameplay traces
- PPO-style policy updates

This setup supports centralized training while operating with decentralized execution at inference time.

## Evaluation

The project includes benchmarking utilities that compare:
- a legal decentralized heuristic baseline
- the trained MAPPO policy

This allows direct comparison of win rate, mission completion rate, and goal completion statistics.

## Notes

- The current code is structured for research and experimentation rather than production deployment.
- The repository is well suited for demonstrating reinforcement learning, game design, and multi-agent decision-making skills.
- Model files are generated under `models/` and benchmark reports under `reports/`.

## Future work

- refined mission design and game balancing
- policy training with stronger curriculum learning
- richer opponent strategies
- visual simulator / dashboard
- more benchmark reporting and ablation studies

## License

This project is licensed under the MIT license. See `LICENSE` for details.

## Contact

For questions or collaboration opportunities, please reach out through the repository owner profile on GitHub.
