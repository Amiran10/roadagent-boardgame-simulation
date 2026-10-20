"""Train and save one recurrent MAPPO checkpoint for every established mission."""
from __future__ import annotations

import argparse
from pathlib import Path

from mappo import save_model, train
from missions import MISSIONS, get_mission


def main() -> None:
    parser = argparse.ArgumentParser(description="Train recurrent MAPPO policies for Road Agent.")
    parser.add_argument("--mission", default="all", help="Mission key or 'all'")
    parser.add_argument("--updates", type=int, default=120, help="PPO updates per mission")
    parser.add_argument("--episodes-per-update", type=int, default=64, help="Episodes collected per PPO update")
    parser.add_argument("--imitation-episodes", type=int, default=256, help="Legal-policy warm-start episodes")
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--device", default="cpu", help="Torch device, e.g. cpu or mps")
    args = parser.parse_args()
    selected = MISSIONS if args.mission == "all" else (get_mission(args.mission),)
    for index, mission in enumerate(selected):
        print(f"\nTraining {mission.name}: {mission.description}")
        actor, critic, history = train(mission.rule, seed=args.seed + index * 10_000, updates=args.updates, episodes_per_update=args.episodes_per_update, imitation_episodes=args.imitation_episodes, device=args.device)
        path = args.models_dir / f"mappo_{mission.key}.pt"
        save_model(path, actor, critic, mission_key=mission.key, seed=args.seed + index * 10_000, history=history)
        print(f"Saved model: {path.resolve()}")


if __name__ == "__main__":
    main()
