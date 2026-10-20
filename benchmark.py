"""Compare MAPPO checkpoints with a legal decentralized heuristic under the same seed."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import random
import torch

from mappo import _baseline_mulligan, _goal_progress, _pass_mask, _play_actor, load_actor
from missions import MISSIONS
from road_agent import GoalKind, Phase, RoadAgentGame, standard_goals


def play_baseline(mission_rule, rng: random.Random):
    players = ("Player 1", "Player 2", "Player 3")
    goals = dict(zip(players, rng.sample(standard_goals(), 3)))
    game = RoadAgentGame(players, goals, rng=random.Random(rng.randrange(2**32)), mission_rule=mission_rule)
    game.start()
    for player in players:
        if _baseline_mulligan(game.observation(player)):
            game.mulligan(player)
    game.begin()
    while game.phase is not Phase.FINISHED:
        if game.phase is Phase.LOCK:
            for player in players:
                observation = game.observation(player)
                game.lock(player, _baseline_lock(observation))
        else:
            # Fixed leftward cycle is a legal decentralized pass convention.
            game.pass_cups({players[index]: players[(index + 1) % 3] for index in range(3)})
    assert game.result is not None
    return game.result, goals


def _baseline_lock(observation) -> int:
    dice, goal, locked = observation.own_cup, observation.own_goal, observation.own_locked
    if goal.kind is GoalKind.MATCH:
        return next((index for index, die in enumerate(dice) if die == goal.value), 0)
    if goal.kind is GoalKind.FOUR_KIND:
        from collections import Counter
        target = Counter(locked).most_common(1)[0][0] if locked else Counter(dice).most_common(1)[0][0]
        return next((index for index, die in enumerate(dice) if die == target), 0)
    return max(range(len(dice)), key=lambda index: _goal_progress(goal, (*locked, dice[index])))


def measure(policy, mission_rule, games: int, seed: int):
    rng = random.Random(seed)
    wins = mission_wins = 0
    attempts, successes = Counter(), Counter()
    for _ in range(games):
        result, goals = policy(mission_rule, rng)
        wins += result.won
        mission_wins += result.mission_satisfied
        for player, goal in goals.items():
            attempts[goal.label()] += 1
            successes[goal.label()] += result.goal_results[player]
    return wins / games, mission_wins / games, {goal: successes[goal] / attempts[goal] for goal in sorted(attempts)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Same-seed Road Agent baseline vs MAPPO report.")
    parser.add_argument("--games", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--output", type=Path, default=Path("reports/mappo_comparison.md"))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    lines = ["# Road Agent: MAPPO vs baseline", "", f"- Games per mission/policy: {args.games}", f"- Same evaluation seed per comparison: {args.seed}", "- Players: 3", "- MAPPO: shared GRU actor, centralized GRU critic during training, decentralized execution, legal action masks.", ""]
    for mission in MISSIONS:
        model_path = args.models_dir / f"mappo_{mission.key}.pt"
        if not model_path.exists():
            raise FileNotFoundError(f"Missing {model_path}. Run train_mappo.py first.")
        baseline = measure(play_baseline, mission.rule, args.games, args.seed)
        actor = load_actor(model_path, args.device)
        neural = measure(lambda rule, rng: _play_actor(actor, rule, rng, torch.device(args.device)), mission.rule, args.games, args.seed)
        lines += [f"## {mission.name}", "", mission.description, "", "| Metric | Goal-focused baseline | MAPPO | Difference |", "|---|---:|---:|---:|", f"| Team win rate | {baseline[0]:.1%} | {neural[0]:.1%} | {neural[0] - baseline[0]:+.1%} |", f"| Mission-rule success | {baseline[1]:.1%} | {neural[1]:.1%} | {neural[1] - baseline[1]:+.1%} |", "", "| Secret goal | Baseline | MAPPO | Difference |", "|---|---:|---:|---:|"]
        for goal in sorted(baseline[2]):
            lines.append(f"| {goal} | {baseline[2][goal]:.1%} | {neural[2][goal]:.1%} | {neural[2][goal] - baseline[2][goal]:+.1%} |")
        lines.append("")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote comparison report: {args.output.resolve()}")


if __name__ == "__main__":
    main()
