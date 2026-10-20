"""Play one human Road Agent mission in the terminal."""
from __future__ import annotations

import random

from missions import get_mission
from road_agent import Phase, RoadAgentGame, RuleError, standard_goals


def main() -> None:
    mission = get_mission(input("Mission [standard/minimum_total/team_sixes] (standard): ").strip() or "standard")
    while True:
        players = tuple(name.strip() for name in input("Enter 3–5 names, separated by commas: ").split(",") if name.strip())
        if 3 <= len(players) <= 5 and len(set(players)) == len(players):
            break
        print("Please enter 3–5 unique names.")
    rng = random.Random()
    game = RoadAgentGame(players, dict(zip(players, rng.sample(standard_goals(), len(players)))), rng=rng, mission_rule=mission.rule)
    game.start()
    print(f"\n{mission.name}: {mission.description}")
    for player in players:
        input(f"\n{player}: press Enter to view your private information. ")
        obs = game.observation(player)
        print(f"Goal: {obs.own_goal.label()} | Cup: {list(obs.own_cup)}")
        if input("Mulligan? [y/N] ").strip().lower() in {"y", "yes"}:
            game.mulligan(player)
            print(f"New cup: {list(game.observation(player).own_cup)}")
        input("Hide the screen; press Enter. ")
    game.begin()
    while game.phase is not Phase.FINISHED:
        print(f"\n=== Round {game.round_number} ===")
        for player in players:
            input(f"{player}: press Enter for your cup. ")
            obs = game.observation(player)
            print(f"Goal: {obs.own_goal.label()} | Cup: {list(obs.own_cup)} | Public locks: {dict(obs.public_locked)}")
            while True:
                try:
                    die = game.lock(player, int(input(f"Lock die number 1–{len(obs.own_cup)}: ")) - 1)
                    print(f"Locked {die}.")
                    break
                except (ValueError, RuleError):
                    print("Invalid die position.")
            input("Hide the screen; press Enter. ")
        while True:
            recipients = {player: input(f"{player} passes to: ").strip() for player in players}
            try:
                game.pass_cups(recipients)
                break
            except RuleError as error:
                print(f"Invalid passing: {error}. Enter all destinations again.")
    assert game.result is not None
    print("\n=== Result ===")
    for player in players:
        print(f"{player}: {list(game.result.locked[player])}, {game.goals[player].label()} — {'SUCCESS' if game.result.goal_results[player] else 'FAILED'}")
    print("MISSION SUCCESS" if game.result.won else "MISSION FAILED")


if __name__ == "__main__":
    main()
