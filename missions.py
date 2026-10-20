"""Established mission rules used before training and benchmarking."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

MissionRule = Callable[[Mapping[str, tuple[int, ...]]], bool]


@dataclass(frozen=True)
class Mission:
    key: str
    name: str
    description: str
    rule: MissionRule


MISSIONS = (
    Mission("standard", "Standard mission", "No extra rule: every player must satisfy their secret goal.", lambda _: True),
    Mission("minimum_total", "Minimum total", "Every player's five locked dice must total at least 15.", lambda pools: all(sum(dice) >= 15 for dice in pools.values())),
    Mission("team_sixes", "Team sixes", "The whole team must lock at least three 6s.", lambda pools: sum(die == 6 for dice in pools.values() for die in dice) >= 3),
)


def get_mission(key: str) -> Mission:
    for mission in MISSIONS:
        if mission.key == key:
            return mission
    raise KeyError(f"Unknown mission {key!r}. Choose one of: {', '.join(m.key for m in MISSIONS)}")
