"""Rules engine for Road Agent. Actor observations never contain other cups or goals."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import Enum
import random
from typing import Callable, Mapping, Sequence

PLAYERS = range(3, 6)


class RuleError(ValueError):
    pass


class Phase(str, Enum):
    SETUP = "setup"
    MULLIGAN = "mulligan"
    LOCK = "lock"
    PASS = "pass"
    FINISHED = "finished"


class GoalKind(str, Enum):
    MATCH = "match"
    STRAIGHT = "straight"
    FOUR_KIND = "four_kind"


@dataclass(frozen=True)
class Goal:
    kind: GoalKind
    value: int | None = None

    @classmethod
    def three(cls, value: int) -> "Goal":
        return cls(GoalKind.MATCH, value)

    @classmethod
    def straight(cls) -> "Goal":
        return cls(GoalKind.STRAIGHT)

    @classmethod
    def four_kind(cls) -> "Goal":
        return cls(GoalKind.FOUR_KIND)

    def label(self) -> str:
        if self.kind is GoalKind.MATCH:
            return f"at least 3 {self.value}s"
        return "small straight" if self.kind is GoalKind.STRAIGHT else "four of a kind"

    def met(self, dice: Sequence[int]) -> bool:
        if len(dice) != 5:
            return False
        counts = Counter(dice)
        if self.kind is GoalKind.MATCH:
            return counts[self.value] >= 3
        if self.kind is GoalKind.FOUR_KIND:
            return max(counts.values()) >= 4
        values = set(dice)
        return any(set(run) <= values for run in ((1, 2, 3, 4), (2, 3, 4, 5), (3, 4, 5, 6)))


@dataclass(frozen=True)
class Observation:
    player: str
    phase: Phase
    round_number: int
    own_cup: tuple[int, ...]
    own_locked: tuple[int, ...]
    own_goal: Goal
    public_locked: Mapping[str, tuple[int, ...]]
    mulligan_available: bool


@dataclass(frozen=True)
class Result:
    won: bool
    goal_results: Mapping[str, bool]
    mission_satisfied: bool
    locked: Mapping[str, tuple[int, ...]]


MissionRule = Callable[[Mapping[str, tuple[int, ...]]], bool]


class RoadAgentGame:
    def __init__(self, players: Sequence[str], goals: Mapping[str, Goal], *, rng: random.Random, mission_rule: MissionRule) -> None:
        self.players = tuple(players)
        if len(self.players) not in PLAYERS or len(set(self.players)) != len(self.players) or set(goals) != set(self.players):
            raise RuleError("Need 3–5 unique players and exactly one goal per player.")
        self.goals, self.rng, self.mission_rule = dict(goals), rng, mission_rule
        self.phase, self.round_number = Phase.SETUP, 0
        self._cups: dict[str, list[int]] = {}
        self._locked: dict[str, list[int]] = {player: [] for player in self.players}
        self._locked_this_round: set[str] = set()
        self._mulligans: set[str] = set()
        self.result: Result | None = None

    def start(self) -> None:
        self._phase(Phase.SETUP)
        self._cups = {player: self._roll(5) for player in self.players}
        self.phase = Phase.MULLIGAN

    def mulligan(self, player: str) -> None:
        self._phase(Phase.MULLIGAN)
        self._player(player)
        if player in self._mulligans:
            raise RuleError("Mulligan already used.")
        self._cups[player] = self._roll(5)
        self._mulligans.add(player)

    def begin(self) -> None:
        self._phase(Phase.MULLIGAN)
        self.round_number, self.phase = 1, Phase.LOCK

    def lock(self, player: str, index: int) -> int:
        self._phase(Phase.LOCK)
        self._player(player)
        if player in self._locked_this_round or not isinstance(index, int) or index < 0 or index >= len(self._cups[player]):
            raise RuleError("Illegal lock action.")
        die = self._cups[player].pop(index)
        self._locked[player].append(die)
        self._locked_this_round.add(player)
        if len(self._locked_this_round) == len(self.players):
            self.phase = Phase.PASS
        return die

    def pass_cups(self, recipients: Mapping[str, str]) -> None:
        self._phase(Phase.PASS)
        if set(recipients) != set(self.players) or set(recipients.values()) != set(self.players):
            raise RuleError("Passing must give each player exactly one cup.")
        if any(sender == recipients[sender] for sender in self.players):
            raise RuleError("No player may keep their cup.")
        old = self._cups
        self._cups = {recipient: old[sender] for sender, recipient in recipients.items()}
        self._locked_this_round.clear()
        if self.round_number == 5:
            frozen = {player: tuple(self._locked[player]) for player in self.players}
            goals = {player: self.goals[player].met(frozen[player]) for player in self.players}
            mission = bool(self.mission_rule(frozen))
            self.result = Result(all(goals.values()) and mission, goals, mission, frozen)
            self.phase = Phase.FINISHED
        else:
            self.round_number += 1
            self.phase = Phase.LOCK

    def observation(self, player: str) -> Observation:
        self._player(player)
        if self.phase is Phase.SETUP:
            raise RuleError("Game has not started.")
        return Observation(player, self.phase, self.round_number, tuple(self._cups[player]), tuple(self._locked[player]), self.goals[player], {p: tuple(self._locked[p]) for p in self.players}, self.phase is Phase.MULLIGAN and player not in self._mulligans)

    def legal_lock_mask(self, player: str, max_actions: int = 5) -> list[bool]:
        self._player(player)
        return [self.phase is Phase.LOCK and player not in self._locked_this_round and i < len(self._cups[player]) for i in range(max_actions)]

    def private_training_state(self) -> tuple[Mapping[str, tuple[int, ...]], Mapping[str, tuple[int, ...]], Mapping[str, Goal]]:
        """CTDE-only state. Never send this method's data to an actor at execution time."""
        return ({p: tuple(self._cups[p]) for p in self.players}, {p: tuple(self._locked[p]) for p in self.players}, dict(self.goals))

    def _roll(self, count: int) -> list[int]:
        return [self.rng.randint(1, 6) for _ in range(count)]

    def _phase(self, phase: Phase) -> None:
        if self.phase is not phase:
            raise RuleError(f"Expected {phase.value}, got {self.phase.value}.")

    def _player(self, player: str) -> None:
        if player not in self.players:
            raise RuleError("Unknown player.")


def standard_goals() -> tuple[Goal, ...]:
    return tuple(Goal.three(value) for value in range(1, 7)) + (Goal.straight(), Goal.four_kind())
