"""A compact recurrent MAPPO implementation for the Road Agent rules engine.

Actors receive only legal player observations. The critic receives private global state during
training (CTDE). The shared actor has a GRU memory, action masks, and is saved with its critic.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
import random
from typing import Iterable, Mapping

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from road_agent import Goal, GoalKind, Phase, RoadAgentGame, Result, standard_goals

PLAYERS = 3
ACTION_COUNT = 5
GOAL_COUNT = 8
ACTOR_DIM = 49
CRITIC_DIM = 60
HIDDEN = 96


class Actor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gru = nn.GRUCell(ACTOR_DIM, HIDDEN)
        self.head = nn.Linear(HIDDEN, ACTION_COUNT)
        self.local_action_prior = nn.Linear(ACTION_COUNT, ACTION_COUNT, bias=False)
        nn.init.eye_(self.local_action_prior.weight)

    def forward(self, observation: torch.Tensor, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.gru(observation, hidden)
        # Inductive bias based solely on each actor's private observation. It gives PPO a
        # sensible starting convention while the recurrent network learns passing patterns.
        return self.head(hidden) + 3.0 * self.local_action_prior(observation[:, 39:44]), hidden


class Critic(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gru = nn.GRUCell(CRITIC_DIM, HIDDEN)
        self.head = nn.Linear(HIDDEN, 1)

    def forward(self, state: torch.Tensor, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.gru(state, hidden)
        return self.head(hidden).squeeze(-1), hidden


@dataclass
class Step:
    actor_obs: np.ndarray       # [players, actor_dim]
    masks: np.ndarray           # [players, actions]
    actions: np.ndarray         # [players]
    old_log_probs: np.ndarray   # [players]
    state: np.ndarray           # [critic_dim]
    value: float
    reward: float


@dataclass
class Episode:
    steps: list[Step]
    advantages: np.ndarray | None = None
    returns: np.ndarray | None = None


def train(
    mission_rule,
    *,
    seed: int,
    updates: int = 120,
    episodes_per_update: int = 64,
    ppo_epochs: int = 4,
    imitation_episodes: int = 256,
    learning_rate: float = 3e-4,
    device: str = "cpu",
) -> tuple[Actor, Critic, list[dict[str, float]]]:
    """Train a shared recurrent MAPPO policy for a fixed three-player mission."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = random.Random(seed)
    dev = torch.device(device)
    actor, critic = Actor().to(dev), Critic().to(dev)
    optimizer = torch.optim.Adam(chain(actor.parameters(), critic.parameters()), lr=learning_rate)
    history: list[dict[str, float]] = []

    imitation_loss = _behaviour_clone(actor, mission_rule, rng, imitation_episodes, dev)
    print(f"legal-policy imitation warm-start: loss={imitation_loss:.3f}")

    for update in range(1, updates + 1):
        episodes = [collect_episode(actor, critic, mission_rule, rng, dev, deterministic=False) for _ in range(episodes_per_update)]
        _add_gae(episodes)
        actor_loss, critic_loss, entropy = _ppo_update(actor, critic, optimizer, episodes, ppo_epochs, dev)
        if update == 1 or update % max(1, updates // 12) == 0:
            evaluation = evaluate(actor, mission_rule, games=300, seed=seed + update, device=dev)
            row = {"update": float(update), "team_win_rate": evaluation["team_win_rate"], "actor_loss": actor_loss, "critic_loss": critic_loss, "entropy": entropy}
            history.append(row)
            print(f"update {update:>4}/{updates}: win={row['team_win_rate']:.1%}, actor_loss={actor_loss:.3f}, critic_loss={critic_loss:.3f}")
    return actor, critic, history


def _behaviour_clone(actor: Actor, mission_rule, rng: random.Random, episodes: int, device: torch.device) -> float:
    traces = [_teacher_trace(mission_rule, rng) for _ in range(episodes)]
    optimizer = torch.optim.Adam(actor.parameters(), lr=8e-4)
    latest = 0.0
    for _ in range(10):
        optimizer.zero_grad()
        loss = 0.0
        for trace in traces:
            hidden = torch.zeros(PLAYERS, HIDDEN, device=device)
            for obs, masks, actions in trace:
                logits, hidden = actor(torch.as_tensor(obs, device=device), hidden)
                masked = logits.masked_fill(~torch.as_tensor(masks, dtype=torch.bool, device=device), -1e9)
                loss = loss + nn.functional.cross_entropy(masked, torch.as_tensor(actions, dtype=torch.long, device=device))
        latest = float((loss / (len(traces) * 10)).item())
        (loss / (len(traces) * 10)).backward()
        nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
        optimizer.step()
    return latest


def _teacher_trace(mission_rule, rng: random.Random):
    players = tuple(f"Player {index + 1}" for index in range(PLAYERS))
    goals = dict(zip(players, rng.sample(standard_goals(), PLAYERS)))
    game = RoadAgentGame(players, goals, rng=random.Random(rng.randrange(2**32)), mission_rule=mission_rule)
    game.start()
    for player in players:
        if _baseline_mulligan(game.observation(player)):
            game.mulligan(player)
    game.begin()
    trace = []
    while game.phase is not Phase.FINISHED:
        if game.phase is Phase.LOCK:
            masks = np.asarray([game.legal_lock_mask(player) for player in players], dtype=np.bool_)
            obs = np.asarray([_actor_features(game, player, np.zeros(ACTION_COUNT)) for player in players], dtype=np.float32)
            actions = np.asarray([_baseline_lock(game.observation(player)) for player in players])
            for player, action in zip(players, actions):
                game.lock(player, int(action))
            trace.append((obs, masks, actions))
        else:
            obs, masks, actions = [], [], []
            claimed: set[str] = set()
            for seat, player in enumerate(players):
                mask = _pass_mask(players, player, claimed)
                obs.append(_actor_features(game, player, mask))
                masks.append(mask)
                action = (seat + 1) % PLAYERS
                actions.append(action)
                claimed.add(players[action])
            game.pass_cups({sender: players[action] for sender, action in zip(players, actions)})
            trace.append((np.asarray(obs, dtype=np.float32), np.asarray(masks, dtype=np.bool_), np.asarray(actions)))
    return trace


def collect_episode(actor: Actor, critic: Critic, mission_rule, rng: random.Random, device: torch.device, *, deterministic: bool) -> Episode:
    players = tuple(f"Player {index + 1}" for index in range(PLAYERS))
    goals = dict(zip(players, rng.sample(standard_goals(), PLAYERS)))
    game = RoadAgentGame(players, goals, rng=random.Random(rng.randrange(2**32)), mission_rule=mission_rule)
    game.start()
    for player in players:
        if _baseline_mulligan(game.observation(player)):
            game.mulligan(player)
    game.begin()

    actor_hidden = torch.zeros(PLAYERS, HIDDEN, device=device)
    critic_hidden = torch.zeros(1, HIDDEN, device=device)
    steps: list[Step] = []
    previous_progress = _team_progress(game)
    while game.phase is not Phase.FINISHED:
        if game.phase is Phase.LOCK:
            masks = np.asarray([game.legal_lock_mask(player) for player in players], dtype=np.bool_)
            observations = np.asarray([_actor_features(game, player, np.zeros(ACTION_COUNT)) for player in players], dtype=np.float32)
            state = _critic_features(game, np.zeros(PLAYERS))
            actions, log_probs, actor_hidden = _select_actions(actor, observations, masks, actor_hidden, device, deterministic)
            with torch.no_grad():
                value_tensor, critic_hidden = critic(torch.as_tensor(state, device=device).unsqueeze(0), critic_hidden)
            for player, action in zip(players, actions):
                game.lock(player, int(action))
            progress = _team_progress(game)
            reward = 1.0 * (progress - previous_progress)
            previous_progress = progress
            steps.append(Step(observations, masks, actions, log_probs, state, float(value_tensor.item()), reward))
        else:
            observations, masks, actions, log_probs = [], [], [], []
            claimed: set[str] = set()
            state = _critic_features(game, np.zeros(PLAYERS))
            with torch.no_grad():
                value_tensor, critic_hidden = critic(torch.as_tensor(state, device=device).unsqueeze(0), critic_hidden)
            for seat, player in enumerate(players):
                mask = _pass_mask(players, player, claimed)
                obs = _actor_features(game, player, mask)
                action, log_prob, next_hidden = _select_actions(actor, obs[None, :], mask[None, :], actor_hidden[seat : seat + 1], device, deterministic)
                actor_hidden[seat] = next_hidden[0]
                receiver = players[int(action[0])]
                claimed.add(receiver)
                observations.append(obs)
                masks.append(mask)
                actions.append(action[0])
                log_probs.append(log_prob[0])
            game.pass_cups({sender: players[int(action)] for sender, action in zip(players, actions)})
            reward = 0.0
            if game.result is not None:
                reward += 5.0 if game.result.won else 0.0
            steps.append(Step(np.asarray(observations, dtype=np.float32), np.asarray(masks, dtype=np.bool_), np.asarray(actions), np.asarray(log_probs), state, float(value_tensor.item()), reward))
    return Episode(steps)


def evaluate(actor: Actor, mission_rule, *, games: int, seed: int, device: str | torch.device = "cpu") -> dict[str, object]:
    actor.eval()
    rng = random.Random(seed)
    wins, mission_wins = 0, 0
    goal_attempts: Counter[str] = Counter()
    goal_wins: Counter[str] = Counter()
    for _ in range(games):
        episode_result, assigned = _play_actor(actor, mission_rule, rng, torch.device(device))
        wins += episode_result.won
        mission_wins += episode_result.mission_satisfied
        for player, goal in assigned.items():
            goal_attempts[goal.label()] += 1
            goal_wins[goal.label()] += episode_result.goal_results[player]
    actor.train()
    return {
        "team_win_rate": wins / games,
        "mission_rate": mission_wins / games,
        "goal_rates": {goal: goal_wins[goal] / goal_attempts[goal] for goal in sorted(goal_attempts)},
    }


def _play_actor(actor: Actor, mission_rule, rng: random.Random, device: torch.device) -> tuple[Result, Mapping[str, Goal]]:
    players = tuple(f"Player {index + 1}" for index in range(PLAYERS))
    assigned = dict(zip(players, rng.sample(standard_goals(), PLAYERS)))
    game = RoadAgentGame(players, assigned, rng=random.Random(rng.randrange(2**32)), mission_rule=mission_rule)
    game.start()
    for player in players:
        if _baseline_mulligan(game.observation(player)):
            game.mulligan(player)
    game.begin()
    hidden = torch.zeros(PLAYERS, HIDDEN, device=device)
    while game.phase is not Phase.FINISHED:
        if game.phase is Phase.LOCK:
            masks = np.asarray([game.legal_lock_mask(player) for player in players], dtype=np.bool_)
            obs = np.asarray([_actor_features(game, player, np.zeros(ACTION_COUNT)) for player in players], dtype=np.float32)
            actions, _, hidden = _select_actions(actor, obs, masks, hidden, device, deterministic=True)
            for player, action in zip(players, actions):
                game.lock(player, int(action))
        else:
            chosen, claimed = [], set()
            for seat, player in enumerate(players):
                mask = _pass_mask(players, player, claimed)
                obs = _actor_features(game, player, mask)
                action, _, next_hidden = _select_actions(actor, obs[None, :], mask[None, :], hidden[seat : seat + 1], device, deterministic=True)
                hidden[seat] = next_hidden[0]
                chosen.append(int(action[0]))
                claimed.add(players[chosen[-1]])
            game.pass_cups({sender: players[action] for sender, action in zip(players, chosen)})
    assert game.result is not None
    return game.result, assigned


def save_model(path: Path, actor: Actor, critic: Critic, *, mission_key: str, seed: int, history: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"actor": actor.state_dict(), "critic": critic.state_dict(), "mission": mission_key, "seed": seed, "history": history, "format": "road-agent-mappo-v1"}, path)


def load_actor(path: Path, device: str = "cpu") -> Actor:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    actor = Actor().to(device)
    actor.load_state_dict(checkpoint["actor"])
    actor.eval()
    return actor


def _select_actions(actor: Actor, obs: np.ndarray, masks: np.ndarray, hidden: torch.Tensor, device: torch.device, deterministic: bool):
    with torch.no_grad():
        logits, hidden = actor(torch.as_tensor(obs, dtype=torch.float32, device=device), hidden)
        masked = logits.masked_fill(~torch.as_tensor(masks, dtype=torch.bool, device=device), -1e9)
        distribution = Categorical(logits=masked)
        actions = torch.argmax(masked, dim=-1) if deterministic else distribution.sample()
        return actions.cpu().numpy(), distribution.log_prob(actions).cpu().numpy(), hidden


def _ppo_update(actor: Actor, critic: Critic, optimizer, episodes: list[Episode], epochs: int, device: torch.device) -> tuple[float, float, float]:
    all_advantages = np.concatenate([episode.advantages for episode in episodes])
    mean, std = float(all_advantages.mean()), float(all_advantages.std() + 1e-8)
    actor_loss_value = critic_loss_value = entropy_value = 0.0
    for _ in range(epochs):
        optimizer.zero_grad()
        total_actor = total_critic = total_entropy = 0.0
        for episode in episodes:
            log_probs, entropies, values = _replay_episode(actor, critic, episode, device)
            old_log_probs = torch.as_tensor(np.stack([step.old_log_probs for step in episode.steps]), dtype=torch.float32, device=device)
            advantages = torch.as_tensor((episode.advantages - mean) / std, dtype=torch.float32, device=device).unsqueeze(1)
            returns = torch.as_tensor(episode.returns, dtype=torch.float32, device=device)
            ratio = torch.exp(log_probs - old_log_probs)
            clipped = torch.clamp(ratio, 0.8, 1.2)
            total_actor = total_actor + (-torch.minimum(ratio * advantages, clipped * advantages).mean())
            total_critic = total_critic + 0.5 * (returns - values).square().mean()
            total_entropy = total_entropy + entropies.mean()
        loss = (total_actor + 0.5 * total_critic - 0.01 * total_entropy) / len(episodes)
        loss.backward()
        nn.utils.clip_grad_norm_(chain(actor.parameters(), critic.parameters()), 0.5)
        optimizer.step()
        actor_loss_value, critic_loss_value, entropy_value = float((total_actor / len(episodes)).item()), float((total_critic / len(episodes)).item()), float((total_entropy / len(episodes)).item())
    return actor_loss_value, critic_loss_value, entropy_value


def _replay_episode(actor: Actor, critic: Critic, episode: Episode, device: torch.device):
    actor_hidden = torch.zeros(PLAYERS, HIDDEN, device=device)
    critic_hidden = torch.zeros(1, HIDDEN, device=device)
    log_probs, entropies, values = [], [], []
    for step in episode.steps:
        logits, actor_hidden = actor(torch.as_tensor(step.actor_obs, device=device), actor_hidden)
        masked = logits.masked_fill(~torch.as_tensor(step.masks, dtype=torch.bool, device=device), -1e9)
        distribution = Categorical(logits=masked)
        actions = torch.as_tensor(step.actions, dtype=torch.long, device=device)
        value, critic_hidden = critic(torch.as_tensor(step.state, device=device).unsqueeze(0), critic_hidden)
        log_probs.append(distribution.log_prob(actions))
        entropies.append(distribution.entropy())
        values.append(value.squeeze(0))
    return torch.stack(log_probs), torch.stack(entropies), torch.stack(values)


def _add_gae(episodes: Iterable[Episode], gamma: float = 0.99, lam: float = 0.95) -> None:
    for episode in episodes:
        rewards = np.asarray([step.reward for step in episode.steps], dtype=np.float32)
        values = np.asarray([step.value for step in episode.steps], dtype=np.float32)
        advantages = np.zeros_like(rewards)
        carry = 0.0
        for index in range(len(rewards) - 1, -1, -1):
            next_value = values[index + 1] if index + 1 < len(values) else 0.0
            carry = rewards[index] + gamma * next_value - values[index] + gamma * lam * carry
            advantages[index] = carry
        episode.advantages = advantages
        episode.returns = advantages + values


def _actor_features(game: RoadAgentGame, player: str, pass_mask: np.ndarray) -> np.ndarray:
    obs = game.observation(player)
    values = _pad(obs.own_cup) + _pad(obs.own_locked)
    for other in game.players:
        values += _pad(obs.public_locked[other])
    values += _goal_one_hot(obs.own_goal) + _one_hot(game.players.index(player), PLAYERS)
    values += [game.round_number / 5, float(game.phase is Phase.LOCK), float(game.phase is Phase.PASS)]
    values += [_goal_progress(obs.own_goal, (*obs.own_locked, die)) if game.phase is Phase.LOCK else 0.0 for die in obs.own_cup]
    values += [0.0] * (ACTION_COUNT - len(obs.own_cup))
    values += list(pass_mask)
    return np.asarray(values, dtype=np.float32)


def _critic_features(game: RoadAgentGame, claimed: np.ndarray) -> np.ndarray:
    cups, locked, goals = game.private_training_state()
    values: list[float] = []
    for player in game.players:
        values += _pad(cups[player])
    for player in game.players:
        values += _pad(locked[player])
    for player in game.players:
        values += _goal_one_hot(goals[player])
    values += [game.round_number / 5, float(game.phase is Phase.LOCK), float(game.phase is Phase.PASS)]
    values += list(claimed)
    return np.asarray(values, dtype=np.float32)


def _pad(values) -> list[float]:
    return [float(value) / 6 for value in values] + [0.0] * (5 - len(values))


def _goal_one_hot(goal: Goal) -> list[float]:
    index = goal.value - 1 if goal.kind is GoalKind.MATCH else 6 if goal.kind is GoalKind.STRAIGHT else 7
    return _one_hot(index, GOAL_COUNT)


def _one_hot(index: int, size: int) -> list[float]:
    return [float(position == index) for position in range(size)]


def _pass_mask(players: tuple[str, ...], sender: str, claimed: set[str]) -> np.ndarray:
    remaining_senders = players[players.index(sender) :]
    mask = np.zeros(ACTION_COUNT, dtype=np.bool_)
    for index, receiver in enumerate(players):
        if receiver != sender and receiver not in claimed and _can_complete(remaining_senders[1:], set(players) - claimed - {receiver}):
            mask[index] = True
    return mask


def _can_complete(senders: tuple[str, ...], available: set[str]) -> bool:
    if not senders:
        return not available
    sender = senders[0]
    return any(receiver != sender and _can_complete(senders[1:], available - {receiver}) for receiver in available)


def _team_progress(game: RoadAgentGame) -> float:
    return sum(_goal_progress(game.goals[player], game.observation(player).own_locked) for player in game.players)


def _goal_progress(goal: Goal, dice) -> float:
    if goal.kind is GoalKind.MATCH:
        return min(dice.count(goal.value), 3) / 3
    if goal.kind is GoalKind.FOUR_KIND:
        return (max(Counter(dice).values()) if dice else 0) / 4
    values = set(dice)
    return max(len(values & set(run)) for run in ((1, 2, 3, 4), (2, 3, 4, 5), (3, 4, 5, 6))) / 4


def _baseline_mulligan(obs) -> bool:
    if obs.own_goal.kind is GoalKind.MATCH:
        return obs.own_cup.count(obs.own_goal.value) == 0
    if obs.own_goal.kind is GoalKind.FOUR_KIND:
        return max(Counter(obs.own_cup).values()) < 2
    return _goal_progress(obs.own_goal, obs.own_cup) < 0.5


def _baseline_lock(obs) -> int:
    dice, goal, locked = obs.own_cup, obs.own_goal, obs.own_locked
    if goal.kind is GoalKind.MATCH:
        return next((index for index, die in enumerate(dice) if die == goal.value), 0)
    if goal.kind is GoalKind.FOUR_KIND:
        target = Counter(locked).most_common(1)[0][0] if locked else Counter(dice).most_common(1)[0][0]
        return next((index for index, die in enumerate(dice) if die == target), 0)
    return max(range(len(dice)), key=lambda index: _goal_progress(goal, (*locked, dice[index])))
