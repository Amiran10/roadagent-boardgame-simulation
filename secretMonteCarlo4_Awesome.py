import random
import itertools
from collections import Counter
from typing import List, Tuple, Optional, Dict


# --- 1. FOG OF WAR & INFERENCE ENGINE ---

def infer_needed_dice(locked: List[int]) -> Dict[int, float]:
    """Infers probable dice needs (1-6) for a player based solely on their public locked dice."""
    needs = {i: 0.0 for i in range(1, 7)}
    if not locked:
        for i in [2, 3, 4, 5]:
            needs[i] = 0.5
        return needs

    counts = Counter(locked)
    most_common_val, max_count = counts.most_common(1)[0]
    unique_locked = set(locked)

    # Hypothesis 1: Collecting Sets (Three of a Kind / Four of a Kind)
    if max_count >= 1:
        needs[most_common_val] += max_count * 2.0

    # Hypothesis 2: Collecting Straights (Small Straight)
    if len(unique_locked) > 1:
        straights = [{1, 2, 3, 4}, {2, 3, 4, 5}, {3, 4, 5, 6}]
        for s in straights:
            overlap = len(s.intersection(unique_locked))
            if overlap >= 1:
                for missing in (s - unique_locked):
                    needs[missing] += overlap * 1.0
    return needs


def evaluate_self_utility(cup: List[int], goal_type: str, locked: List[int], target_val: Optional[int]) -> int:
    """Calculates private utility of the current cup for the player's own secret goal."""
    counts = Counter(cup)
    locked_counts = Counter(locked)

    if goal_type.startswith("Three ") or goal_type == "Four of a Kind":
        if target_val is None:
            return counts.most_common(1)[0][1] if counts else 0
        needed = (3 if goal_type.startswith("Three ") else 4) - locked_counts[target_val]
        return min(max(0, needed), counts[target_val])

    elif goal_type == "Small Straight":
        unique_locked = set(locked)
        best_util = 0
        for straight in [{1, 2, 3, 4}, {2, 3, 4, 5}, {3, 4, 5, 6}]:
            missing = straight - unique_locked
            util = len(set(cup).intersection(missing))
            if util > best_util:
                best_util = util
        return best_util
    return 0


# --- 2. BOT STRATEGIES ---

class BotStrategy:
    def wants_mulligan(self, player: 'Player', cup: List[int]) -> bool:
        raise NotImplementedError

    def choose_die(self, player: 'Player', cup: List[int], all_players: List['Player']) -> int:
        raise NotImplementedError

    def evaluate_pass(self, player: 'Player', cup: List[int], receiver: 'Player') -> float:
        raise NotImplementedError


class SmartCoopBot(BotStrategy):
    """Smart Cooperative Bot: Oxygen Mask rule on Mulligan, locks useful dice, passes optimal cups."""

    def wants_mulligan(self, player, cup) -> bool:
        return evaluate_self_utility(cup, player.goal_type, player.locked, player.target_val) == 0

    def choose_die(self, player, cup, all_players) -> int:
        valid_choices = []
        if player.goal_type.startswith("Three ") or player.goal_type == "Four of a Kind":
            if player.target_val in cup:
                valid_choices.append(player.target_val)
        elif player.goal_type == "Small Straight":
            for t in [1, 2, 3, 4, 5, 6]:
                if t in cup and t not in player.locked:
                    valid_choices.append(t)

        team_needs = {
            die: sum(infer_needed_dice(p.locked).get(die, 0) for p in all_players if p.id != player.id)
            for die in set(cup)
        }

        # Pick useful die for self that damages team the least
        if valid_choices:
            return min(valid_choices, key=lambda d: team_needs.get(d, 0))
        # Garbage collection: discard die least needed by teammates
        return min(set(cup), key=lambda d: team_needs.get(d, 0))

    def evaluate_pass(self, player, cup, receiver) -> float:
        needs = infer_needed_dice(receiver.locked)
        return sum(needs.get(die, 0) for die in cup)


class SelfishBot(BotStrategy):
    """Selfish Bot: Prioritizes own goal exclusively; ignores team needs."""

    def wants_mulligan(self, player, cup) -> bool:
        return evaluate_self_utility(cup, player.goal_type, player.locked, player.target_val) == 0

    def choose_die(self, player, cup, all_players) -> int:
        if player.goal_type.startswith("Three ") or player.goal_type == "Four of a Kind":
            if player.target_val in cup:
                return player.target_val
        elif player.goal_type == "Small Straight":
            for t in [2, 3, 4, 5, 1, 6]:
                if t in cup and t not in player.locked:
                    return t
        return random.choice(cup)

    def evaluate_pass(self, player, cup, receiver) -> float:
        return random.random()


class SaboteurBot(BotStrategy):
    """Saboteur Bot: Uses deduction to steal key dice and sabotage cup passing."""

    def wants_mulligan(self, player, cup) -> bool:
        inferred_team_util = sum(
            sum(infer_needed_dice(p.locked).get(d, 0) for d in cup)
            for p in player.all_players if p.id != player.id
        )
        return inferred_team_util > 3.0

    def choose_die(self, player, cup, all_players) -> int:
        team_needs = {
            die: sum(infer_needed_dice(p.locked).get(die, 0) for p in all_players if p.id != player.id)
            for die in set(cup)
        }
        return max(set(cup), key=lambda d: team_needs.get(d, 0))

    def evaluate_pass(self, player, cup, receiver) -> float:
        needs = infer_needed_dice(receiver.locked)
        return -sum(needs.get(die, 0) for die in cup)


class RandomBot(BotStrategy):
    """Random Bot: Unpredictable actions."""

    def wants_mulligan(self, player, cup) -> bool:
        return random.choice([True, False])

    def choose_die(self, player, cup, all_players) -> int:
        return random.choice(cup)

    def evaluate_pass(self, player, cup, receiver) -> float:
        return random.random()


# --- 3. GAME ENGINE ---

class Player:
    def __init__(self, player_id: int, goal_type: str, strategy_name: str):
        self.id = player_id
        self.goal_type = goal_type
        self.strategy_name = strategy_name
        self.locked: List[int] = []
        self.target_val: Optional[int] = int(goal_type.split()[1][0]) if goal_type.startswith("Three ") else None

        strategies = {
            "Coop": SmartCoopBot(),
            "Self": SelfishBot(),
            "Sab": SaboteurBot(),
            "Rnd": RandomBot()
        }
        self.strategy = strategies.get(strategy_name, SmartCoopBot())
        self.all_players: List['Player'] = []

    def check_victory(self) -> bool:
        counts = Counter(self.locked)
        if self.goal_type.startswith("Three "):
            return counts[self.target_val] >= 3
        elif self.goal_type == "Four of a Kind":
            return any(c >= 4 for c in counts.values())
        elif self.goal_type == "Small Straight":
            return any(s.issubset(set(self.locked)) for s in [{1, 2, 3, 4}, {2, 3, 4, 5}, {3, 4, 5, 6}])
        return False


def run_single_game(playstyles: List[str], goals: List[str], derangements: List[Tuple[int, ...]]) -> Tuple[
    bool, List[Tuple[str, bool]]]:
    num_players = len(playstyles)
    players = [Player(i, goals[i], playstyles[i]) for i in range(num_players)]
    for p in players:
        p.all_players = players

    cups = [[random.randint(1, 6) for _ in range(5)] for _ in range(num_players)]

    # PHASE 1: Mulligan (Initial reroll before any dice locking)
    for p in players:
        if p.goal_type == "Four of a Kind" and p.target_val is None:
            counts = Counter(cups[p.id])
            p.target_val = counts.most_common(1)[0][0] if counts else 1

        if p.strategy.wants_mulligan(p, cups[p.id]):
            cups[p.id] = [random.randint(1, 6) for _ in range(5)]
            if p.goal_type == "Four of a Kind":
                p.target_val = Counter(cups[p.id]).most_common(1)[0][0]

    # PHASE 2: 5 Sequential Rounds
    for _ in range(5):
        # Sequential Locking: Player 0 locks -> Player 1 sees Player 0's pick and locks -> ...
        for p in players:
            die = p.strategy.choose_die(p, cups[p.id], players)
            p.locked.append(die)
            cups[p.id].remove(die)

        # Passing Phase: Determine best cup assignment via blind voting evaluation
        best_perm, best_score = None, -float('inf')
        for perm in derangements:
            score = sum(
                players[j].strategy.evaluate_pass(players[j], cups[j], players[perm[j]]) for j in range(num_players))
            if score > best_score:
                best_score, best_perm = score, perm

        cups = [cups[giver_idx] for giver_idx in best_perm]

    results = [(p.goal_type, p.check_victory()) for p in players]
    team_win = all(success for _, success in results)
    return team_win, results


# --- 4. ANALYTICS ENGINE ---

def run_simulation(num_players: int = 4, iterations: int = 1500):
    goals_list = ["Three 1s", "Three 2s", "Three 3s", "Three 4s", "Three 5s", "Three 6s", "Small Straight",
                  "Four of a Kind"]
    agendas = ["Coop", "Self", "Rnd", "Sab"]
    derangements = [p for p in itertools.permutations(range(num_players)) if all(p[i] != i for i in range(num_players))]

    # REPORT 1: Goal Win Rates across Homogenous Teams
    print("=" * 80)
    print("REPORT 1: SECRET GOAL SUCCESS RATES (HOMOGENEOUS TEAMS WITH FOG OF WAR)")
    print("=" * 80)

    goal_stats = {agenda: {g: {"wins": 0, "attempts": 0} for g in goals_list} for agenda in agendas}

    for agenda in agendas:
        playstyles = [agenda] * num_players
        for _ in range(iterations):
            sampled_goals = random.choices(goals_list, k=num_players)
            _, results = run_single_game(playstyles, sampled_goals, derangements)
            for goal, success in results:
                goal_stats[agenda][goal]["attempts"] += 1
                if success:
                    goal_stats[agenda][goal]["wins"] += 1

    header1 = f"{'Secret Goal':<18} | {'All Coop':>10} | {'All Self':>10} | {'All Rnd':>10} | {'All Sab':>10}"
    print(header1)
    print("-" * len(header1))
    for goal in goals_list:
        rates = []
        for agenda in agendas:
            att = goal_stats[agenda][goal]["attempts"]
            w = goal_stats[agenda][goal]["wins"]
            rates.append((w / att * 100) if att > 0 else 0.0)
        print(f"{goal:<18} | {rates[0]:>9.2f}% | {rates[1]:>9.2f}% | {rates[2]:>9.2f}% | {rates[3]:>9.2f}%")

    # REPORT 2: Meta-Analysis of all 35 Team Combinations
    print("\n" + "=" * 80)
    print("REPORT 2: META-ANALYSIS OF ALL 35 PLAYER COMBINATIONS")
    print("=" * 80)

    all_combos = list(itertools.combinations_with_replacement(agendas, num_players))
    meta_results = []

    for combo in all_combos:
        team_wins, indiv_wins = 0, 0
        sim_count = int(iterations * 0.4)

        for _ in range(sim_count):
            shuffled_team = list(combo)
            random.shuffle(shuffled_team)
            sampled_goals = random.sample(goals_list, num_players)

            t_win, ind_results = run_single_game(shuffled_team, sampled_goals, derangements)
            if t_win:
                team_wins += 1
            indiv_wins += sum(1 for _, success in ind_results if success)

        t_rate = (team_wins / sim_count) * 100
        i_rate = (indiv_wins / (sim_count * num_players)) * 100

        counts = Counter(combo)
        comp_name = ", ".join([f"{c} {r}" for r, c in counts.items()])
        meta_results.append((comp_name, t_rate, i_rate))

    meta_results.sort(key=lambda x: x[1], reverse=True)

    header2 = f"{'Team Composition':<35} | {'Team Win Rate':>14} | {'Indiv Win Rate':>14}"
    print(header2)
    print("-" * len(header2))
    for comp, t_rate, i_rate in meta_results:
        print(f"{comp:<35} | {t_rate:>13.2f}% | {i_rate:>13.2f}%")


if __name__ == "__main__":
    run_simulation(num_players=3, iterations=10000)
# different number of players have different strategy ranking
# All coop players become more important with more players