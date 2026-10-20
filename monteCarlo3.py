import random
import itertools
from collections import Counter
from typing import List, Tuple, Optional


# --- ВСПОМОГАТЕЛЬНАЯ ФУНКЦИЯ ОЦЕНКИ ---

def evaluate_cup_utility(cup: List[int], goal_type: str, locked: List[int], target_val: Optional[int]) -> int:
    counts = Counter(cup)
    locked_counts = Counter(locked)

    if goal_type.startswith("Three ") or goal_type == "Four of a Kind":
        if target_val is None:
            return counts.most_common(1)[0][1] if counts else 0
        needed = (3 if goal_type.startswith("Three ") else 4) - locked_counts[target_val]
        return min(max(0, needed), counts[target_val])

    elif goal_type == "Small Straight":
        unique_locked = set(locked)
        straights = [{1, 2, 3, 4}, {2, 3, 4, 5}, {3, 4, 5, 6}]
        best_utility = 0
        for s in straights:
            missing = s - unique_locked
            utility = len(set(cup).intersection(missing))
            if utility > best_utility:
                best_utility = utility
        return best_utility
    return 0


# --- СТРАТЕГИИ ПОВЕДЕНИЯ ---

class BotStrategy:
    def choose_die(self, player: 'Player', cup: List[int], all_players: List['Player']) -> int: pass

    def evaluate_pass(self, player: 'Player', cup: List[int], receiver: 'Player') -> float: pass

    def wants_reroll(self, player: 'Player', cup: List[int]) -> bool: pass


class CooperativeBot(BotStrategy):
    def wants_reroll(self, player, cup):
        return evaluate_cup_utility(cup, player.goal_type, player.locked, player.target_val) == 0

    def choose_die(self, player, cup, all_players):
        if evaluate_cup_utility(cup, player.goal_type, player.locked, player.target_val) > 0:
            if player.goal_type == "Small Straight":
                for t in [2, 3, 4, 5, 1, 6]:
                    if t in cup and t not in set(player.locked): return t
            else:
                if player.target_val in cup: return player.target_val

        best_fallback, min_need = cup[0], float('inf')
        for die in set(cup):
            team_need = sum(evaluate_cup_utility([die], p.goal_type, p.locked, p.target_val)
                            for p in all_players if p.id != player.id)
            if team_need < min_need:
                min_need, best_fallback = team_need, die
        return best_fallback

    def evaluate_pass(self, player, cup, receiver):
        return evaluate_cup_utility(cup, receiver.goal_type, receiver.locked, receiver.target_val)


class SelfishBot(BotStrategy):
    def wants_reroll(self, player, cup):
        return evaluate_cup_utility(cup, player.goal_type, player.locked, player.target_val) == 0

    def choose_die(self, player, cup, all_players):
        if evaluate_cup_utility(cup, player.goal_type, player.locked, player.target_val) > 0:
            if player.goal_type.startswith("Three ") or player.goal_type == "Four of a Kind":
                if player.target_val in cup: return player.target_val
            elif player.goal_type == "Small Straight":
                for t in [2, 3, 4, 5, 1, 6]:
                    if t in cup and t not in player.locked: return t
        return random.choice(cup)

    def evaluate_pass(self, player, cup, receiver):
        return random.random()


class SaboteurBot(BotStrategy):
    def wants_reroll(self, player, cup):
        return sum(evaluate_cup_utility(cup, p.goal_type, p.locked, p.target_val) for p in player.all_players) > 2

    def choose_die(self, player, cup, all_players):
        worst_die, max_damage = cup[0], -1
        for die in set(cup):
            damage = sum(evaluate_cup_utility([die], p.goal_type, p.locked, p.target_val)
                         for p in all_players if p.id != player.id)
            if damage > max_damage:
                max_damage, worst_die = damage, die
        return worst_die

    def evaluate_pass(self, player, cup, receiver):
        return -evaluate_cup_utility(cup, receiver.goal_type, receiver.locked, receiver.target_val)


class RandomBot(BotStrategy):
    def wants_reroll(self, player, cup): return random.choice([True, False])

    def choose_die(self, player, cup, all_players): return random.choice(cup)

    def evaluate_pass(self, player, cup, receiver): return random.random()


# --- СИСТЕМА ИГРОКА ---

class Player:
    def __init__(self, player_id: int, goal_type: str, strategy_name: str):
        self.id = player_id
        self.goal_type = goal_type
        self.locked: List[int] = []
        self.target_val: Optional[int] = int(goal_type.split()[1][0]) if goal_type.startswith("Three ") else None

        strategies = {"coop": CooperativeBot(), "self": SelfishBot(), "sab": SaboteurBot(), "rnd": RandomBot()}
        self.strategy = strategies.get(strategy_name, CooperativeBot())
        self.all_players = []

    def check_victory(self) -> bool:
        counts = Counter(self.locked)
        if self.goal_type.startswith("Three "):
            return counts[self.target_val] >= 3
        elif self.goal_type == "Four of a Kind":
            return any(c >= 4 for c in counts.values())
        elif self.goal_type == "Small Straight":
            return any(s.issubset(set(self.locked)) for s in [{1, 2, 3, 4}, {2, 3, 4, 5}, {3, 4, 5, 6}])
        return False


# --- ДВИЖОК СИМУЛЯЦИИ ---

def run_single_game(playstyles: List[str], goals_to_test: List[str], derangements: List[Tuple]) -> Tuple[bool, int]:
    num_players = len(playstyles)
    assigned_goals = random.sample(goals_to_test, num_players)
    players = [Player(i, assigned_goals[i], playstyles[i]) for i in range(num_players)]
    for p in players: p.all_players = players

    cups = [[random.randint(1, 6) for _ in range(5)] for _ in range(num_players)]

    for p in players:
        if p.goal_type == "Four of a Kind" and p.target_val is None:
            counts = Counter(cups[p.id])
            p.target_val = counts.most_common(1)[0][0] if counts else 1

        if p.strategy.wants_reroll(p, cups[p.id]):
            cups[p.id] = [random.randint(1, 6) for _ in range(5)]
            if p.goal_type == "Four of a Kind":
                p.target_val = Counter(cups[p.id]).most_common(1)[0][0]

    for _ in range(5):
        for p in players:
            die = p.strategy.choose_die(p, cups[p.id], players)
            p.locked.append(die)
            cups[p.id].remove(die)

        best_perm, best_score = None, -float('inf')

        for perm in derangements:
            score = sum(
                players[i].strategy.evaluate_pass(players[i], cups[i], players[perm[i]]) for i in range(num_players))
            if score > best_score:
                best_score, best_perm = score, perm

        cups = [cups[giver_idx] for giver_idx in best_perm]

    individual_wins = sum(1 for p in players if p.check_victory())
    return individual_wins == num_players, individual_wins


# --- ГЕНЕРАТОР ТЕСТОВ ---

def format_team_composition(combo: Tuple[str, ...]) -> str:
    """Форматирует кортеж ('coop', 'coop', 'sab') в читаемую строку '2 Coop, 1 Sab'."""
    counts = Counter(combo)
    return ", ".join([f"{count} {role.title()}" for role, count in counts.items()])


def run_all_combinations(num_players: int = 4, iterations_per_comp: int = 500):
    goals = ["Three 1s", "Three 2s", "Three 3s", "Three 4s", "Three 5s", "Three 6s", "Small Straight", "Four of a Kind"]
    agendas = ["coop", "self", "sab", "rnd"]

    # Предрасчет валидных передач стакана (деранжирований) для ускорения
    derangements = [p for p in itertools.permutations(range(num_players)) if all(p[i] != i for i in range(num_players))]
    all_combos = list(itertools.combinations_with_replacement(agendas, num_players))

    print(f"=== АНАЛИЗ МЕТЫ: {num_players} ИГРОКОВ ===")
    print(f"Тестируем {len(all_combos)} уникальных составов (по {iterations_per_comp} игр на состав)\n")
    print(f"{'Состав Команды':<35} | {'Team Win Rate':<15} | {'Avg Indiv Win Rate'}")
    print("-" * 75)

    results_data = []

    for combo in all_combos:
        team_wins = 0
        total_indiv_wins = 0

        for _ in range(iterations_per_comp):
            # Перемешиваем состав, чтобы рассадка за столом была случайной
            shuffled_team = list(combo)
            random.shuffle(shuffled_team)

            t_win, ind_wins = run_single_game(shuffled_team, goals, derangements)
            if t_win: team_wins += 1
            total_indiv_wins += ind_wins

        team_win_rate = (team_wins / iterations_per_comp) * 100
        indiv_win_rate = (total_indiv_wins / (iterations_per_comp * num_players)) * 100

        results_data.append({
            "comp": format_team_composition(combo),
            "team_rate": team_win_rate,
            "indiv_rate": indiv_win_rate
        })

    # Сортировка по убыванию командного винрейта
    results_data.sort(key=lambda x: x["team_rate"], reverse=True)

    for res in results_data:
        print(f"{res['comp']:<35} | {res['team_rate']:>6.2f}%         | {res['indiv_rate']:>6.2f}%")


if __name__ == "__main__":
    # Легко переключается между 3, 4 или 5 игроками
    run_all_combinations(num_players=3, iterations_per_comp=10000)