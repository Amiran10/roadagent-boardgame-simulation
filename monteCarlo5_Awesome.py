import random
import itertools
from collections import Counter, defaultdict
from typing import List, Tuple, Optional, Dict


# --- 1. ЯДРО ОЦЕНКИ ПОЛЕЗНОСТИ ---

def evaluate_cup_utility(cup: List[int], goal_type: str, locked: List[int], target_val: Optional[int]) -> int:
    """Возвращает точное количество кубиков в стакане, которые продвигают игрока к цели."""
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


# --- 2. СТРАТЕГИИ ИГРОКОВ (АГЕНТЫ) ---

class BotStrategy:
    def choose_die(self, player: 'Player', cup: List[int], all_players: List['Player']) -> int: pass

    def evaluate_pass(self, player: 'Player', cup: List[int], receiver: 'Player') -> float: pass

    def wants_reroll(self, player: 'Player', cup: List[int]) -> bool: pass


class MasterCoopBot(BotStrategy):
    """Улучшенный кооперативный ИИ. Максимизирует синергию команды."""

    def wants_reroll(self, player, cup):
        # Перебрасывает, только если кубики бесполезны и для него, и слабо полезны для команды
        my_util = evaluate_cup_utility(cup, player.goal_type, player.locked, player.target_val)
        team_util = sum(evaluate_cup_utility(cup, p.goal_type, p.locked, p.target_val) for p in player.all_players if
                        p.id != player.id)
        return my_util == 0 and team_util <= 1

    def choose_die(self, player, cup, all_players):
        valid_choices = []
        # Определяем полезные для себя кубики
        if player.goal_type.startswith("Three ") or player.goal_type == "Four of a Kind":
            if player.target_val in cup: valid_choices.append(player.target_val)
        elif player.goal_type == "Small Straight":
            for t in [1, 2, 3, 4, 5, 6]:
                if t in cup and t not in player.locked:
                    valid_choices.append(t)

        # Если есть полезные кубики, берем тот, который МЕНЬШЕ всего нужен команде
        if valid_choices:
            best_choice = valid_choices[0]
            min_team_damage = float('inf')
            for die in valid_choices:
                damage = sum(evaluate_cup_utility([die], p.goal_type, p.locked, p.target_val) for p in all_players if
                             p.id != player.id)
                if damage < min_team_damage:
                    min_team_damage = damage
                    best_choice = die
            return best_choice

        # Сброс балласта: если мне ничего не нужно, забираю кубик-мусор, не нужный команде
        best_fallback, min_need = cup[0], float('inf')
        for die in set(cup):
            team_need = sum(evaluate_cup_utility([die], p.goal_type, p.locked, p.target_val) for p in all_players if
                            p.id != player.id)
            if team_need < min_need:
                min_need, best_fallback = team_need, die
        return best_fallback

    def evaluate_pass(self, player, cup, receiver):
        return evaluate_cup_utility(cup, receiver.goal_type, receiver.locked, receiver.target_val)


class SelfishBot(BotStrategy):
    """Эгоист. Думает только о своей цели, остальное делает случайно."""

    def wants_reroll(self, player, cup):
        return evaluate_cup_utility(cup, player.goal_type, player.locked, player.target_val) == 0

    def choose_die(self, player, cup, all_players):
        if player.goal_type.startswith("Three ") or player.goal_type == "Four of a Kind":
            if player.target_val in cup: return player.target_val
        elif player.goal_type == "Small Straight":
            for t in [2, 3, 4, 5, 1, 6]:
                if t in cup and t not in player.locked: return t
        return random.choice(cup)

    def evaluate_pass(self, player, cup, receiver):
        return random.random()


class SaboteurBot(BotStrategy):
    """Саботажник. Целенаправленно рушит игру команды."""

    def wants_reroll(self, player, cup):
        return sum(evaluate_cup_utility(cup, p.goal_type, p.locked, p.target_val) for p in player.all_players) > 2

    def choose_die(self, player, cup, all_players):
        worst_die, max_damage = cup[0], -1
        for die in set(cup):
            damage = sum(evaluate_cup_utility([die], p.goal_type, p.locked, p.target_val) for p in all_players if
                         p.id != player.id)
            if damage > max_damage:
                max_damage, worst_die = damage, die
        return worst_die

    def evaluate_pass(self, player, cup, receiver):
        return -evaluate_cup_utility(cup, receiver.goal_type, receiver.locked, receiver.target_val)


class RandomBot(BotStrategy):
    """Хаотик. Полный рандом."""

    def wants_reroll(self, player, cup): return random.choice([True, False])

    def choose_die(self, player, cup, all_players): return random.choice(cup)

    def evaluate_pass(self, player, cup, receiver): return random.random()


# --- 3. ИГРОК И ДВИЖОК ---

class Player:
    def __init__(self, player_id: int, goal_type: str, strategy_name: str):
        self.id = player_id
        self.goal_type = goal_type
        self.locked = []
        self.target_val = int(goal_type.split()[1][0]) if goal_type.startswith("Three ") else None

        strategies = {"coop": MasterCoopBot(), "self": SelfishBot(), "sab": SaboteurBot(), "rnd": RandomBot()}
        self.strategy = strategies.get(strategy_name, MasterCoopBot())
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


def run_single_game(playstyles: List[str], goals: List[str], derangements: List[Tuple]) -> Tuple[
    bool, List[Tuple[str, bool]]]:
    num_players = len(playstyles)
    players = [Player(i, goals[i], playstyles[i]) for i in range(num_players)]
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

    results = [(p.goal_type, p.check_victory()) for p in players]
    team_win = all(success for _, success in results)
    return team_win, results


# --- 4. АНАЛИТИКА: ОТЧЕТ ПО ЦЕЛЯМ И ОТЧЕТ ПО КОМАНДАМ ---

def run_analytics(num_players: int = 4, iterations: int = 1500):
    goals_list = ["Three 1s", "Three 2s", "Three 3s", "Three 4s", "Three 5s", "Three 6s", "Small Straight",
                  "Four of a Kind"]
    agendas = ["coop", "self", "rnd", "sab"]
    derangements = [p for p in itertools.permutations(range(num_players)) if all(p[i] != i for i in range(num_players))]

    print(f"\n{'=' * 70}")
    print(f"ОТЧЕТ 1: ШАНСЫ ВЫПОЛНЕНИЯ КОНКРЕТНЫХ ЦЕЛЕЙ (ОДНОРОДНЫЕ КОМАНДЫ)")
    print(f"{'=' * 70}")

    # Структура: {agenda: {goal: {"wins": 0, "attempts": 0}}}
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

    header = f"{'Секретная Цель':<18} | {'All Coop':>10} | {'All Self':>10} | {'All Rnd':>10} | {'All Sab':>10}"
    print(header)
    print("-" * len(header))
    for goal in goals_list:
        rates = []
        for agenda in agendas:
            att = goal_stats[agenda][goal]["attempts"]
            w = goal_stats[agenda][goal]["wins"]
            rates.append((w / att * 100) if att > 0 else 0)
        print(f"{goal:<18} | {rates[0]:>9.2f}% | {rates[1]:>9.2f}% | {rates[2]:>9.2f}% | {rates[3]:>9.2f}%")

    print(f"\n{'=' * 70}")
    print(f"ОТЧЕТ 2: МЕТА-АНАЛИЗ СОСТАВОВ КОМАНД (СИМБИОЗ АГЕНТОВ)")
    print(f"{'=' * 70}")

    all_combos = list(itertools.combinations_with_replacement(agendas, num_players))
    meta_results = []

    for combo in all_combos:
        team_wins = 0
        indiv_wins = 0
        total_games = int(iterations * 0.5)  # Немного снижаем итерации для скорости

        for _ in range(total_games):
            shuffled_team = list(combo)
            random.shuffle(shuffled_team)
            sampled_goals = random.sample(goals_list, num_players)

            t_win, ind_results = run_single_game(shuffled_team, sampled_goals, derangements)
            if t_win: team_wins += 1
            indiv_wins += sum(1 for _, success in ind_results if success)

        t_rate = (team_wins / total_games) * 100
        i_rate = (indiv_wins / (total_games * num_players)) * 100

        counts = Counter(combo)
        comp_name = ", ".join([f"{c} {r.title()}" for r, c in counts.items()])
        meta_results.append((comp_name, t_rate, i_rate))

    meta_results.sort(key=lambda x: x[1], reverse=True)

    meta_header = f"{'Состав Команды':<35} | {'Team Win Rate':>14} | {'Indiv Win Rate':>14}"
    print(meta_header)
    print("-" * len(meta_header))
    for comp, t_rate, i_rate in meta_results:
        print(f"{comp:<35} | {t_rate:>13.2f}% | {i_rate:>13.2f}%")


if __name__ == "__main__":
    run_analytics(num_players=4, iterations=10000)