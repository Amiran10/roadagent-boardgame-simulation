import random
import itertools
from collections import Counter
from typing import List, Tuple, Optional, Dict

ALL_GOALS = [
    "Three 1s", "Three 2s", "Three 3s",
    "Three 4s", "Three 5s", "Three 6s",
    "Small Straight", "Four of a Kind"
]


# --- 1. ДЕДУКТИВНЫЙ ДВИЖОК (BAYESIAN INFERENCE) ---

def get_goal_probabilities(locked: List[int]) -> Dict[str, float]:
    """
    Вычисляет вероятностное распределение секретной цели игрока
    на основе его публично заблокированных костей.
    """
    if not locked:
        # Если костей нет, все цели равновероятны
        return {g: 1.0 / len(ALL_GOALS) for g in ALL_GOALS}

    counts = Counter(locked)
    unique_locked = set(locked)
    scores = {}

    for goal in ALL_GOALS:
        score = 1.0  # Априорный вес

        if goal.startswith("Three "):
            target_val = int(goal.split()[1][0])
            score += counts[target_val] * 4.0  # Сильный сигнал при совпадении

        elif goal == "Four of a Kind":
            max_freq = counts.most_common(1)[0][1] if counts else 0
            score += max_freq * 3.0

        elif goal == "Small Straight":
            straights = [{1, 2, 3, 4}, {2, 3, 4, 5}, {3, 4, 5, 6}]
            max_match = max(len(unique_locked.intersection(s)) for s in straights)
            score += max_match * 3.5

        scores[goal] = score

    # Нормализация вероятностей (сумма = 1.0)
    total_score = sum(scores.values())
    return {goal: score / total_score for goal, score in scores.items()}


def evaluate_cup_utility(cup: List[int], goal_type: str, locked: List[int], target_val: Optional[int]) -> int:
    """Точная оценка полезности стакана для известной цели."""
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


def calculate_expected_team_utility(cup_or_die: List[int], other_player: 'Player') -> float:
    """
    Рассчитывает Математическое Ожидание полезности кубика/стакана для партнера,
    не зная его секретной цели напрямую.
    """
    probs = get_goal_probabilities(other_player.locked)
    expected_utility = 0.0

    for goal, prob in probs.items():
        # Гипотетический выбор целевого числа для гипотезы
        target_val = None
        if goal.startswith("Three "):
            target_val = int(goal.split()[1][0])
        elif goal == "Four of a Kind":
            counts = Counter(other_player.locked)
            target_val = counts.most_common(1)[0][0] if counts else None

        util = evaluate_cup_utility(cup_or_die, goal, other_player.locked, target_val)
        expected_utility += prob * util

    return expected_utility


# --- 2. СТРАТЕГИИ ИГРОКОВ В УСЛОВИЯХ ТУМАНА ВОЙНЫ ---

class BotStrategy:
    def choose_die(self, player: 'Player', cup: List[int], all_players: List['Player']) -> int: pass

    def evaluate_pass(self, player: 'Player', cup: List[int], receiver: 'Player') -> float: pass

    def wants_reroll(self, player: 'Player', cup: List[int]) -> bool: pass


class MasterCoopDeductiveBot(BotStrategy):
    """Кооперативный ИИ с модулем дедукции под Туманом Войны."""

    def wants_reroll(self, player, cup):
        my_util = evaluate_cup_utility(cup, player.get_secret_goal(), player.locked, player.target_val)
        return my_util == 0

    def choose_die(self, player, cup, all_players):
        secret_goal = player.get_secret_goal()
        valid_choices = []

        if secret_goal.startswith("Three ") or secret_goal == "Four of a Kind":
            if player.target_val in cup: valid_choices.append(player.target_val)
        elif secret_goal == "Small Straight":
            for t in [1, 2, 3, 4, 5, 6]:
                if t in cup and t not in player.locked:
                    valid_choices.append(t)

        # 1. Забираем полезный кубик себе (выбирая тот, у которого минимальное МО урона команде)
        if valid_choices:
            best_choice = valid_choices[0]
            min_team_damage = float('inf')
            for die in valid_choices:
                exp_damage = sum(calculate_expected_team_utility([die], p) for p in all_players if p.id != player.id)
                if exp_damage < min_team_damage:
                    min_team_damage = exp_damage
                    best_choice = die
            return best_choice

        # 2. Сброс балласта: забираем мусор с минимальным МО нужды для команды
        best_fallback, min_need = cup[0], float('inf')
        for die in set(cup):
            exp_need = sum(calculate_expected_team_utility([die], p) for p in all_players if p.id != player.id)
            if exp_need < min_need:
                min_need = exp_need
                best_fallback = die
        return best_fallback

    def evaluate_pass(self, player, cup, receiver):
        # Оценка передачи на основе математического ожидания
        return calculate_expected_team_utility(cup, receiver)


class SelfishBot(BotStrategy):
    def wants_reroll(self, player, cup):
        return evaluate_cup_utility(cup, player.get_secret_goal(), player.locked, player.target_val) == 0

    def choose_die(self, player, cup, all_players):
        goal = player.get_secret_goal()
        if goal.startswith("Three ") or goal == "Four of a Kind":
            if player.target_val in cup: return player.target_val
        elif goal == "Small Straight":
            for t in [2, 3, 4, 5, 1, 6]:
                if t in cup and t not in player.locked: return t
        return random.choice(cup)

    def evaluate_pass(self, player, cup, receiver):
        return random.random()


class SaboteurDeductiveBot(BotStrategy):
    """Саботажник: вычисляет самые вероятные цели жертв и крадет их ресурсы."""

    def wants_reroll(self, player, cup):
        return random.choice([True, False])

    def choose_die(self, player, cup, all_players):
        worst_die, max_damage = cup[0], -1.0
        for die in set(cup):
            exp_damage = sum(calculate_expected_team_utility([die], p) for p in all_players if p.id != player.id)
            if exp_damage > max_damage:
                max_damage = exp_damage
                worst_die = die
        return worst_die

    def evaluate_pass(self, player, cup, receiver):
        return -calculate_expected_team_utility(cup, receiver)


class RandomBot(BotStrategy):
    def wants_reroll(self, player, cup): return random.choice([True, False])

    def choose_die(self, player, cup, all_players): return random.choice(cup)

    def evaluate_pass(self, player, cup, receiver): return random.random()


# --- 3. ИГРОК С ПРИВАТНЫМ СОСТОЯНИЕМ ---

class Player:
    def __init__(self, player_id: int, secret_goal: str, strategy_name: str):
        self.id = player_id
        self.__secret_goal = secret_goal  # Инкапсулированная секретная цель
        self.locked: List[int] = []  # Публичная информация
        self.target_val: Optional[int] = int(secret_goal.split()[1][0]) if secret_goal.startswith("Three ") else None

        strategies = {
            "coop": MasterCoopDeductiveBot(),
            "self": SelfishBot(),
            "sab": SaboteurDeductiveBot(),
            "rnd": RandomBot()
        }
        self.strategy = strategies.get(strategy_name, MasterCoopDeductiveBot())

    def get_secret_goal(self) -> str:
        """Только сам игрок имеет доступ к своей секретной цели."""
        return self.__secret_goal

    def check_victory(self) -> bool:
        counts = Counter(self.locked)
        goal = self.__secret_goal
        if goal.startswith("Three "):
            return counts[self.target_val] >= 3
        elif goal == "Four of a Kind":
            return any(c >= 4 for c in counts.values())
        elif goal == "Small Straight":
            return any(s.issubset(set(self.locked)) for s in [{1, 2, 3, 4}, {2, 3, 4, 5}, {3, 4, 5, 6}])
        return False


# --- 4. ДВИЖОК СИМУЛЯЦИИ И АНАЛИТИКА ---

def run_single_game(playstyles: List[str], goals: List[str], derangements: List[Tuple]) -> Tuple[
    bool, List[Tuple[str, bool]]]:
    num_players = len(playstyles)
    players = [Player(i, goals[i], playstyles[i]) for i in range(num_players)]
    cups = [[random.randint(1, 6) for _ in range(5)] for _ in range(num_players)]

    for p in players:
        if p.get_secret_goal() == "Four of a Kind" and p.target_val is None:
            counts = Counter(cups[p.id])
            p.target_val = counts.most_common(1)[0][0] if counts else 1

        if p.strategy.wants_reroll(p, cups[p.id]):
            cups[p.id] = [random.randint(1, 6) for _ in range(5)]
            if p.get_secret_goal() == "Four of a Kind":
                p.target_val = Counter(cups[p.id]).most_common(1)[0][0]

    for _ in range(5):
        # 1. Фаза выбора
        for p in players:
            die = p.strategy.choose_die(p, cups[p.id], players)
            p.locked.append(die)
            cups[p.id].remove(die)

        # 2. Фаза передачи (аукцион на основе вероятностных ожиданий)
        best_perm, best_score = None, -float('inf')
        for perm in derangements:
            score = sum(
                players[i].strategy.evaluate_pass(players[i], cups[i], players[perm[i]]) for i in range(num_players))
            if score > best_score:
                best_score, best_perm = score, perm

        cups = [cups[giver_idx] for giver_idx in best_perm]

    results = [(p.get_secret_goal(), p.check_victory()) for p in players]
    return all(success for _, success in results), results


def run_fog_of_war_analytics(num_players: int = 4, iterations: int = 1000):
    agendas = ["coop", "self", "rnd", "sab"]
    derangements = [p for p in itertools.permutations(range(num_players)) if all(p[i] != i for i in range(num_players))]

    print(f"\n======================================================================")
    print(f"СИМУЛЯЦИЯ ПОД ТУМАНОМ ВОЙНЫ (FOG OF WAR | {num_players} ИГРОКОВ)")
    print(f"======================================================================\n")

    # Сводный отчет по однородным составам
    print(f"{'Состав Команды':<20} | {'Team Win Rate':>15} | {'Indiv Win Rate':>15}")
    print("-" * 56)

    for agenda in agendas:
        team_wins, indiv_wins = 0, 0
        playstyles = [agenda] * num_players

        for _ in range(iterations):
            sampled_goals = random.choices(ALL_GOALS, k=num_players)
            t_win, results = run_single_game(playstyles, sampled_goals, derangements)
            if t_win: team_wins += 1
            indiv_wins += sum(1 for _, success in results if success)

        t_rate = (team_wins / iterations) * 100
        i_rate = (indiv_wins / (iterations * num_players)) * 100
        print(f"All {agenda.title():<16} | {t_rate:>14.2f}% | {i_rate:>14.2f}%")


if __name__ == "__main__":
    run_fog_of_war_analytics(num_players=4, iterations=1000)