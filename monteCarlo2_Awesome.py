import random
import itertools
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional

def evaluate_cup_utility(cup: List[int], goal_type: str, locked: List[int], target_val: Optional[int]) -> int:
    """Оценивает, сколько полезных костей в стакане для конкретной цели."""
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


# --- СТРАТЕГИИ ПОВЕДЕНИЯ (ПАТТЕРН "СТРАТЕГИЯ") ---

class BotStrategy:
    """Базовый интерфейс для стилей игры."""

    def choose_die(self, player: 'Player', cup: List[int], all_players: List['Player']) -> int:
        pass

    def evaluate_pass(self, player: 'Player', cup: List[int], receiver: 'Player') -> float:
        pass

    def wants_reroll(self, player: 'Player', cup: List[int]) -> bool:
        pass


class CooperativeBot(BotStrategy):
    def wants_reroll(self, player: 'Player', cup: List[int]) -> bool:
        return evaluate_cup_utility(cup, player.goal_type, player.locked, player.target_val) == 0

    def choose_die(self, player: 'Player', cup: List[int], all_players: List['Player']) -> int:
        # 1. Берем себе
        utility = evaluate_cup_utility(cup, player.goal_type, player.locked, player.target_val)
        if utility > 0:
            if player.goal_type == "Small Straight":
                unique = set(player.locked)
                for t in [2, 3, 4, 5, 1, 6]:
                    if t in cup and t not in unique: return t
            else:
                if player.target_val in cup: return player.target_val

        # 2. Фолбэк: берем то, что меньше всего нужно команде
        best_fallback, min_need = cup[0], float('inf')
        for die in set(cup):
            team_need = sum(evaluate_cup_utility([die], p.goal_type, p.locked, p.target_val)
                            for p in all_players if p.id != player.id)
            if team_need < min_need:
                min_need = team_need
                best_fallback = die
        return best_fallback

    def evaluate_pass(self, player: 'Player', cup: List[int], receiver: 'Player') -> float:
        return evaluate_cup_utility(cup, receiver.goal_type, receiver.locked, receiver.target_val)


class SelfishBot(BotStrategy):
    def wants_reroll(self, player: 'Player', cup: List[int]) -> bool:
        return evaluate_cup_utility(cup, player.goal_type, player.locked, player.target_val) == 0

    def choose_die(self, player: 'Player', cup: List[int], all_players: List['Player']) -> int:
        # Берет только себе. Если нет полезного - берет случайный (наплевать на команду)
        utility = evaluate_cup_utility(cup, player.goal_type, player.locked, player.target_val)
        if utility > 0:
            if player.goal_type.startswith("Three ") or player.goal_type == "Four of a Kind":
                if player.target_val in cup: return player.target_val
            elif player.goal_type == "Small Straight":
                for t in [2, 3, 4, 5, 1, 6]:
                    if t in cup and t not in player.locked: return t
        return random.choice(cup)

    def evaluate_pass(self, player: 'Player', cup: List[int], receiver: 'Player') -> float:
        return random.random()  # Эгоисту все равно, кому отдавать кубики


class SaboteurBot(BotStrategy):
    def wants_reroll(self, player: 'Player', cup: List[int]) -> bool:
        # Перебрасывает, если в стакане СЛИШКОМ много хорошего для команды
        return sum(evaluate_cup_utility(cup, p.goal_type, p.locked, p.target_val) for p in player.all_players) > 2

    def choose_die(self, player: 'Player', cup: List[int], all_players: List['Player']) -> int:
        # Саботажник ворует самые нужные команде кубики
        worst_die_for_team, max_team_damage = cup[0], -1
        for die in set(cup):
            damage = sum(evaluate_cup_utility([die], p.goal_type, p.locked, p.target_val)
                         for p in all_players if p.id != player.id)
            if damage > max_team_damage:
                max_team_damage = damage
                worst_die_for_team = die
        return worst_die_for_team

    def evaluate_pass(self, player: 'Player', cup: List[int], receiver: 'Player') -> float:
        # Максимально хочет отдать стакан тому, кому он НЕ нужен (отрицательная полезность)
        return -evaluate_cup_utility(cup, receiver.goal_type, receiver.locked, receiver.target_val)


class RandomBot(BotStrategy):
    def wants_reroll(self, player: 'Player', cup: List[int]) -> bool: return random.choice([True, False])

    def choose_die(self, player: 'Player', cup: List[int], all_players: List['Player']) -> int: return random.choice(
        cup)

    def evaluate_pass(self, player: 'Player', cup: List[int], receiver: 'Player') -> float: return random.random()


# --- СИСТЕМА ИГРОКА И ДВИЖОК ---

class Player:
    def __init__(self, player_id: int, goal_type: str, strategy_name: str):
        self.id = player_id
        self.goal_type = goal_type
        self.locked: List[int] = []
        self.target_val: Optional[int] = int(goal_type.split()[1][0]) if goal_type.startswith("Three ") else None

        strategies = {
            "cooperative": CooperativeBot(),
            "selfish": SelfishBot(),
            "saboteur": SaboteurBot(),
            "random": RandomBot()
        }
        self.strategy = strategies.get(strategy_name.lower(), CooperativeBot())
        self.all_players = []  # Ссылка для саботажника (настраивается в движке)

    def check_victory(self) -> bool:
        counts = Counter(self.locked)
        if self.goal_type.startswith("Three "):
            return counts[self.target_val] >= 3
        elif self.goal_type == "Four of a Kind":
            return any(c >= 4 for c in counts.values())
        elif self.goal_type == "Small Straight":
            return any(s.issubset(set(self.locked)) for s in [{1, 2, 3, 4}, {2, 3, 4, 5}, {3, 4, 5, 6}])
        return False


def run_single_game(playstyles: List[str], goals_to_test: List[str]) -> Tuple[bool, Dict[str, bool]]:
    num_players = len(playstyles)
    assigned_goals = random.sample(goals_to_test, num_players)
    players = [Player(i, assigned_goals[i], playstyles[i]) for i in range(num_players)]

    # Инициализируем знания о других игроках
    for p in players: p.all_players = players

    cups = [[random.randint(1, 6) for _ in range(5)] for _ in range(num_players)]

    # Фаза 1: Переброс (определяется ботом)
    for p in players:
        # Устанавливаем цель для Four of a Kind заранее, чтобы боты могли оценивать
        if p.goal_type == "Four of a Kind" and p.target_val is None:
            counts = Counter(cups[p.id])
            p.target_val = counts.most_common(1)[0][0] if counts else 1

        if p.strategy.wants_reroll(p, cups[p.id]):
            cups[p.id] = [random.randint(1, 6) for _ in range(5)]
            if p.goal_type == "Four of a Kind":  # Обновляем таргет после переброса
                p.target_val = Counter(cups[p.id]).most_common(1)[0][0]

    # Фаза 2: 5 Раундов
    for _ in range(5):
        # Блокировка
        for p in players:
            die_to_lock = p.strategy.choose_die(p, cups[p.id], players)
            p.locked.append(die_to_lock)
            cups[p.id].remove(die_to_lock)

        # Фаза 3: Разрешение передачи стаканов (Аукцион желаний)
        best_perm, best_score = None, -float('inf')
        valid_perms = [p for p in itertools.permutations(range(num_players)) if
                       all(p[i] != i for i in range(num_players))]

        for perm in valid_perms:
            score = sum(
                players[i].strategy.evaluate_pass(players[i], cups[i], players[perm[i]]) for i in range(num_players))
            if score > best_score:
                best_score = score
                best_perm = perm

        new_cups = [None] * num_players
        for giver_idx, receiver_idx in enumerate(best_perm):
            new_cups[receiver_idx] = cups[giver_idx]
        cups = new_cups

    # Фаза 4: Оценка
    results = {p.goal_type: p.check_victory() for p in players}
    return all(results.values()), results


# --- ЭКСПЕРИМЕНТАЛЬНЫЙ ЗАПУСК ---

def evaluate_scenario(scenario_name: str, playstyles: List[str], iterations: int = 5000):
    goals = ["Three 1s", "Three 2s", "Three 3s", "Three 4s", "Three 5s", "Three 6s", "Small Straight", "Four of a Kind"]
    team_wins = 0
    goal_attempts = defaultdict(int)
    goal_wins = defaultdict(int)

    for _ in range(iterations):
        team_win, results = run_single_game(playstyles, goals)
        if team_win: team_wins += 1
        for goal, success in results.items():
            goal_attempts[goal] += 1
            if success: goal_wins[goal] += 1

    print(f"\n=== СЦЕНАРИЙ: {scenario_name} ===")
    print(f"Состав команды: {', '.join(playstyles).title()}")
    print(f"Командный Винрейт: {team_wins / iterations * 100:.2f}%\n")
    print(f"{'Секретная Цель':<20} | {'Индивидуальный Винрейт'}")
    print("-" * 45)
    for goal in goals:
        rate = (goal_wins[goal] / goal_attempts[goal] * 100) if goal_attempts[goal] > 0 else 0
        print(f"{goal:<20} | {rate:.2f}%")


if __name__ == "__main__":
    evaluate_scenario("Идеальная команда", ["cooperative", "cooperative", "cooperative"])
    evaluate_scenario("Один эгоист в команде", ["cooperative", "cooperative", "selfish"])
    evaluate_scenario("Предатель среди нас", ["cooperative", "cooperative", "saboteur"])
    evaluate_scenario("Абсолютный Хаос", ["random", "random", "random"])