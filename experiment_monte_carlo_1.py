import random
from collections import Counter
import itertools
from typing import List, Dict, Tuple, Optional


# --- МОДЕЛЬ ИГРОКА И СОСТОЯНИЯ ---

class Player:
    def __init__(self, player_id: int, goal_type: str):
        self.id = player_id
        self.goal_type = goal_type
        self.locked: List[int] = []
        self.target_val: Optional[int] = None

        # Предустановка цели для "Three X"
        if goal_type.startswith("Three "):
            self.target_val = int(goal_type.split()[1][0])

    def check_victory(self) -> bool:
        """Проверка выполнения секретной цели."""
        counts = Counter(self.locked)

        if self.goal_type.startswith("Three "):
            return counts[self.target_val] >= 3
        elif self.goal_type == "Four of a Kind":
            return any(count >= 4 for count in counts.values())
        elif self.goal_type == "Small Straight":
            unique_vals = set(self.locked)
            straights = [{1, 2, 3, 4}, {2, 3, 4, 5}, {3, 4, 5, 6}]
            return any(s.issubset(unique_vals) for s in straights)
        return False


# --- ОЦЕНКА ПОЛЕЗНОСТИ И ЭВРИСТИКА ---

def evaluate_cup_utility(cup: List[int], player: Player) -> int:
    """
    Рассчитывает утилитарную ценность (полезность) стакана для конкретного игрока.
    """
    counts = Counter(cup)
    locked_counts = Counter(player.locked)

    if player.goal_type.startswith("Three ") or player.goal_type == "Four of a Kind":
        # Если цель Four of a Kind еще не выбрала target_val
        if player.target_val is None:
            return counts.most_common(1)[0][1] if counts else 0

        needed = (3 if player.goal_type.startswith("Three ") else 4) - locked_counts[player.target_val]
        needed = max(0, needed)
        return min(needed, counts[player.target_val])

    elif player.goal_type == "Small Straight":
        unique_locked = set(player.locked)
        straights = [{1, 2, 3, 4}, {2, 3, 4, 5}, {3, 4, 5, 6}]

        best_utility = 0
        for s in straights:
            missing = s - unique_locked
            utility = len(set(cup).intersection(missing))
            if utility > best_utility:
                best_utility = utility
        return best_utility

    return 0


def choose_best_die(player: Player, cup: List[int], all_players: List[Player]) -> int:
    """
    Выбирает наилучшую кость для блокировки из текущего стакана.
    """
    counts = Counter(cup)

    # Первичный выбор target_val для Four of a Kind
    if player.goal_type == "Four of a Kind" and player.target_val is None:
        player.target_val = counts.most_common(1)[0][0]

    # 1. Приоритет: Забрать кость под свою цель
    if player.goal_type.startswith("Three ") or player.goal_type == "Four of a Kind":
        if player.target_val in cup:
            return player.target_val

    elif player.goal_type == "Small Straight":
        unique_locked = set(player.locked)
        straights = [{1, 2, 3, 4}, {2, 3, 4, 5}, {3, 4, 5, 6}]

        # Ищем кость, закрывающую незаполненные позиции стритов
        for target in [2, 3, 4, 5, 1, 6]:
            if target in cup and target not in unique_locked:
                return target

    # 2. Фолбэк: Если нужных костей нет, выбираем кость, наименее нужную союзникам
    best_fallback = cup[0]
    min_team_need = float('inf')

    for die in set(cup):
        team_need = 0
        for other in all_players:
            if other.id != player.id:
                if other.target_val == die:
                    team_need += 2
                elif other.goal_type == "Small Straight" and die in [2, 3, 4, 5]:
                    team_need += 1
        if team_need < min_team_need:
            min_team_need = team_need
            best_fallback = die

    return best_fallback


# --- КООПЕРАТИВНЫЙ АЛГОРИТМ ПЕРЕДАЧИ ---

def pass_cups_cooperatively(cups: List[List[int]], players: List[Player]) -> List[List[int]]:
    """
    Выполняет направленную передачу стаканов, максимизируя общую пользу команды.
    """
    num_players = len(players)
    best_perm = None
    best_total_utility = -1

    # Перебор всех возможных перестановок передачи без передачи самому себе (деранжирование)
    for perm in itertools.permutations(range(num_players)):
        if any(perm[i] == i for i in range(num_players)):
            continue  # Нельзя передавать стакан самому себе

        total_utility = 0
        for giver_idx, receiver_idx in enumerate(perm):
            cup = cups[giver_idx]
            receiver = players[receiver_idx]
            total_utility += evaluate_cup_utility(cup, receiver)

        if total_utility > best_total_utility:
            best_total_utility = total_utility
            best_perm = perm

    # Применение лучшей схемы передачи
    new_cups = [None] * num_players
    for giver_idx, receiver_idx in enumerate(best_perm):
        new_cups[receiver_idx] = cups[giver_idx]

    return new_cups


# --- ДВИЖОК ИГРЫ ---

def run_single_game(num_players: int, goals_to_test: List[str]) -> Tuple[bool, Dict[str, bool]]:
    assigned_goals = random.sample(goals_to_test, num_players)
    players = [Player(i, assigned_goals[i]) for i in range(num_players)]

    # Генерация стартовых стаканов (5 костей)
    cups = [[random.randint(1, 6) for _ in range(5)] for _ in range(num_players)]

    # Фаза 1: Умный начальный переброс
    for p in players:
        # Перебрасываем стакан только если полезность равна 0
        if evaluate_cup_utility(cups[p.id], p) == 0:
            cups[p.id] = [random.randint(1, 6) for _ in range(5)]

    # Фаза 2: 5 раундов игры
    for _ in range(5):
        # 1. Захват 1 кости каждым игроком
        for p in players:
            die_to_lock = choose_best_die(p, cups[p.id], players)
            p.locked.append(die_to_lock)
            cups[p.id].remove(die_to_lock)

        # 2. Направленный кооперативный обмен стаканами
        cups = pass_cups_cooperatively(cups, players)

    # Фаза 3: Подсчет результатов
    results = {}
    team_success = True
    for p in players:
        passed = p.check_victory()
        results[p.goal_type] = passed
        if not passed:
            team_success = False

    return team_success, results


# --- MONTE CARLO СИМУЛЯЦИЯ ---

def run_monte_carlo(iterations: int = 10000, num_players: int = 3):
    goals = [
        "Three 1s", "Three 2s", "Three 3s",
        "Three 4s", "Three 5s", "Three 6s",
        "Small Straight", "Four of a Kind"
    ]

    goal_attempts = Counter()
    goal_wins = Counter()
    total_team_wins = 0

    for _ in range(iterations):
        team_win, player_results = run_single_game(num_players, goals)
        if team_win:
            total_team_wins += 1

        for goal, success in player_results.items():
            goal_attempts[goal] += 1
            if success:
                goal_wins[goal] += 1

    print(f"=== REFACTORED MONTE CARLO RESULTS ({iterations:,} Games | {num_players} Players) ===")
    print(f"Overall Team Win Rate: {total_team_wins / iterations * 100:.2f}%\n")
    print(f"{'Secret Goal':<20} | {'Individual Win Rate':<20}")
    print("-" * 45)
    for goal in goals:
        rate = (goal_wins[goal] / goal_attempts[goal]) * 100 if goal_attempts[goal] > 0 else 0
        print(f"{goal:<20} | {rate:.2f}%")


if __name__ == "__main__":
    run_monte_carlo(iterations=100000, num_players=3)
