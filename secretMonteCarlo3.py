import random
import itertools
from collections import Counter
from typing import List, Tuple, Optional, Dict


# --- 1. ТУМАН ВОЙНЫ: ДВИЖОК ДЕДУКЦИИ ---

def infer_needed_dice(locked: List[int]) -> Dict[int, float]:
    needs = {i: 0.0 for i in range(1, 7)}
    if not locked:
        for i in [2, 3, 4, 5]: needs[i] = 0.5
        return needs

    counts = Counter(locked)
    most_common_val, max_count = counts.most_common(1)[0]
    unique_locked = set(locked)

    # Гипотеза: собирает одинаковые кости
    if max_count >= 1:
        needs[most_common_val] += max_count * 2.0

    # Гипотеза: собирает стрит
    if len(unique_locked) > 1:
        for s in [{1, 2, 3, 4}, {2, 3, 4, 5}, {3, 4, 5, 6}]:
            overlap = len(s.intersection(unique_locked))
            if overlap >= 1:
                for m in (s - unique_locked):
                    needs[m] += overlap * 1.0
    return needs


def evaluate_self_utility(cup: List[int], goal_type: str, locked: List[int], target_val: Optional[int]) -> int:
    counts = Counter(cup)
    if goal_type.startswith("Three ") or goal_type == "Four of a Kind":
        if target_val is None: return counts.most_common(1)[0][1] if counts else 0
        needed = (3 if goal_type.startswith("Three ") else 4) - Counter(locked)[target_val]
        return min(max(0, needed), counts[target_val])
    elif goal_type == "Small Straight":
        unique_locked = set(locked)
        best_util = 0
        for s in [{1, 2, 3, 4}, {2, 3, 4, 5}, {3, 4, 5, 6}]:
            missing = s - unique_locked
            util = len(set(cup).intersection(missing))
            if util > best_util: best_util = util
        return best_util
    return 0


# --- 2. СТРАТЕГИИ БОТОВ (ОБНОВЛЕННЫЕ) ---

class BotStrategy:
    def wants_mulligan(self, player: 'Player', cup: List[int]) -> bool: pass

    def choose_die(self, player: 'Player', cup: List[int], all_players: List['Player']) -> int: pass

    def evaluate_pass(self, player: 'Player', cup: List[int], receiver: 'Player') -> float: pass


class SmartCoopBot(BotStrategy):
    def wants_mulligan(self, player, cup):
        # ПРАВИЛО КИСЛОРОДНОЙ МАСКИ: Если мне ничего не нужно - перебрасываю. Точка.
        return evaluate_self_utility(cup, player.goal_type, player.locked, player.target_val) == 0

    def choose_die(self, player, cup, all_players):
        valid_choices = []
        if player.goal_type.startswith("Three ") or player.goal_type == "Four of a Kind":
            if player.target_val in cup: valid_choices.append(player.target_val)
        elif player.goal_type == "Small Straight":
            for t in [1, 2, 3, 4, 5, 6]:
                if t in cup and t not in player.locked: valid_choices.append(t)

        team_needs = {die: sum(infer_needed_dice(p.locked).get(die, 0) for p in all_players if p.id != player.id) for
                      die in set(cup)}

        # Если есть полезные для МЕНЯ кости - беру ту, что МЕНЬШЕ всего нужна команде (оставляю лучшее)
        if valid_choices:
            return min(valid_choices, key=lambda d: team_needs[d])
        # Если полезных нет - сбрасываю балласт, который никому не нужен
        return min(set(cup), key=lambda d: team_needs.get(d, 0))

    def evaluate_pass(self, player, cup, receiver):
        needs = infer_needed_dice(receiver.locked)
        return sum(needs.get(die, 0) for die in cup)


class SelfishBot(BotStrategy):
    def wants_mulligan(self, player, cup):
        return evaluate_self_utility(cup, player.goal_type, player.locked, player.target_val) == 0

    def choose_die(self, player, cup, all_players):
        # Берет первое попавшееся нужное, игнорируя команду
        if player.goal_type.startswith("Three ") or player.goal_type == "Four of a Kind":
            if player.target_val in cup: return player.target_val
        elif player.goal_type == "Small Straight":
            for t in [2, 3, 4, 5, 1, 6]:
                if t in cup and t not in player.locked: return t
        return random.choice(cup)

    def evaluate_pass(self, player, cup, receiver):
        return random.random()  # Эгоисту плевать, кому отдавать стакан


class SaboteurBot(BotStrategy):
    def wants_mulligan(self, player, cup):
        # Саботажник перебрасывает, если в стакане СЛИШКОМ много хорошего для команды
        inferred_team_util = sum(
            sum(infer_needed_dice(p.locked).get(d, 0) for d in cup) for p in player.all_players if p.id != player.id)
        return inferred_team_util > 3.0

    def choose_die(self, player, cup, all_players):
        # Ворует кость, которая сильнее всего нужна команде
        team_needs = {die: sum(infer_needed_dice(p.locked).get(die, 0) for p in all_players if p.id != player.id) for
                      die in set(cup)}
        return max(set(cup), key=lambda d: team_needs.get(d, 0))

    def evaluate_pass(self, player, cup, receiver):
        needs = infer_needed_dice(receiver.locked)
        return -sum(needs.get(die, 0) for die in cup)


# --- 3. ИГРОК И ДВИЖОК С ПОСЛЕДОВАТЕЛЬНОЙ БЛОКИРОВКОЙ ---

class Player:
    def __init__(self, player_id: int, goal_type: str, strategy_name: str):
        self.id = player_id
        self.goal_type = goal_type
        self.strategy_name = strategy_name
        self.locked = []
        self.target_val = int(goal_type.split()[1][0]) if goal_type.startswith("Three ") else None

        strategies = {"Coop": SmartCoopBot(), "Selfish": SelfishBot(), "Saboteur": SaboteurBot()}
        self.strategy = strategies.get(strategy_name, SmartCoopBot())
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


def run_mixed_game(playstyles: List[str], goals: List[str]) -> Dict:
    num_players = len(playstyles)
    players = [Player(i, goals[i], playstyles[i]) for i in range(num_players)]
    for p in players: p.all_players = players

    cups = [[random.randint(1, 6) for _ in range(5)] for _ in range(num_players)]
    derangements = [p for p in itertools.permutations(range(num_players)) if all(p[i] != i for i in range(num_players))]

    # 1. ФАЗА MULLIGAN (Начальный переброс до блокировки)
    for p in players:
        if p.goal_type == "Four of a Kind" and p.target_val is None:
            counts = Counter(cups[p.id])
            p.target_val = counts.most_common(1)[0][0] if counts else 1

        if p.strategy.wants_mulligan(p, cups[p.id]):
            cups[p.id] = [random.randint(1, 6) for _ in range(5)]
            if p.goal_type == "Four of a Kind":
                p.target_val = Counter(cups[p.id]).most_common(1)[0][0]

    # 2. ИГРОВОЙ ЦИКЛ (5 раундов)
    for _ in range(5):
        # ПОСЛЕДОВАТЕЛЬНАЯ БЛОКИРОВКА
        for i, p in enumerate(players):
            # Бот выбирает кубик, уже видя, что заблокировали игроки ДО него в этом раунде
            die = p.strategy.choose_die(p, cups[p.id], players)
            p.locked.append(die)
            cups[p.id].remove(die)

        # ФАЗА ПЕРЕДАЧИ (Выбор, кому отдать оставшийся стакан)
        best_perm, best_score = None, -float('inf')
        for perm in derangements:
            score = sum(
                players[j].strategy.evaluate_pass(players[j], cups[j], players[perm[j]]) for j in range(num_players))
            if score > best_score:
                best_score, best_perm = score, perm

        cups = [cups[giver_idx] for giver_idx in best_perm]

    return {p.strategy_name: p.check_victory() for p in players}


# --- 4. АНАЛИТИКА ---

def run_showdown(iterations: int = 2000):
    goals_list = ["Three 1s", "Three 3s", "Three 6s", "Small Straight", "Four of a Kind"]

    print(f"=== ТЕСТ 1: ВСЕ 3 БОТА ЗА ОДНИМ СТОЛОМ (Coop vs Selfish vs Saboteur) ===")
    mixed_team = ["Coop", "Selfish", "Saboteur"]
    wins = {"Coop": 0, "Selfish": 0, "Saboteur": 0}
    team_wins = 0

    for _ in range(iterations):
        goals = random.sample(goals_list, 3)
        results = run_mixed_game(mixed_team, goals)
        if all(results.values()): team_wins += 1
        for bot, success in results.items():
            if success: wins[bot] += 1

    print(f"Игр сыграно: {iterations}")
    print(f"Побед всей команды: {team_wins} (Винрейт: {(team_wins / iterations) * 100:.2f}%)")
    print("-" * 45)
    print(f"{'Тип Бота':<15} | {'Индивидуальный Винрейт'}")
    for bot in mixed_team:
        print(f"{bot:<15} | {(wins[bot] / iterations) * 100:.2f}%")

    print(f"\n=== ТЕСТ 2: ОДНОРОДНЫЕ КОМАНДЫ (С обновленной логикой Mulligan) ===")
    for team_type in [["Coop", "Coop", "Coop"], ["Selfish", "Selfish", "Selfish"]]:
        t_wins = 0
        for _ in range(iterations):
            goals = random.sample(goals_list, 3)
            results = run_mixed_game(team_type, goals)
            if all(results.values()): t_wins += 1

        name = "All Cooperative" if team_type[0] == "Coop" else "All Selfish"
        print(f"{name:<20} | Team Win Rate: {(t_wins / iterations) * 100:.2f}%")


if __name__ == "__main__":
    run_showdown(iterations=10000)