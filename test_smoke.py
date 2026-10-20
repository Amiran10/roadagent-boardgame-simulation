import random
import unittest

from mappo import Actor, Critic, collect_episode, evaluate
from missions import get_mission
from road_agent import Goal, RoadAgentGame, RuleError


class RoadAgentSmokeTests(unittest.TestCase):
    def test_hidden_observation_and_illegal_self_pass(self):
        players = ("A", "B", "C")
        game = RoadAgentGame(players, {"A": Goal.three(1), "B": Goal.three(2), "C": Goal.straight()}, rng=random.Random(1), mission_rule=lambda _: True)
        game.start()
        self.assertFalse(hasattr(game.observation("A"), "other_cups"))
        game.begin()
        for player in players:
            game.lock(player, 0)
        with self.assertRaises(RuleError):
            game.pass_cups({"A": "A", "B": "C", "C": "B"})

    def test_mappo_collects_a_complete_legal_episode(self):
        episode = collect_episode(Actor(), Critic(), get_mission("standard").rule, random.Random(2), __import__("torch").device("cpu"), deterministic=False)
        self.assertEqual(len(episode.steps), 10)  # five lock phases plus five pass phases
        self.assertTrue(all(step.actions.shape == (3,) for step in episode.steps))


if __name__ == "__main__":
    unittest.main(verbosity=2)
