import random
import tempfile
import unittest
from pathlib import Path

from chess_ai import best_move, load_tables, train_both_colours
from chess_engine import BLACK, Board


class QLearningPersistenceTests(unittest.TestCase):
    def test_trains_colours_separately_and_adds_to_existing_model(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "chess_q_table.json"
            trained = train_both_colours(1, reset=True, path=path)
            self.assertTrue(path.exists())
            self.assertEqual(trained["stats"]["white_episodes"], 1)
            self.assertEqual(trained["stats"]["black_episodes"], 1)
            self.assertTrue(trained["white"])
            self.assertTrue(trained["black"])

            added = train_both_colours(1, path=path)
            self.assertEqual(added["stats"]["white_episodes"], 2)
            self.assertEqual(added["stats"]["black_episodes"], 2)
            self.assertEqual(added, load_tables(path))

    def test_learned_agent_always_selects_a_legal_move(self):
        board = Board()
        board.push(board.legal_moves()[0])  # It is now Black's turn.
        tables = {"white": {}, "black": {}, "stats": {}}
        move = best_move(board, BLACK, tables, random.Random(7))
        self.assertIn(move, board.legal_moves())


if __name__ == "__main__":
    unittest.main()
