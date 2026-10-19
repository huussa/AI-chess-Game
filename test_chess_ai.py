import random
import tempfile
import unittest
import json
from pathlib import Path

from chess_ai import QTableStore, best_move, import_legacy_json, train_both_colours
from chess_engine import BLACK, Board


class QLearningPersistenceTests(unittest.TestCase):
    def test_trains_colours_separately_and_adds_to_existing_model(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "chess_q_table"
            trained = train_both_colours(1, reset=True, path=path)
            self.assertTrue(path.exists())
            self.assertEqual(trained["white_episodes"], 1)
            self.assertEqual(trained["black_episodes"], 1)
            self.assertGreater(trained["white_states"], 0)
            self.assertGreater(trained["black_states"], 0)

            added = train_both_colours(1, path=path)
            self.assertEqual(added["white_episodes"], 2)
            self.assertEqual(added["black_episodes"], 2)

    def test_ai_reads_only_current_state_and_selects_legal_move(self):
        with tempfile.TemporaryDirectory() as directory:
            store = QTableStore(Path(directory) / "chess_q_table")
            try:
                board = Board()
                board.push(board.legal_moves()[0])  # It is now Black's turn.
                move = best_move(board, BLACK, store, random.Random(7))
                self.assertIn(move, board.legal_moves())
            finally:
                store.close()

    def test_legacy_json_import_preserves_colour_data_without_json_load(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            legacy = root / "legacy.json"
            database = root / "chess_q_table"
            with legacy.open("w", encoding="utf-8") as file:
                json.dump({
                    "format": "chess-q-learning-v1",
                    "white": {"white-state": {"e2e4": 1.5}},
                    "black": {"black-state": {"e7e5": -0.4}},
                    "stats": {"white_episodes": 8, "black_episodes": 9},
                }, file)
            stats = import_legacy_json(legacy, database)
            self.assertEqual(stats["white_episodes"], 8)
            self.assertEqual(stats["black_episodes"], 9)
            store = QTableStore(database)
            try:
                self.assertEqual(store.values_for("w", "white-state"), {"e2e4": 1.5})
                self.assertEqual(store.values_for("b", "black-state"), {"e7e5": -0.4})
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
