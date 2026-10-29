import random
import tempfile
import unittest
import json
from pathlib import Path

from chess_ai import (
    QTableStore, best_move, import_legacy_json, minimax_move,
    record_game_transition, state_id, train_both_colours,
)
from chess_engine import BLACK, WHITE, Board


class QLearningPersistenceTests(unittest.TestCase):
    def test_trains_colours_separately_and_adds_to_existing_model(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "chess_q_table"
            trained = train_both_colours(1, reset=True, path=path, max_plies=1)
            self.assertTrue(path.exists())
            self.assertEqual(trained["white_episodes"], 1)
            self.assertEqual(trained["black_episodes"], 1)
            self.assertGreater(trained["white_states"], 0)
            self.assertGreater(trained["black_states"], 0)

            added = train_both_colours(1, path=path, max_plies=1)
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

    def test_minimax_prefers_an_available_high_value_capture(self):
        board = Board()
        board.grid = [[None for _ in range(8)] for _ in range(8)]
        board.grid[7][4] = "wK"  # e1
        board.grid[0][4] = "bK"  # e8
        board.grid[7][3] = "wQ"  # d1
        board.grid[1][3] = "bR"  # d7
        board.turn = WHITE

        self.assertEqual(minimax_move(board, WHITE, depth=1).uci(), "d1d7")

    def test_unseen_game_position_uses_minimax_instead_of_random_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            store = QTableStore(Path(directory) / "chess_q_table")
            try:
                board = Board()
                board.grid = [[None for _ in range(8)] for _ in range(8)]
                board.grid[2][2] = "wK"  # c6 protects d7 after the capture.
                board.grid[0][4] = "bK"
                board.grid[7][3] = "wQ"
                board.grid[1][3] = "bR"
                board.turn = WHITE
                self.assertEqual(best_move(board, WHITE, store).uci(), "d1d7")
            finally:
                store.close()

    def test_real_game_transition_is_persisted_for_future_q_learning(self):
        with tempfile.TemporaryDirectory() as directory:
            store = QTableStore(Path(directory) / "chess_q_table")
            try:
                board = Board()
                board.push(next(move for move in board.legal_moves() if move.uci() == "e2e4"))
                state = state_id(board)
                action = next(move for move in board.legal_moves() if move.uci() == "e7e5")
                board.push(action)
                board.push(next(move for move in board.legal_moves() if move.uci() == "g1f3"))
                record_game_transition(store, BLACK, state, action, None, board)
                self.assertIn("e7e5", store.values_for(BLACK, state))
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
