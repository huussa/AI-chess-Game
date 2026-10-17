import unittest

from chess_engine import BLACK, WHITE, Board


class KingRulesTests(unittest.TestCase):
    def test_king_moves_remain_available_after_rules_are_checked(self):
        """Regression test for a consumed king-direction generator."""
        board = Board()
        board.legal_moves()  # This used to consume the king move directions.
        board.grid = [[None for _ in range(8)] for _ in range(8)]
        board.grid[7][4] = "wK"  # e1
        board.grid[0][4] = "bK"  # e8
        board.turn = WHITE
        moves = board.legal_moves()
        self.assertIn("e1e2", {move.uci() for move in moves})
        self.assertFalse(board.outcome())

    def test_king_cannot_capture_the_other_king(self):
        board = Board()
        board.grid = [[None for _ in range(8)] for _ in range(8)]
        board.grid[7][4] = "wK"  # e1
        board.grid[6][4] = "bK"  # e2
        board.turn = WHITE
        self.assertNotIn("e1e2", {move.uci() for move in board.pseudo_legal_moves()})

    def test_checked_king_can_escape_to_a_safe_square(self):
        board = Board()
        board.grid = [[None for _ in range(8)] for _ in range(8)]
        board.grid[0][4] = "bK"  # e8
        board.grid[7][0] = "wK"  # a1
        board.grid[7][4] = "wR"  # e1 gives check along the e-file
        board.turn = BLACK
        moves = {move.uci() for move in board.legal_moves()}
        self.assertIn("e8d8", moves)
        self.assertIn("e8f8", moves)
        self.assertIsNone(board.outcome())

    def test_checkmate_requires_check_and_no_legal_escape(self):
        board = Board()
        board.grid = [[None for _ in range(8)] for _ in range(8)]
        board.grid[0][0] = "bK"  # a8
        board.grid[2][2] = "wK"  # c6
        board.grid[1][1] = "wQ"  # b7: checkmate
        board.turn = BLACK
        self.assertEqual(board.outcome(), "checkmate: White wins")

        board.grid[1][1] = None  # Remove the checking queen.
        self.assertIsNone(board.outcome())


if __name__ == "__main__":
    unittest.main()
