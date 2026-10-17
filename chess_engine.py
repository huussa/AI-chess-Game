"""Pure chess rules and a small RL-friendly environment.

The UI deliberately does not live in this module: a learning agent can use
``ChessEnv`` without importing pygame or opening a window.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

WHITE, BLACK = "w", "b"
FILES = "abcdefgh"
KNIGHT_DELTAS = ((-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1))
# This must be a reusable tuple, not a generator.  The old generator was
# exhausted on the first rules check, causing kings to lose their normal moves
# and creating false checkmates later in a game.
KING_DELTAS = tuple((dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc)


def opponent(color: str) -> str:
    return BLACK if color == WHITE else WHITE


def inside(square: tuple[int, int]) -> bool:
    return 0 <= square[0] < 8 and 0 <= square[1] < 8


@dataclass(frozen=True)
class Move:
    start: tuple[int, int]
    end: tuple[int, int]
    promotion: Optional[str] = None
    en_passant: bool = False
    castle: bool = False

    def uci(self) -> str:
        def name(s: tuple[int, int]) -> str:
            return FILES[s[1]] + str(8 - s[0])
        return name(self.start) + name(self.end) + (self.promotion or "").lower()


class Board:
    """Chess position; pieces are two-character strings such as ``wP``."""

    def __init__(self) -> None:
        self.grid: list[list[Optional[str]]] = []
        self.turn = WHITE
        self.castling = {"wK": True, "wQ": True, "bK": True, "bQ": True}
        self.en_passant: Optional[tuple[int, int]] = None
        self.halfmove_clock = 0
        self.fullmove_number = 1
        self.history: list[Move] = []
        self.reset()

    def reset(self) -> None:
        self.grid = [[None for _ in range(8)] for _ in range(8)]
        order = "RNBQKBNR"
        for col, kind in enumerate(order):
            self.grid[0][col] = BLACK + kind
            self.grid[1][col] = BLACK + "P"
            self.grid[6][col] = WHITE + "P"
            self.grid[7][col] = WHITE + kind
        self.turn, self.en_passant = WHITE, None
        self.castling = {"wK": True, "wQ": True, "bK": True, "bQ": True}
        self.halfmove_clock, self.fullmove_number, self.history = 0, 1, []

    def copy(self) -> "Board":
        other = Board.__new__(Board)
        other.grid = [row[:] for row in self.grid]
        other.turn = self.turn
        other.castling = self.castling.copy()
        other.en_passant = self.en_passant
        other.halfmove_clock = self.halfmove_clock
        other.fullmove_number = self.fullmove_number
        other.history = self.history[:]
        return other

    def piece_at(self, square: tuple[int, int]) -> Optional[str]:
        return self.grid[square[0]][square[1]]

    def _add_ray_moves(self, square: tuple[int, int], color: str, directions: Iterable[tuple[int, int]]) -> list[Move]:
        moves = []
        for dr, dc in directions:
            row, col = square
            while True:
                row, col = row + dr, col + dc
                if not inside((row, col)):
                    break
                target = self.grid[row][col]
                if target is None:
                    moves.append(Move(square, (row, col)))
                else:
                    if target[0] != color and target[1] != "K":
                        moves.append(Move(square, (row, col)))
                    break
        return moves

    def pseudo_legal_moves(self, color: Optional[str] = None) -> list[Move]:
        color = color or self.turn
        moves: list[Move] = []
        for row in range(8):
            for col in range(8):
                piece = self.grid[row][col]
                if not piece or piece[0] != color:
                    continue
                kind, start = piece[1], (row, col)
                if kind == "P":
                    direction, home, promotion_row = (-1, 6, 0) if color == WHITE else (1, 1, 7)
                    one = (row + direction, col)
                    if inside(one) and not self.piece_at(one):
                        if one[0] == promotion_row:
                            moves.extend(Move(start, one, p) for p in "QRBN")
                        else:
                            moves.append(Move(start, one))
                        two = (row + 2 * direction, col)
                        if row == home and not self.piece_at(two):
                            moves.append(Move(start, two))
                    for dc in (-1, 1):
                        end = (row + direction, col + dc)
                        if not inside(end):
                            continue
                        target = self.piece_at(end)
                        if target and target[0] != color and target[1] != "K":
                            if end[0] == promotion_row:
                                moves.extend(Move(start, end, p) for p in "QRBN")
                            else:
                                moves.append(Move(start, end))
                        if end == self.en_passant:
                            moves.append(Move(start, end, en_passant=True))
                elif kind == "N":
                    for dr, dc in KNIGHT_DELTAS:
                        end = (row + dr, col + dc)
                        if inside(end) and (not self.piece_at(end) or (self.piece_at(end)[0] != color and self.piece_at(end)[1] != "K")):
                            moves.append(Move(start, end))
                elif kind in "RBQ":
                    directions = []
                    if kind in "RQ": directions += [(1, 0), (-1, 0), (0, 1), (0, -1)]
                    if kind in "BQ": directions += [(1, 1), (1, -1), (-1, 1), (-1, -1)]
                    moves.extend(self._add_ray_moves(start, color, directions))
                elif kind == "K":
                    for dr, dc in KING_DELTAS:
                        end = (row + dr, col + dc)
                        if inside(end) and (not self.piece_at(end) or (self.piece_at(end)[0] != color and self.piece_at(end)[1] != "K")):
                            moves.append(Move(start, end))
                    moves.extend(self._castle_moves(color))
        return moves

    def _castle_moves(self, color: str) -> list[Move]:
        row = 7 if color == WHITE else 0
        enemy = opponent(color)
        if self.in_check(color):
            return []
        result = []
        if self.castling[color + "K"] and self.grid[row][7] == color + "R":
            if not self.grid[row][5] and not self.grid[row][6] and not self.is_attacked((row, 5), enemy) and not self.is_attacked((row, 6), enemy):
                result.append(Move((row, 4), (row, 6), castle=True))
        if self.castling[color + "Q"] and self.grid[row][0] == color + "R":
            if not self.grid[row][1] and not self.grid[row][2] and not self.grid[row][3] and not self.is_attacked((row, 3), enemy) and not self.is_attacked((row, 2), enemy):
                result.append(Move((row, 4), (row, 2), castle=True))
        return result

    def is_attacked(self, square: tuple[int, int], by_color: str) -> bool:
        # Generate attacks directly, avoiding recursive castling calls.
        row, col = square
        pawn_row = row + (1 if by_color == WHITE else -1)
        for dc in (-1, 1):
            src = (pawn_row, col + dc)
            if inside(src) and self.piece_at(src) == by_color + "P": return True
        for dr, dc in KNIGHT_DELTAS:
            src = (row + dr, col + dc)
            if inside(src) and self.piece_at(src) == by_color + "N": return True
        for dr, dc in KING_DELTAS:
            src = (row + dr, col + dc)
            if inside(src) and self.piece_at(src) == by_color + "K": return True
        for directions, kinds in [([(1,0),(-1,0),(0,1),(0,-1)], "RQ"), ([(1,1),(1,-1),(-1,1),(-1,-1)], "BQ")]:
            for dr, dc in directions:
                r, c = row + dr, col + dc
                while inside((r, c)):
                    p = self.piece_at((r, c))
                    if p:
                        if p[0] == by_color and p[1] in kinds: return True
                        break
                    r, c = r + dr, c + dc
        return False

    def king_square(self, color: str) -> tuple[int, int]:
        for r in range(8):
            for c in range(8):
                if self.grid[r][c] == color + "K": return (r, c)
        raise ValueError("Position has no king")

    def in_check(self, color: str) -> bool:
        return self.is_attacked(self.king_square(color), opponent(color))

    def legal_moves(self, color: Optional[str] = None) -> list[Move]:
        color = color or self.turn
        return [m for m in self.pseudo_legal_moves(color) if not self._leaves_king_in_check(m, color)]

    def _leaves_king_in_check(self, move: Move, color: str) -> bool:
        trial = self.copy()
        trial._apply(move, switch_turn=False)
        return trial.in_check(color)

    def _apply(self, move: Move, switch_turn: bool = True) -> None:
        piece = self.piece_at(move.start)
        assert piece is not None
        captured = self.piece_at(move.end)
        self.grid[move.start[0]][move.start[1]] = None
        if move.en_passant:
            captured = self.grid[move.start[0]][move.end[1]]
            self.grid[move.start[0]][move.end[1]] = None
        self.grid[move.end[0]][move.end[1]] = piece[0] + (move.promotion or piece[1])
        if move.castle:
            row = move.start[0]
            rook_start, rook_end = ((row, 7), (row, 5)) if move.end[1] == 6 else ((row, 0), (row, 3))
            self.grid[rook_end[0]][rook_end[1]] = self.grid[rook_start[0]][rook_start[1]]
            self.grid[rook_start[0]][rook_start[1]] = None
        # A rook moving *or being captured* on its original square removes the
        # corresponding right (the latter is easy to overlook).
        for color in (WHITE, BLACK):
            home = 7 if color == WHITE else 0
            for side, square in (("Q", (home, 0)), ("K", (home, 7))):
                if move.start == square or move.end == square:
                    self.castling[color + side] = False
        if piece[1] == "K": self.castling[piece[0] + "K"] = self.castling[piece[0] + "Q"] = False
        self.en_passant = ((move.start[0] + move.end[0]) // 2, move.start[1]) if piece[1] == "P" and abs(move.start[0] - move.end[0]) == 2 else None
        self.halfmove_clock = 0 if piece[1] == "P" or captured else self.halfmove_clock + 1
        if switch_turn:
            self.history.append(move)
            if self.turn == BLACK: self.fullmove_number += 1
            self.turn = opponent(self.turn)

    def push(self, move: Move) -> None:
        if move not in self.legal_moves(): raise ValueError("Illegal move")
        self._apply(move)

    def outcome(self) -> Optional[str]:
        if self.legal_moves(): return None
        return ("checkmate: " + ("Black" if self.turn == WHITE else "White") + " wins") if self.in_check(self.turn) else "stalemate"

    def state_key(self) -> tuple:
        """Hashable state intended as the Q-table key (compact baseline encoding)."""
        cells = tuple(piece or "--" for row in self.grid for piece in row)
        rights = tuple(k for k in sorted(self.castling) if self.castling[k])
        return cells, self.turn, rights, self.en_passant


class ChessEnv:
    """Minimal Gym-like adapter: reset(), legal_actions(), step(action)."""
    def __init__(self) -> None: self.board = Board()
    def reset(self) -> tuple:
        self.board.reset(); return self.board.state_key()
    def legal_actions(self) -> list[Move]: return self.board.legal_moves()
    def step(self, action: Move) -> tuple[tuple, float, bool, dict]:
        before = self.board.piece_at(action.end)
        self.board.push(action)
        result = self.board.outcome()
        done = result is not None
        reward = 1.0 if result and "wins" in result else (-0.01 if not before else 0.1)
        return self.board.state_key(), reward, done, {"outcome": result}
