"""Tabular Q-learning support for the Pygame chess project.

The implementation deliberately keeps a separate table for each AI colour.
It is a small, transparent baseline rather than a high-performance chess
engine: full chess has a huge state space, so quality improves gradually as
more episodes are added.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Callable, Optional

from chess_engine import BLACK, WHITE, Board, Move, opponent

# JSON content, kept under the exact user-facing model filename.
TABLE_PATH = Path(__file__).with_name("chess_q_table")
PIECE_VALUE = {"P": 1.0, "N": 3.0, "B": 3.0, "R": 5.0, "Q": 9.0, "K": 0.0}


def empty_tables() -> dict:
    return {
        "format": "chess-q-learning-v1",
        "white": {},
        "black": {},
        "stats": {"white_episodes": 0, "black_episodes": 0},
    }


def load_tables(path: Path = TABLE_PATH) -> dict:
    if not path.exists():
        return empty_tables()
    try:
        with path.open("r", encoding="utf-8") as file:
            tables = json.load(file)
        if tables.get("format") != "chess-q-learning-v1":
            return empty_tables()
        tables.setdefault("white", {})
        tables.setdefault("black", {})
        tables.setdefault("stats", {"white_episodes": 0, "black_episodes": 0})
        return tables
    except (OSError, json.JSONDecodeError):
        # A damaged or partial file must not prevent the game from launching.
        return empty_tables()


def save_tables(tables: dict, path: Path = TABLE_PATH) -> None:
    """Atomically persist the learned values so an interrupted save is safe."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(tables, file, ensure_ascii=False, separators=(",", ":"))
    temporary.replace(path)


def state_id(board: Board) -> str:
    """Compact, JSON-safe state key containing all move-relevant position data."""
    squares = "".join(piece or "--" for row in board.grid for piece in row)
    rights = "".join(key for key in sorted(board.castling) if board.castling[key]) or "-"
    ep = "-" if board.en_passant is None else f"{board.en_passant[0]}{board.en_passant[1]}"
    return f"{squares}|{board.turn}|{rights}|{ep}"


def _table_name(color: str) -> str:
    return "white" if color == WHITE else "black"


def choose_move(board: Board, color: str, table: dict, epsilon: float, rng: random.Random) -> Move:
    legal = board.legal_moves()
    if not legal:
        raise ValueError("No legal move available")
    if rng.random() < epsilon:
        return rng.choice(legal)
    values = table.get(state_id(board), {})
    highest = max(values.get(move.uci(), 0.0) for move in legal)
    best = [move for move in legal if values.get(move.uci(), 0.0) == highest]
    return rng.choice(best)


def best_move(board: Board, color: str, tables: dict, rng: Optional[random.Random] = None) -> Move:
    """Choose the strongest learned legal move; falls back to a legal random move."""
    return choose_move(board, color, tables[_table_name(color)], 0.0, rng or random.Random())


def _reward_for_result(result: Optional[str], ai_color: str, captured: Optional[str]) -> float:
    reward = PIECE_VALUE.get(captured[1], 0.0) * 0.08 if captured else -0.005
    if not result:
        return reward
    if result == "stalemate":
        return reward + 0.2
    winner = WHITE if "White wins" in result else BLACK
    return reward + (10.0 if winner == ai_color else -10.0)


def train_color(
    tables: dict,
    color: str,
    episodes: int,
    progress: Optional[Callable[[str, int, int], None]] = None,
    rng: Optional[random.Random] = None,
    max_plies: int = 80,
) -> None:
    """Train one colour against a random legal-move opponent."""
    rng = rng or random.Random()
    table = tables[_table_name(color)]
    alpha, gamma = 0.18, 0.92
    for episode in range(episodes):
        board = Board()
        # Black's opponent makes the opening move before Black acts.
        if color == BLACK:
            board.push(rng.choice(board.legal_moves()))
        for _ in range(max_plies):
            if board.turn != color:
                legal = board.legal_moves()
                if not legal:
                    break
                board.push(rng.choice(legal))
            if board.outcome():
                break

            state = state_id(board)
            epsilon = max(0.06, 0.35 * (1 - episode / max(episodes, 1)))
            move = choose_move(board, color, table, epsilon, rng)
            captured = board.piece_at(move.end)
            board.push(move)
            result = board.outcome()

            if not result:
                opponent_moves = board.legal_moves()
                if opponent_moves:
                    board.push(rng.choice(opponent_moves))
                result = board.outcome()

            reward = _reward_for_result(result, color, captured)
            next_best = 0.0
            if not result and board.turn == color:
                next_values = table.get(state_id(board), {})
                next_legal = board.legal_moves()
                if next_legal:
                    next_best = max(next_values.get(next_move.uci(), 0.0) for next_move in next_legal)
            old_value = table.setdefault(state, {}).get(move.uci(), 0.0)
            table[state][move.uci()] = old_value + alpha * (reward + gamma * next_best - old_value)
            if result:
                break
        tables["stats"][_table_name(color) + "_episodes"] += 1
        if progress and ((episode + 1) % 25 == 0 or episode + 1 == episodes):
            progress(_table_name(color), episode + 1, episodes)


def train_both_colours(
    episodes: int,
    reset: bool = False,
    progress: Optional[Callable[[str, int, int], None]] = None,
    path: Path = TABLE_PATH,
) -> dict:
    """Train White for ``episodes`` and Black for ``episodes``, then save."""
    if episodes <= 0:
        raise ValueError("Training episodes must be greater than zero")
    tables = empty_tables() if reset else load_tables(path)
    if reset:
        save_tables(tables, path)  # Deliberately replaces the previous model.
    rng = random.Random()
    train_color(tables, WHITE, episodes, progress, rng)
    train_color(tables, BLACK, episodes, progress, rng)
    save_tables(tables, path)
    return tables
