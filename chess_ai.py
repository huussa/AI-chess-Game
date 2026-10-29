"""Disk-backed tabular Q-learning for the chess game.

The Q-table is stored in SQLite rather than one giant JSON dictionary. During
a game, only values for the current board state are queried, keeping RAM
usage stable after lengthy training runs.
"""
from __future__ import annotations

import random
import sqlite3
import json
from pathlib import Path
from typing import Callable, Optional

from chess_engine import BLACK, WHITE, Board, Move

TABLE_PATH = Path(__file__).with_name("chess_q_table")
LEGACY_PATH = Path(__file__).with_name("chess_q_table.legacy.json")
MAX_STATES_PER_COLOR = 75_000
PIECE_VALUE = {"P": 1.0, "N": 3.0, "B": 3.0, "R": 5.0, "Q": 9.0, "K": 0.0}
MINIMAX_TRAINING_DEPTH = 2
MAX_TRAINING_PLIES = 32
MATE_VALUE = 10_000.0


def _table_name(color: str) -> str:
    return "white" if color == WHITE else "black"


def state_id(board: Board) -> str:
    squares = "".join(piece or "--" for row in board.grid for piece in row)
    rights = "".join(key for key in sorted(board.castling) if board.castling[key]) or "-"
    ep = "-" if board.en_passant is None else f"{board.en_passant[0]}{board.en_passant[1]}"
    return f"{squares}|{board.turn}|{rights}|{ep}"


class QTableStore:
    """A bounded, on-disk Q-table with independent White and Black data."""

    def __init__(self, path: Path = TABLE_PATH) -> None:
        self.path = path
        self._preserve_legacy_json()
        self.connection = sqlite3.connect(self.path, timeout=30)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=NORMAL")
        self._create_schema()

    def _preserve_legacy_json(self) -> None:
        """Move the old JSON model aside without loading it into RAM."""
        if not self.path.exists():
            return
        with self.path.open("rb") as file:
            is_sqlite = file.read(16) == b"SQLite format 3\x00"
        if is_sqlite:
            return
        legacy = self.path.with_name(self.path.name + ".legacy.json")
        if legacy.exists():
            # Keep both existing files safe; use a sidecar database instead.
            self.path = self.path.with_name(self.path.name + ".sqlite")
            return
        self.path.replace(legacy)

    def _create_schema(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS states (
                color TEXT NOT NULL,
                state TEXT NOT NULL,
                PRIMARY KEY (color, state)
            );
            CREATE TABLE IF NOT EXISTS q_values (
                color TEXT NOT NULL,
                state TEXT NOT NULL,
                action TEXT NOT NULL,
                value REAL NOT NULL,
                visits INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (color, state, action)
            );
            CREATE INDEX IF NOT EXISTS q_values_state_idx ON q_values(color, state);
            """
        )
        for key in ("white_episodes", "black_episodes", "white_states", "black_states"):
            self.connection.execute("INSERT OR IGNORE INTO metadata(key, value) VALUES (?, 0)", (key,))
        self.connection.commit()

    def close(self) -> None:
        self.connection.commit()
        self.connection.close()

    def reset(self) -> None:
        with self.connection:
            self.connection.execute("DELETE FROM q_values")
            self.connection.execute("DELETE FROM states")
            for key in ("white_episodes", "black_episodes", "white_states", "black_states"):
                self.connection.execute("UPDATE metadata SET value = 0 WHERE key = ?", (key,))

    def _meta(self, key: str) -> int:
        return int(self.connection.execute("SELECT value FROM metadata WHERE key = ?", (key,)).fetchone()[0])

    def _set_meta(self, key: str, value: int) -> None:
        self.connection.execute("UPDATE metadata SET value = ? WHERE key = ?", (value, key))

    def values_for(self, color: str, state: str) -> dict[str, float]:
        rows = self.connection.execute(
            "SELECT action, value FROM q_values WHERE color = ? AND state = ?", (_table_name(color), state)
        )
        return {action: value for action, value in rows}

    def _admit_state(self, color: str, state: str, allow_overflow: bool = False) -> bool:
        name = _table_name(color)
        found = self.connection.execute(
            "SELECT 1 FROM states WHERE color = ? AND state = ?", (name, state)
        ).fetchone()
        if found:
            return True
        count_key = name + "_states"
        count = self._meta(count_key)
        if count >= MAX_STATES_PER_COLOR and not allow_overflow:
            return False
        self.connection.execute("INSERT INTO states(color, state) VALUES (?, ?)", (name, state))
        self._set_meta(count_key, count + 1)
        return True

    def update(self, color: str, state: str, action: str, value: float, *, allow_overflow: bool = False) -> None:
        if not self._admit_state(color, state, allow_overflow):
            return
        name = _table_name(color)
        self.connection.execute(
            """
            INSERT INTO q_values(color, state, action, value, visits)
            VALUES (?, ?, ?, ?, 1)
            ON CONFLICT(color, state, action) DO UPDATE SET
                value = excluded.value,
                visits = q_values.visits + 1
            """,
            (name, state, action, value),
        )

    def increment_episodes(self, color: str) -> None:
        key = _table_name(color) + "_episodes"
        self.connection.execute("UPDATE metadata SET value = value + 1 WHERE key = ?", (key,))

    def stats(self) -> dict[str, int]:
        keys = ("white_episodes", "black_episodes", "white_states", "black_states")
        return {key: self._meta(key) for key in keys}


class _StreamingJson:
    """Small JSON reader that never retains a complete large JSON object."""

    def __init__(self, path: Path) -> None:
        self.file = path.open("r", encoding="utf-8")
        self.buffer = ""
        self.position = 0
        self.decoder = json.JSONDecoder()
        self.finished = False

    def close(self) -> None:
        self.file.close()

    def _compact(self) -> None:
        if self.position > 65_536:
            self.buffer = self.buffer[self.position:]
            self.position = 0

    def _fill(self) -> bool:
        self._compact()
        chunk = self.file.read(65_536)
        if not chunk:
            self.finished = True
            return False
        self.buffer += chunk
        return True

    def _skip_space(self) -> None:
        while True:
            while self.position < len(self.buffer) and self.buffer[self.position].isspace():
                self.position += 1
            if self.position < len(self.buffer) or not self._fill():
                return

    def peek(self) -> str:
        self._skip_space()
        if self.position >= len(self.buffer):
            raise ValueError("Unexpected end of legacy Q-table")
        return self.buffer[self.position]

    def expect(self, character: str) -> None:
        if self.peek() != character:
            raise ValueError(f"Expected {character!r} in legacy Q-table")
        self.position += 1

    def value(self):
        self._skip_space()
        while True:
            try:
                decoded, end = self.decoder.raw_decode(self.buffer, self.position)
                self.position = end
                self._compact()
                return decoded
            except json.JSONDecodeError:
                if not self._fill():
                    raise ValueError("Invalid or incomplete legacy Q-table")


def _read_legacy_table(stream: _StreamingJson, color: str, write_row, progress) -> int:
    stream.expect("{")
    states = 0
    while stream.peek() != "}":
        state = stream.value()
        if not isinstance(state, str):
            raise ValueError("Legacy Q-table has an invalid state key")
        stream.expect(":")
        actions = stream.value()
        if not isinstance(actions, dict):
            raise ValueError("Legacy Q-table has invalid action data")
        write_row(color, state, actions)
        states += 1
        if progress and states % 10_000 == 0:
            progress(_table_name(color), states)
        separator = stream.peek()
        if separator == ",":
            stream.expect(",")
        elif separator != "}":
            raise ValueError("Legacy Q-table has an invalid table separator")
    stream.expect("}")
    return states


def import_legacy_json(
    legacy_path: Path = LEGACY_PATH,
    path: Path = TABLE_PATH,
    progress: Optional[Callable[[str, int], None]] = None,
) -> dict[str, int]:
    """Copy an old JSON model into SQLite without loading it all into RAM.

    The source file is deliberately left untouched. The destination database is
    reset first because this is a complete model import, not incremental train.
    """
    if not legacy_path.exists():
        raise FileNotFoundError(f"Legacy model not found: {legacy_path.name}")
    store = QTableStore(path)
    stream = _StreamingJson(legacy_path)
    pending_values = []
    pending_states = []
    imported = {"white": 0, "black": 0}
    legacy_stats = {}

    def flush() -> None:
        if not pending_values:
            return
        store.connection.executemany(
            "INSERT OR IGNORE INTO states(color, state) VALUES (?, ?)", pending_states
        )
        store.connection.executemany(
            """
            INSERT INTO q_values(color, state, action, value, visits)
            VALUES (?, ?, ?, ?, 1)
            ON CONFLICT(color, state, action) DO UPDATE SET value = excluded.value
            """,
            pending_values,
        )
        store.connection.commit()
        pending_values.clear()
        pending_states.clear()

    def write_row(color: str, state: str, actions: dict) -> None:
        name = _table_name(color)
        imported[name] += 1
        pending_states.append((name, state))
        for action, value in actions.items():
            pending_values.append((name, state, action, float(value)))
        if len(pending_values) >= 5_000:
            flush()

    try:
        store.reset()
        stream.expect("{")
        while stream.peek() != "}":
            key = stream.value()
            stream.expect(":")
            if key == "white":
                _read_legacy_table(stream, WHITE, write_row, progress)
            elif key == "black":
                _read_legacy_table(stream, BLACK, write_row, progress)
            else:
                value = stream.value()
                if key == "stats" and isinstance(value, dict):
                    legacy_stats = value
            separator = stream.peek()
            if separator == ",":
                stream.expect(",")
            elif separator != "}":
                raise ValueError("Legacy Q-table has an invalid top-level separator")
        stream.expect("}")
        flush()
        with store.connection:
            store._set_meta("white_states", imported["white"])
            store._set_meta("black_states", imported["black"])
            store._set_meta("white_episodes", int(legacy_stats.get("white_episodes", 0)))
            store._set_meta("black_episodes", int(legacy_stats.get("black_episodes", 0)))
        return store.stats()
    finally:
        stream.close()
        store.close()


def choose_move(board: Board, color: str, store: QTableStore, epsilon: float, rng: random.Random) -> Move:
    legal = board.legal_moves()
    if not legal:
        raise ValueError("No legal move available")
    if rng.random() < epsilon:
        return rng.choice(legal)
    values = store.values_for(color, state_id(board))
    highest = max(values.get(move.uci(), 0.0) for move in legal)
    best = [move for move in legal if values.get(move.uci(), 0.0) == highest]
    return rng.choice(best)


def best_move(board: Board, color: str, store: QTableStore, rng: Optional[random.Random] = None) -> Move:
    legal = board.legal_moves()
    if not legal:
        raise ValueError("No legal move available")
    values = store.values_for(color, state_id(board))
    known_legal = [move for move in legal if move.uci() in values]
    # Q-learning stores exact positions.  An unseen position uses the tactical
    # search rather than a random fallback, then becomes a learnable state.
    if not known_legal:
        return minimax_move(board, color)
    highest = max(values[move.uci()] for move in known_legal)
    best = [move for move in known_legal if values[move.uci()] == highest]
    return (rng or random.Random()).choice(best)


def _position_value(board: Board, perspective: str) -> float:
    """Evaluate a position from one side's perspective.

    Material remains the main signal; the small centre/development bonus only
    breaks otherwise equal opening moves, avoiding arbitrary pawn shuffles.
    """
    value = 0.0
    for row, cells in enumerate(board.grid):
        for column, piece in enumerate(cells):
            if not piece:
                continue
            sign = 1.0 if piece[0] == perspective else -1.0
            kind = piece[1]
            material = PIECE_VALUE[kind]
            centre = 3.5 - max(abs(row - 3.5), abs(column - 3.5))
            positional = 0.03 * max(centre, 0.0) if kind in "PNB" else 0.0
            if kind == "P":
                advance = (6 - row) if piece[0] == WHITE else (row - 1)
                positional += 0.02 * advance
            value += sign * (material + positional)
    return value


def _move_priority(board: Board, move: Move) -> float:
    """Order forcing moves first so alpha-beta can prune more branches."""
    captured = board.piece_at(move.end)
    if move.en_passant:
        captured = board.piece_at((move.start[0], move.end[1]))
    capture_value = PIECE_VALUE.get(captured[1], 0.0) if captured else 0.0
    promotion_value = PIECE_VALUE.get(move.promotion or "", 0.0)
    return capture_value * 10 + promotion_value


def _minimax_value(board: Board, perspective: str, depth: int, alpha: float, beta: float) -> float:
    legal = board.legal_moves()
    if not legal:
        if board.in_check(board.turn):
            return -MATE_VALUE if board.turn == perspective else MATE_VALUE
        return 0.0  # Stalemate.
    if depth == 0:
        return _position_value(board, perspective)

    maximizing = board.turn == perspective
    ordered_moves = sorted(legal, key=lambda move: _move_priority(board, move), reverse=maximizing)
    if maximizing:
        value = float("-inf")
        for move in ordered_moves:
            trial = board.copy()
            # ``move`` came from ``legal_moves`` above; applying it directly
            # avoids regenerating every legal move at every search node.
            trial._apply(move)
            value = max(value, _minimax_value(trial, perspective, depth - 1, alpha, beta))
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return value

    value = float("inf")
    for move in ordered_moves:
        trial = board.copy()
        trial._apply(move)
        value = min(value, _minimax_value(trial, perspective, depth - 1, alpha, beta))
        beta = min(beta, value)
        if alpha >= beta:
            break
    return value


def minimax_move(board: Board, color: str, depth: int = MINIMAX_TRAINING_DEPTH) -> Move:
    """Choose a tactical move using Minimax with alpha-beta pruning.

    This is deliberately shallow: it produces purposeful training opponents
    without turning every Q-learning episode into an impractically long search.
    """
    if board.turn != color:
        raise ValueError("Minimax color must match the side to move")
    if depth < 1:
        raise ValueError("Minimax depth must be at least one")
    legal = board.legal_moves()
    if not legal:
        raise ValueError("No legal move available")
    best_value, best = float("-inf"), []
    alpha, beta = float("-inf"), float("inf")
    for move in sorted(legal, key=lambda candidate: _move_priority(board, candidate), reverse=True):
        trial = board.copy()
        trial._apply(move)
        value = _minimax_value(trial, color, depth - 1, alpha, beta)
        if value > best_value:
            best_value, best = value, [move]
        elif value == best_value:
            best.append(move)
        alpha = max(alpha, best_value)
    # Keep equivalent choices stable between runs rather than choosing randomly.
    return min(best, key=Move.uci)


def _reward_for_result(result: Optional[str], ai_color: str, captured: Optional[str]) -> float:
    reward = PIECE_VALUE.get(captured[1], 0.0) * 0.08 if captured else -0.005
    if not result:
        return reward
    if result == "stalemate":
        return reward + 0.2
    winner = WHITE if "White wins" in result else BLACK
    return reward + (10.0 if winner == ai_color else -10.0)


def record_game_transition(
    store: QTableStore,
    color: str,
    state: str,
    action: Move,
    captured_by_ai: Optional[str],
    board_after_reply: Board,
    captured_by_opponent: Optional[str] = None,
    result: Optional[str] = None,
) -> None:
    """Learn one computer move from a real match and retain it on disk.

    Live games may add a small number of new states even when an imported
    legacy model already exceeds the training state cap.  This is intentionally
    limited to player-vs-AI experience; batch training keeps its normal cap.
    """
    result = result if result is not None else board_after_reply.outcome()
    reward = _reward_for_result(result, color, captured_by_ai)
    if captured_by_opponent:
        reward -= PIECE_VALUE.get(captured_by_opponent[1], 0.0) * 0.08
    next_best = 0.0
    if not result and board_after_reply.turn == color:
        next_values = store.values_for(color, state_id(board_after_reply))
        known_next = [
            next_values[move.uci()]
            for move in board_after_reply.legal_moves()
            if move.uci() in next_values
        ]
        if known_next:
            next_best = max(known_next)
    old_value = store.values_for(color, state).get(action.uci(), 0.0)
    alpha, gamma = 0.18, 0.92
    store.update(
        color,
        state,
        action.uci(),
        old_value + alpha * (reward + gamma * next_best - old_value),
        allow_overflow=True,
    )


def train_color(
    store: QTableStore,
    color: str,
    episodes: int,
    progress: Optional[Callable[[str, int, int], None]] = None,
    max_plies: int = MAX_TRAINING_PLIES,
) -> None:
    alpha, gamma = 0.18, 0.92
    for episode in range(episodes):
        board = Board()
        if color == BLACK:
            board.push(minimax_move(board, WHITE))
        for _ in range(max_plies):
            state = state_id(board)
            # Both sides follow the same Minimax policy during training.  This
            # replaces random exploration with positions that have tactical or
            # material significance, while Q-learning still records its values.
            move = minimax_move(board, color)
            captured = board.piece_at(move.end)
            board.push(move)
            result = board.outcome()
            if not result:
                board.push(minimax_move(board, board.turn))
                result = board.outcome()

            reward = _reward_for_result(result, color, captured)
            next_best = 0.0
            if not result and board.turn == color:
                next_values = store.values_for(color, state_id(board))
                next_legal = board.legal_moves()
                if next_legal:
                    next_best = max(next_values.get(next_move.uci(), 0.0) for next_move in next_legal)
            old_value = store.values_for(color, state).get(move.uci(), 0.0)
            store.update(color, state, move.uci(), old_value + alpha * (reward + gamma * next_best - old_value))
            if result:
                break
        store.increment_episodes(color)
        if (episode + 1) % 25 == 0 or episode + 1 == episodes:
            store.connection.commit()
            if progress:
                progress(_table_name(color), episode + 1, episodes)
    store.connection.commit()


def train_both_colours(
    episodes: int,
    reset: bool = False,
    progress: Optional[Callable[[str, int, int], None]] = None,
    path: Path = TABLE_PATH,
    max_plies: int = MAX_TRAINING_PLIES,
) -> dict[str, int]:
    if episodes <= 0:
        raise ValueError("Training episodes must be greater than zero")
    store = QTableStore(path)
    try:
        if reset:
            store.reset()
        train_color(store, WHITE, episodes, progress, max_plies)
        train_color(store, BLACK, episodes, progress, max_plies)
        return store.stats()
    finally:
        store.close()
