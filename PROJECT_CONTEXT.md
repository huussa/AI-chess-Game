# Project context — Chess + Pygame + Q-learning

## Verified current state (2026-10-19)

- Python/Pygame chess project at the repository root. Runtime dependency: `pygame>=2.5`; no build step.
- UI flow: main menu → opponent selection → AI-training setup or local-player flow → color selection → board. `Esc` navigates back through menu screens.
- The board supports legal moves, check/checkmate/stalemate, castling, en passant, and promotion (UI auto-promotes to queen). The board is rotated when the human selects Black.
- Against AI, the human can move only on their turn; the computer is assigned the opposite color and selects its move from known Q-values. For an unseen state it uses depth-2 Minimax with alpha-beta pruning rather than a random fallback, then records the computer move and the player's reply into the disk-backed Q-table.
- Training UI accepts a positive integer. `TRAIN MODEL` continues data; `TRAIN FROM SCRATCH` clears the active SQLite tables. Training runs in a background thread and trains the White table for N episodes, then the Black table for N episodes. Training uses depth-2 Minimax with alpha-beta pruning for both sides, with at most 32 plies per episode.
- Current active model is SQLite at `chess_q_table`, not JSON. Current metadata: 101,100 episodes and 7,382,550 states for White; 101,100 episodes and 7,495,747 states for Black.
- The legacy JSON model is retained as `chess_q_table.legacy.json` (2,310,401,677 bytes). Its low-memory migration completed successfully; the log records 100,100 episodes and 7,382,550 / 7,495,747 imported states. The active database later increased to 101,100 episodes per color; the exact event causing the additional 1,000 is not recorded, but it is consistent with later training.
- The active SQLite database is currently 12,994,473,984 bytes. The one-time Task Scheduler task `CodexChessQTableMigration` is disabled and last reported success; it is outside the repository.

## Architecture

| Area | File(s) | Notes |
| --- | --- | --- |
| Pygame UI, menus, perspective, game loop | `main.py` | Owns screen states and starts training in a thread. |
| Chess rules / RL-neutral board | `chess_engine.py` | `Board`, `Move`, and lightweight `ChessEnv`; no Pygame import. |
| Q-learning, SQLite persistence, legacy import | `chess_ai.py` | `QTableStore` queries only the current state during play. |
| One-time JSON → SQLite migration | `migrate_legacy_model.py` | Imports incrementally; it resets the destination DB first. |
| Regression tests | `test_chess_engine.py`, `test_chess_ai.py` | Engine, persistence, and small legacy-import coverage. |

## Important decisions

- **SQLite instead of JSON:** JSON loading of the prior 2GB Q-table exhausted memory and crashed/froze game startup. SQLite avoids loading the entire model; gameplay queries the current `(color, state)` only.
- **White/Black separation:** Q-values retain a `color` key and episode/state metadata is separate per color.
- **Live-game learning:** Real player-vs-AI transitions may add new states even if a legacy import already exceeds the batch-training state cap. This is required to retain game experience; batch training continues to observe the cap.
- **Legacy preservation:** Migration reads `chess_q_table.legacy.json` incrementally and leaves it untouched. It does not use `json.load()`.
- **State cap for new training:** `MAX_STATES_PER_COLOR = 75_000` in `chess_ai.py` limits *newly admitted* states. The full legacy import intentionally bypassed that cap to preserve all old data, which explains why the current DB is still very large.
- **No claimed Elo:** The user reported very weak AI play. No benchmark, Elo measurement, minimax, opening book, neural network, or strong opponent exists.

## Known issues / follow-up work

- The active DB is ~13GB despite RAM-safe lookup. Disk use and lookup latency have not been benchmarked; importing all legacy states defeats the intended 75k-state storage bound.
- Q-learning quality is expected to be limited by high-dimensional full-board states, sparse/repetition-poor data, and the deliberately shallow (depth-2) Minimax training search. This is an architectural limitation, not an established Elo score.
- `chess_q_table.legacy.json` still consumes ~2.31GB. Keep it for backup; remove it only with explicit user approval after validating the SQLite model.
- Do not rerun `migrate_legacy_model.py` casually: it resets the active SQLite model before importing the legacy data again.
- `README.md` documents basic running/rules but not the current menu flow, SQLite model, or migration. No documentation reconciliation has been performed.
- No automated UI/integration test, model-quality evaluation, cancellation/resume for training, or migration progress UI exists.

## Last work and exact next step

The last substantive task was low-memory migration of the legacy JSON Q-table into SQLite; it completed successfully. This turn only created handoff documentation.

Before changing AI behavior, first manually start an AI game with the current SQLite model and measure move latency/memory. Then decide with the user whether to keep the full migrated database or explicitly prune/retrain to the 75,000-state limit. Do not delete either model file without approval.

## Commands

```powershell
py -m pip install -r requirements.txt
py main.py
py -m unittest -q
py -m py_compile main.py chess_engine.py chess_ai.py
```

Migration command — only if explicitly requested and after backing up/accepting the reset behavior:

```powershell
py migrate_legacy_model.py
```

## Repository observations

- At documentation time, Git reports modified `.gitignore`, `chess_ai.py`, `main.py`, and `test_chess_ai.py`; migration script/log files are untracked. The generated model files are ignored.
- No `AGENTS.md` existed before this handoff file was added.
