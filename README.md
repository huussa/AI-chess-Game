# Chess with Pygame + Q-learning foundation

## Run

```powershell
py -m pip install -r requirements.txt
py main.py
```

Click a piece, then one of the green destinations. Press `R` to restart.

## Rules included

Legal movement for pawn, knight, bishop, rook, queen and king, plus captures, check/checkmate, stalemate, castling, en passant and promotion. The UI promotes to a queen automatically; the engine exposes all four promotion choices.

## Reinforcement-learning seam

`chess_engine.py` contains no Pygame dependency. Use `ChessEnv` for training:

```python
from chess_engine import ChessEnv

env = ChessEnv()
state = env.reset()
action = env.legal_actions()[0]  # later: epsilon-greedy choice from your Q table
next_state, reward, done, info = env.step(action)
```

`state_key()` returns an immutable position representation that can be used as a dictionary key. Actions are immutable `Move` objects, also suitable as Q-table keys. For practical full-board chess, tabular Q-learning will grow extremely large; this interface is the correct starting point, and a neural value/policy approximation can later replace the table without rewriting the game rules.
"# AI-chess-Game" 
