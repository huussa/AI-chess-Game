# Chess project instructions

- Keep chess rules in `chess_engine.py` independent of Pygame; UI belongs in `main.py`.
- Keep Q-learning persistence disk-backed through `QTableStore` in `chess_ai.py`. Never load `chess_q_table` as one JSON object.
- Model databases and SQLite sidecar files are ignored by Git. Migration logs are local artifacts; do not commit, delete, or overwrite any of these without explicit user approval.
- Preserve the separate White and Black Q-data paths.
- Run `py -m unittest -q` after changes to chess rules or Q-table persistence. Run `py main.py` for a manual UI check.
