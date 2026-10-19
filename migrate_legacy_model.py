"""One-time, low-memory migration from the legacy JSON Q-table to SQLite."""
from pathlib import Path

from chess_ai import import_legacy_json

LOG_PATH = Path(__file__).with_name("legacy_migration.log")


def log(message: str) -> None:
    print(message, flush=True)
    with LOG_PATH.open("a", encoding="utf-8") as file:
        file.write(message + "\n")


def report(color: str, states: int) -> None:
    log(f"{color}: imported {states:,} states")


if __name__ == "__main__":
    log("Migration started")
    try:
        result = import_legacy_json(progress=report)
        log("Migration complete: " + str(result))
    except Exception as error:
        log("Migration failed: " + repr(error))
        raise
