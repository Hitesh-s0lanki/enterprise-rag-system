"""SQLite setup and explicit, one-time seed initialization."""

import json
import os
import sqlite3
from contextlib import closing
from pathlib import Path

DEFAULT_DATABASE_PATH = Path(__file__).resolve().parents[3] / ".local" / "auth.sqlite3"


def database_path(path: str | Path | None = None) -> Path:
    return Path(path or os.getenv("AUTH_DB_PATH") or DEFAULT_DATABASE_PATH).expanduser().resolve()


def connect_database(path: Path) -> sqlite3.Connection:
    """Open an existing database; a typo must not silently create an empty DB."""
    connection = sqlite3.connect(f"{path.as_uri()}?mode=rw", uri=True, timeout=10)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def initialize_database(path: str | Path | None = None) -> Path:
    """Create tables and seed once; later initialization preserves DB changes."""
    target = database_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    resources = Path(__file__).parent
    with closing(sqlite3.connect(target, timeout=10)) as connection:
        connection.executescript((resources / "schema.sql").read_text(encoding="utf-8"))
        with connection:
            connection.execute("BEGIN IMMEDIATE")
            if connection.execute(
                "SELECT 1 FROM bootstrap_history WHERE name = ?", ("demo_users_v1",),
            ).fetchone():
                return target
            seed = json.loads((resources / "seed.json").read_text(encoding="utf-8"))
            connection.executemany(
                "INSERT INTO roles (name, permissions) VALUES (:name, :permissions)", seed["roles"],
            )
            for user in seed["users"]:
                connection.execute(
                    "INSERT INTO users (id, username, display_name, role) VALUES (?, ?, ?, ?)",
                    (user["id"], user["username"], user["display_name"], user["role"]),
                )
                connection.executemany(
                    "INSERT INTO user_namespace_scopes (user_id, namespace) VALUES (?, ?)",
                    [(user["id"], namespace) for namespace in user["namespaces"]],
                )
            connection.execute("INSERT INTO bootstrap_history (name) VALUES (?)", ("demo_users_v1",))
    return target
