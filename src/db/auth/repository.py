"""Parameterized SQLite queries behind a PostgreSQL-replaceable interface."""

from contextlib import closing
from pathlib import Path
from typing import Protocol
from uuid import UUID

from .database import connect_database, database_path
from .models import Permission, Role, User


class UserRepository(Protocol):
    def get_by_id(self, user_id: UUID) -> User | None: ...
    def get_by_username(self, username: str) -> User | None: ...
    def list_users(self) -> tuple[User, ...]: ...
    def get_permissions(self, role: Role) -> Permission: ...
    def save_user(self, user: User) -> None: ...


class SQLiteUserRepository:
    def __init__(self, path: str | Path | None = None):
        self.path = database_path(path)
        if not self.path.is_file():
            raise FileNotFoundError("Auth database is missing. Run: uv run python -m src.db.auth init")

    def _load_users(self, where: str = "", parameters: tuple = ()) -> tuple[User, ...]:
        # The optional predicate is fixed by the methods below; all values are bound.
        sql = """
            SELECT u.id, u.username, u.display_name, u.role, u.is_active, s.namespace
            FROM users u LEFT JOIN user_namespace_scopes s ON u.id = s.user_id
        """ + where + " ORDER BY u.username, s.namespace"
        with closing(connect_database(self.path)) as connection:
            rows = connection.execute(sql, parameters).fetchall()
        grouped = {}
        for row in rows:
            grouped.setdefault(row["id"], {"row": row, "namespaces": []})
            if row["namespace"] is not None:
                grouped[row["id"]]["namespaces"].append(row["namespace"])
        return tuple(
            User(UUID(key), value["row"]["username"], value["row"]["display_name"],
                 Role(value["row"]["role"]), tuple(value["namespaces"]), bool(value["row"]["is_active"]))
            for key, value in grouped.items()
        )

    def get_by_id(self, user_id: UUID) -> User | None:
        users = self._load_users(" WHERE u.id = ?", (str(user_id),))
        return users[0] if users else None

    def get_by_username(self, username: str) -> User | None:
        users = self._load_users(" WHERE u.username = ?", (username.strip().casefold(),))
        return users[0] if users else None

    def list_users(self) -> tuple[User, ...]:
        return self._load_users()

    def get_permissions(self, role: Role) -> Permission:
        with closing(connect_database(self.path)) as connection:
            row = connection.execute("SELECT permissions FROM roles WHERE name = ?", (role.value,)).fetchone()
        return Permission(row["permissions"]) if row else Permission(0)

    def save_user(self, user: User) -> None:
        """Atomically persist a user and scopes; callers must authorize management."""
        with closing(connect_database(self.path)) as connection, connection:
            connection.execute(
                """INSERT INTO users (id, username, display_name, role, is_active) VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(id) DO UPDATE SET username = excluded.username,
                   display_name = excluded.display_name, role = excluded.role,
                   is_active = excluded.is_active, updated_at = CURRENT_TIMESTAMP""",
                (str(user.id), user.username.strip().casefold(), user.display_name, user.role.value, user.is_active),
            )
            connection.execute("DELETE FROM user_namespace_scopes WHERE user_id = ?", (str(user.id),))
            connection.executemany(
                "INSERT INTO user_namespace_scopes (user_id, namespace) VALUES (?, ?)",
                [(str(user.id), namespace) for namespace in sorted(set(user.namespaces))],
            )
