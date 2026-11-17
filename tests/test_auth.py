"""Behavioral checks for the demo identity registry and permission policy."""

import unittest
import sqlite3
import tempfile
from dataclasses import replace
from contextlib import closing
from pathlib import Path
from uuid import UUID

from src.db.auth import AuthorizationService, SQLiteUserRepository, Permission, Role, initialize_database


class AuthorizationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = initialize_database(Path(self.temp.name) / "auth.sqlite3")
        self.repository = SQLiteUserRepository(self.path)
        self.service = AuthorizationService(self.repository)

    def check(self, username, permission, **kwargs):
        return self.service.require_access(self.repository.get_by_username(username).id, permission,
                                           namespace="it-support", **kwargs)

    def test_three_stable_users_and_roles(self):
        self.assertEqual([(u.display_name, u.role.value) for u in self.repository.list_users()],
                         [("Alice", "employee"), ("Bob", "contributor"), ("Mice", "administrator")])
        self.assertEqual(self.repository.get_by_username(" Alice ").username, "alice")
        self.assertEqual(self.repository.list_users(), SQLiteUserRepository(self.path).list_users())

    def test_employee_reads_but_cannot_write_or_manage(self):
        self.check("alice", Permission.READ_KNOWLEDGE, allowed_roles=["employee"])
        for permission in (Permission.WRITE_KNOWLEDGE, Permission.MANAGE_KNOWLEDGE, Permission.MANAGE_USERS):
            with self.assertRaises(PermissionError):
                self.check("alice", permission)

    def test_contributor_combined_bits_and_document_scope(self):
        self.check("bob", Permission.READ_KNOWLEDGE | Permission.WRITE_KNOWLEDGE,
                   allowed_roles=["contributor"])
        with self.assertRaises(PermissionError):
            self.check("bob", Permission.READ_KNOWLEDGE | Permission.MANAGE_USERS)
        with self.assertRaises(PermissionError):
            self.check("bob", Permission.WRITE_KNOWLEDGE, allowed_roles=["administrator"])

    def test_administrator_manages_namespace_but_not_other_namespaces(self):
        self.check("mice", Permission.MANAGE_USERS | Permission.WRITE_KNOWLEDGE,
                   allowed_roles=["employee"])
        with self.assertRaises(PermissionError):
            self.service.require_access(self.repository.get_by_username("mice").id,
                                        Permission.WRITE_KNOWLEDGE, namespace="finance")

    def test_unknown_inactive_and_invalid_document_roles_denied(self):
        with self.assertRaises(PermissionError):
            self.service.require_access(UUID(int=99), Permission.READ_KNOWLEDGE, namespace="it-support")
        inactive = replace(self.repository.get_by_username("alice"), is_active=False)
        self.repository.save_user(inactive)
        service = AuthorizationService(SQLiteUserRepository(self.path))
        with self.assertRaises(PermissionError):
            service.require_access(inactive.id, Permission.READ_KNOWLEDGE, namespace="it-support")
        for roles in ([], ["superuser"], "administrator"):
            with self.assertRaises(PermissionError):
                self.check("mice", Permission.WRITE_KNOWLEDGE, allowed_roles=roles)

    def test_duplicate_usernames_rejected(self):
        alice = self.repository.get_by_username("alice")
        duplicate = replace(alice, id=UUID(int=99), username="ALICE", role=Role.ADMINISTRATOR)
        with self.assertRaises(sqlite3.IntegrityError):
            self.repository.save_user(duplicate)
        self.assertEqual(self.repository.get_by_username("alice"), alice)

    def test_reinitialization_preserves_changes_and_deleted_users(self):
        bob = self.repository.get_by_username("bob")
        changed = replace(bob, role=Role.EMPLOYEE, namespaces=("finance",), is_active=False)
        self.repository.save_user(changed)
        with closing(sqlite3.connect(self.path)) as connection, connection:
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("DELETE FROM users WHERE username = ?", ("alice",))
        initialize_database(self.path)
        reopened = SQLiteUserRepository(self.path)
        self.assertEqual(reopened.get_by_username("bob"), changed)
        self.assertIsNone(reopened.get_by_username("alice"))
        self.assertEqual(len(reopened.list_users()), 2)

    def test_role_permissions_are_loaded_from_database(self):
        self.check("bob", Permission.WRITE_KNOWLEDGE)
        with closing(sqlite3.connect(self.path)) as connection, connection:
            connection.execute("UPDATE roles SET permissions = ? WHERE name = ?", (1, "contributor"))
        with self.assertRaises(PermissionError):
            self.check("bob", Permission.WRITE_KNOWLEDGE)

    def test_sql_parameters_and_foreign_keys(self):
        self.assertIsNone(self.repository.get_by_username("' OR 1=1 --"))
        alice = self.repository.get_by_username("alice")
        with closing(sqlite3.connect(self.path)) as connection, connection:
            connection.execute("PRAGMA foreign_keys = ON")
            with self.assertRaises(sqlite3.IntegrityError):
                connection.execute("INSERT INTO user_namespace_scopes VALUES (?, ?)", (str(UUID(int=99)), "finance"))
            with self.assertRaises(sqlite3.IntegrityError):
                connection.execute("UPDATE users SET role = ? WHERE id = ?", ("superuser", str(alice.id)))

    def test_failed_user_scope_update_rolls_back_entire_change(self):
        bob = self.repository.get_by_username("bob")
        with self.assertRaises(sqlite3.IntegrityError):
            self.repository.save_user(replace(bob, role=Role.EMPLOYEE, namespaces=("",)))
        self.assertEqual(self.repository.get_by_username("bob"), bob)


if __name__ == "__main__":
    unittest.main()
