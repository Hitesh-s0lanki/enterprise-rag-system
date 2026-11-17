# SQLite Users and RBAC

Auth lives under `src/db/auth` and uses Python's standard-library `sqlite3`. The repository reads users, role permission masks and namespace scopes from database tables. It returns frozen dataclass records with stable UUIDs and string roles; service checks use integer bit flags. There are no hardcoded user records or role permission masks in repository/service code.

Initialize the local database once from the project root:

```sh
uv run python -m src.db.auth init
```

The default file is `.local/auth.sqlite3` under the project root. Set `AUTH_DB_PATH` in `.env` or pass `--db-path PATH` to the initialization command to use another location. Ingestion reads `AUTH_DB_PATH` from `.env` too. Use the same path when initializing and running ingestion. SQLite database and journal files are ignored by Git.

`schema.sql` creates the tables. `seed.json` contains the initial roles and three requested users; initialization inserts them in one transaction and records `demo_users_v1` in `bootstrap_history`. Re-running initialization preserves edits, deactivated/deleted users, scopes and permission changes. Ingestion never reseeds the database.

| User | Username | Role | Permission mask | Capabilities |
| --- | --- | --- | --- | --- |
| Alice | `alice` | `employee` | `1` | Read/query permitted knowledge. |
| Bob | `bob` | `contributor` | `3` | Read and ingest permitted knowledge. |
| Mice | `mice` | `administrator` | `15` | Read, ingest and manage knowledge and users within assigned scope. |

All three are initially assigned the `it-support` namespace. The table above shows initial seed permissions; current values come from SQLite. Change existing users/scopes through `SQLiteUserRepository.save_user` after authorizing the management operation, or use a trusted database maintenance session. Role permission masks live in `roles`. Editing `seed.json` does not change an initialized database. No wildcard namespace grants are used.

`AuthorizationService.require_access` resolves the user through the repository, rejects unknown/inactive users, checks `(granted & required) == required`, and enforces namespace scope. When document roles are provided, non-administrators must have their exact role in `allowed_roles`; there is no implicit role inheritance. Administrators can access all documents within their assigned namespaces. Empty or unknown document roles are denied even for administrators.

The ingestion service requires a user ID and an authorization service. It checks write access and reader-role scope on every chunk before any embedding or upload. The CLI also checks before opening external clients. Index creation through `--create-index` requires `MANAGE_KNOWLEDGE`. Upload metadata records `ingested_by_user_id` and `ingested_by` separately from the source's `last_updated_by`.

Initial ingestion roles come from submitted metadata. The optional [document update tools](document-update-tools.md) use an authoritative catalog with stored reader/writer ACLs and transactional revision publication. Documents must be enrolled by trusted maintainer code before editing; the auth registry alone does not enroll them or authorize caller-supplied replacement ACLs.

```sh
# Bob can prepare/ingest documents permitting contributors.
uv run python -m src.data_ingestion src/data --user bob \
  --allowed-role employee --allowed-role contributor --allowed-role administrator --dry-run

# Alice is denied write access, including for an ingestion dry run.
uv run python -m src.data_ingestion src/data --user alice --allowed-role employee --dry-run
```

This is a local demo identity selector, not authentication: anyone with CLI access can select `--user mice`. A future authenticated frontend must resolve the actor's ID from its trusted session. Hybrid retrieval in `src/retrieval` now checks read permission, namespace scope and document-role access before returning source chunks. Cache access, document deletion and user-management endpoints are not built yet; they must call the same authorization service when added. Having a management permission does not create those endpoints.

## PostgreSQL migration

Implement `UserRepository` in a PostgreSQL adapter and inject it into `AuthorizationService`; policy code and ingestion user IDs can stay the same. Keep the current UUIDs during migration. The SQLite adapter uses parameterized SQL, foreign keys, unique usernames, and transactional user/scope writes.

Suggested storage mapping:

| Table | Fields |
| --- | --- |
| `roles` | `name TEXT PRIMARY KEY`, `permissions INTEGER` |
| `users` | `id UUID PRIMARY KEY`, `username TEXT` (case-insensitive unique), `display_name TEXT`, `role TEXT REFERENCES roles(name)`, `is_active BOOLEAN`, creation/update timestamps |
| `user_namespace_scopes` | `user_id UUID REFERENCES users(id)`, `namespace TEXT`, composite primary key `(user_id, namespace)` |

SQLite currently stores UUIDs as text and active flags as checked integers; map these to native UUID and boolean columns in PostgreSQL. The adapter normalizes usernames using `strip().casefold()` before writing; retain that normalization and uniqueness rule in the new adapter. Reconstruct `User.namespaces` from the scope rows and load permission masks from `roles`. Preserve existing permission bit values. Repository write methods are storage primitives; authenticated management endpoints and document revision publication remain separate work.
