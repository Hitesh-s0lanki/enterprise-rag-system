PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS roles (
    name TEXT PRIMARY KEY CHECK (name IN ('employee', 'contributor', 'administrator')),
    permissions INTEGER NOT NULL CHECK (permissions BETWEEN 0 AND 15)
);

CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY NOT NULL,
    username TEXT NOT NULL COLLATE NOCASE UNIQUE CHECK (length(trim(username)) > 0),
    display_name TEXT NOT NULL,
    role TEXT NOT NULL REFERENCES roles(name),
    is_active INTEGER NOT NULL DEFAULT 1 CHECK (is_active IN (0, 1)),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS user_namespace_scopes (
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    namespace TEXT NOT NULL CHECK (length(trim(namespace)) > 0),
    PRIMARY KEY (user_id, namespace)
);

CREATE TABLE IF NOT EXISTS bootstrap_history (
    name TEXT PRIMARY KEY,
    applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
