CREATE TABLE IF NOT EXISTS knowledge_documents (
    namespace TEXT NOT NULL,
    document_id TEXT NOT NULL CHECK(length(document_id)=64),
    current_revision TEXT NOT NULL CHECK(length(current_revision)=64),
    reader_roles_json TEXT NOT NULL,
    writer_roles_json TEXT NOT NULL,
    pending_update_id TEXT UNIQUE,
    PRIMARY KEY (namespace, document_id)
);
CREATE TABLE IF NOT EXISTS knowledge_revisions (
    update_id TEXT PRIMARY KEY,
    namespace TEXT NOT NULL,
    document_id TEXT NOT NULL,
    revision_id TEXT NOT NULL CHECK(length(revision_id)=64),
    parent_revision TEXT,
    content TEXT NOT NULL CHECK(length(content)>0),
    metadata_json TEXT NOT NULL,
    actor_id TEXT NOT NULL REFERENCES users(id),
    state TEXT NOT NULL CHECK(state IN ('staged','published','failed')),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    published_at TEXT,
    UNIQUE(namespace, document_id, revision_id),
    FOREIGN KEY(namespace, document_id) REFERENCES knowledge_documents(namespace, document_id)
);
