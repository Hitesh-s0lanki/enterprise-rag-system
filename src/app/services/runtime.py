"""Bind demo identity, authoritative revisions and existing agent services."""

from dataclasses import dataclass

from src.agents import RootAgent
from src.data_ingestion.config import IngestionSettings
from src.data_ingestion.bootstrap import bootstrap_catalog
from src.db.auth import AuthorizationService, Permission, SQLiteUserRepository, User, initialize_database
from src.db.documents.repository import SQLiteDocumentRepository, initialize_document_catalog
from src.query.pipeline import KnowledgeQueryService


@dataclass
class AppRuntime:
    users: SQLiteUserRepository
    authorization: AuthorizationService
    catalog: SQLiteDocumentRepository
    knowledge: IngestionSettings

    @classmethod
    def from_env(cls):
        knowledge = IngestionSettings.from_env()
        # Both initializers are additive and repeatable. The auth seed runs only
        # once, preserving later user/permission edits and published revisions.
        # Missing demo revisions are recovered separately after provider verification.
        path = initialize_database()
        initialize_document_catalog(path)
        users = SQLiteUserRepository(path)
        authorization = AuthorizationService(users)
        catalog = SQLiteDocumentRepository(users.path)
        bootstrap_catalog(catalog, authorization, knowledge)
        return cls(users, authorization, catalog, knowledge)

    def revisions(self) -> dict[str, str]:
        return self.catalog.approved_revisions(self.knowledge.namespace)

    def authorize(self, user: User):
        return self.authorization.require_access(user.id, Permission.READ_KNOWLEDGE,
                                                  namespace=self.knowledge.namespace)

    def agent(self, user: User) -> RootAgent:
        self.authorize(user)
        return RootAgent(KnowledgeQueryService(
            actor_id=user.id, authorization=self.authorization, approved_revisions=self.revisions,
            knowledge=self.knowledge,
        ))

    def check_history(self, user: User, turns: list):
        """Recheck stored evidence before rendering or supplying history to a model."""
        self.authorize(user)
        revisions = self.revisions()
        for turn in turns:
            for report in turn.reports:
                for document in report.get("documents", []):
                    metadata = document.get("metadata", {})
                    roles = metadata.get("allowed_roles")
                    if not isinstance(roles, list) or not roles:
                        raise PermissionError("Stored evidence has no access roles")
                    self.authorization.require_access(user.id, Permission.READ_KNOWLEDGE,
                        namespace=self.knowledge.namespace, allowed_roles=roles)
                    if revisions.get(metadata.get("document_id")) != metadata.get("revision_id"):
                        raise PermissionError("Stored evidence is no longer current")
