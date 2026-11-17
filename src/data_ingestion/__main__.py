"""Run with: uv run python -m src.data_ingestion --help."""

import argparse
import json
from datetime import datetime

from dotenv import load_dotenv

from src.db.auth import AuthorizationService, Permission, Role, SQLiteUserRepository

from .config import IngestionSettings
from .embeddings import create_embedding_client
from src.clients.pinecone_client import create_pinecone_client, get_pinecone_index, get_index_embedding_mode
from .pipeline import ingest_documents, prepare_documents


def main():
    parser = argparse.ArgumentParser(description="Ingest UTF-8 text/Markdown into Pinecone for hybrid retrieval")
    parser.add_argument("source", nargs="?", default="src/data", help="A text file or directory")
    parser.add_argument("--user", required=True, help="Demo acting user: alice, bob or mice")
    parser.add_argument("--updated-by", help="Source's last editor; defaults to the acting username")
    parser.add_argument("--allowed-role", action="append", required=True, choices=[role.value for role in Role],
                        help="Allowed reader role; repeat as needed")
    parser.add_argument("--updated-at", help="Source update time in ISO 8601 with timezone; otherwise file mtime")
    parser.add_argument("--dry-run", action="store_true", help="Print chunk metadata without calling either provider")
    parser.add_argument("--create-index", action="store_true", help="Create a missing hybrid-ready index")
    args = parser.parse_args()
    load_dotenv(override=False)
    settings = IngestionSettings.from_env()
    repository = SQLiteUserRepository()
    authorization = AuthorizationService(repository)
    user = repository.get_by_username(args.user)
    if user is None:
        parser.error("Unknown demo user. Choose alice, bob or mice.")
    try:
        authorization.require_access(user.id, Permission.WRITE_KNOWLEDGE,
                                     namespace=settings.namespace, allowed_roles=args.allowed_role)
        if args.create_index:
            authorization.require_access(user.id, Permission.MANAGE_KNOWLEDGE, namespace=settings.namespace)
    except PermissionError as exc:
        parser.error(str(exc))
    updated_at = datetime.fromisoformat(args.updated_at) if args.updated_at else None
    chunks = prepare_documents(
        args.source, settings, updated_by=args.updated_by or user.username,
        allowed_roles=args.allowed_role, updated_at=updated_at,
    )
    if args.dry_run:
        print(json.dumps({"chunks": len(chunks), "metadata": [c.metadata for c in chunks]}, indent=2))
        return
    with create_pinecone_client() as client:
        with get_pinecone_index(client, settings, create=args.create_index) as index:
            mode = get_index_embedding_mode(client, settings)
            embeddings = create_embedding_client(settings) if mode == "external" else None
            print(json.dumps(ingest_documents(
                chunks, settings, embedding_client=embeddings, index=index,
                actor_id=user.id, authorization=authorization, embedding_mode=mode,
            ), indent=2))


if __name__ == "__main__":
    main()
