"""Stream root-agent events as JSONL or SSE from the local demo CLI."""

import argparse
import sys

from dotenv import load_dotenv

from src.db.auth import AuthorizationService, Permission, SQLiteUserRepository
from src.query.pipeline import KnowledgeQueryService
from src.retrieval.service import validate_revisions
from src.utils.logger import configure_logging

from .root_agent import RootAgent


def main():
    parser = argparse.ArgumentParser(description="Run the root agent with streamed model and tool events")
    parser.add_argument("query")
    parser.add_argument("--user", required=True, help="Local demo username from SQLite")
    parser.add_argument("--revision", required=True, action="append", metavar="DOCUMENT_ID:REVISION_ID",
                        help="Trusted approved revision pair; repeat for additional documents")
    parser.add_argument("--format", choices=["jsonl", "sse"], default="jsonl")
    args = parser.parse_args()
    load_dotenv(override=False)
    configure_logging()
    repository = SQLiteUserRepository()
    user = repository.get_by_username(args.user)
    if user is None:
        parser.error("Unknown demo user")
    try:
        revisions = {}
        for pair in args.revision:
            document_id, revision_id = pair.split(":")
            if document_id in revisions and revisions[document_id] != revision_id:
                raise ValueError("Only one approved version per document may be selected")
            revisions[document_id] = revision_id
        revisions = validate_revisions(revisions)
        service = KnowledgeQueryService(actor_id=user.id, authorization=AuthorizationService(repository),
                                        approved_revisions=revisions)
        service.authorization.require_access(user.id, Permission.READ_KNOWLEDGE,
                                              namespace=service.knowledge.namespace)
    except (ValueError, PermissionError):
        parser.error("Invalid revision/configuration or unauthorized knowledge scope")
    agent = RootAgent(service)
    status = 0
    try:
        for event in agent.stream(args.query):
            if args.format == "sse":
                sys.stdout.write(event.to_sse())
                sys.stdout.flush()
            else:
                print(event.to_json(), flush=True)
            if event.type == "run_error":
                status = 1
    except KeyboardInterrupt:
        status = 130
    return status


if __name__ == "__main__":
    raise SystemExit(main())
