"""Local maintainer CLI: uv run python -m src.retrieval --help."""

import argparse
import json
from dataclasses import asdict

from dotenv import load_dotenv

from src.clients.pinecone_client import create_pinecone_client, get_pinecone_index, get_index_embedding_mode
from src.clients.jev_client import create_jev_client
from src.data_ingestion.config import IngestionSettings
from src.data_ingestion.embeddings import create_embedding_client
from src.db.auth import AuthorizationService, Permission, SQLiteUserRepository
from src.grading import DocumentGradingService, GradingSettings
from src.grading.service import RUBRIC_VERSION
from src.clients.jev_client import JevSettings
from src.clients.mem0_client import create_mem0_client
from src.query import MemoryFirstQueryService, QueryCacheSettings

from .config import RetrievalSettings
from .service import HybridRetrievalService, validate_revisions


def main():
    parser = argparse.ArgumentParser(description="Retrieve permitted document chunks with semantic + BM25 fusion")
    parser.add_argument("query")
    parser.add_argument("--user", required=True, help="Acting demo username from SQLite")
    parser.add_argument("--grade", action="store_true", help="Rank retrieved chunks with Jev and include a grading report")
    parser.add_argument("--memory-cache", action="store_true",
                        help="Look up reusable results in Mem0 before retrieval; requires --grade")
    parser.add_argument("--revision", action="append", required=True, metavar="DOCUMENT_ID:REVISION_ID",
                        help="Approved document/revision pair; repeat for additional documents")
    args = parser.parse_args()
    if args.memory_cache and not args.grade:
        parser.error("--memory-cache requires --grade")
    load_dotenv(override=False)
    knowledge, settings = IngestionSettings.from_env(), RetrievalSettings.from_env()
    grading_settings = GradingSettings.from_env() if args.grade else None
    repository = SQLiteUserRepository()
    authorization = AuthorizationService(repository)
    user = repository.get_by_username(args.user)
    if user is None:
        parser.error("Unknown user")
    try:
        authorization.require_access(user.id, Permission.READ_KNOWLEDGE, namespace=knowledge.namespace)
        revisions = {}
        for pair in args.revision:
            doc_id, revision = pair.split(":")
            if doc_id in revisions and revisions[doc_id] != revision:
                raise ValueError("Only one approved revision per document may be selected")
            revisions[doc_id] = revision
        revisions = validate_revisions(revisions)
        if not args.query.strip() or len(args.query) > settings.max_query_characters:
            raise ValueError("Query is empty or exceeds the configured character limit")
    except (ValueError, PermissionError) as exc:
        parser.error(str(exc))

    def run_pipeline():
        # All retrieval clients are created lazily, after the memory lookup.
        with create_pinecone_client() as client:
            with get_pinecone_index(client, knowledge) as index:
                mode = get_index_embedding_mode(client, knowledge)
                embeddings = create_embedding_client(knowledge) if mode == "external" else None
                service = HybridRetrievalService(index=index, embedding_client=embeddings,
                                                 authorization=authorization, knowledge=knowledge, settings=settings,
                                                 embedding_mode=mode)
                result = service.retrieve(args.query, actor_id=user.id, approved_revisions=revisions)
                output = result.to_dict()
                if args.grade:
                    with create_jev_client() as jev:
                        grader = DocumentGradingService(client=jev, authorization=authorization,
                                                         namespace=knowledge.namespace, settings=grading_settings)
                        output["grading"] = grader.grade(args.query, result.documents, actor_id=user.id,
                                                         approved_revisions=revisions).to_dict()
                return output

    if args.memory_cache:
        service = MemoryFirstQueryService(
            memory=create_mem0_client(), authorization=authorization, namespace=knowledge.namespace,
            settings=QueryCacheSettings.from_env(), pipeline_config={
                "knowledge": asdict(knowledge), "retrieval": asdict(settings),
                "grading": asdict(grading_settings), "jev": asdict(JevSettings.from_env()),
                "rubric_version": RUBRIC_VERSION,
            },
        )
        output = service.run(args.query, actor_id=user.id, approved_revisions=revisions, pipeline=run_pipeline)
    else:
        output = run_pipeline()
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
