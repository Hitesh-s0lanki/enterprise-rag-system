"""Weighted reciprocal rank fusion over chunk IDs, not incomparable raw scores."""

from langchain_core.documents import Document

from .config import RetrievalSettings


def fuse_rankings(
    dense: list[Document], keyword: list[Document], settings: RetrievalSettings,
) -> list[Document]:
    documents, scores, provenance = {}, {}, {}
    for signal, ranking, weight in (
        ("dense", dense, settings.dense_weight),
        ("keyword", keyword, 1 - settings.dense_weight),
    ):
        seen = set()
        for rank, document in enumerate(ranking, start=1):
            chunk_id = document.id
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            if chunk_id in documents and documents[chunk_id].page_content != document.page_content:
                raise ValueError("The same chunk ID returned conflicting text across search results")
            documents.setdefault(chunk_id, document)
            scores[chunk_id] = scores.get(chunk_id, 0) + weight / (settings.rrf_k + rank)
            provenance.setdefault(chunk_id, {})[signal] = {
                "rank": rank, "raw_score": document.metadata["retrieval_raw_score"],
            }
    results = []
    for chunk_id in sorted(scores, key=lambda key: (-scores[key], key))[:settings.top_k]:
        original = documents[chunk_id]
        metadata = {k: v for k, v in original.metadata.items() if k != "retrieval_raw_score"}
        metadata["retrieval"] = {"rrf_score": scores[chunk_id], "signals": provenance[chunk_id]}
        results.append(Document(id=chunk_id, page_content=original.page_content, metadata=metadata))
    return results
