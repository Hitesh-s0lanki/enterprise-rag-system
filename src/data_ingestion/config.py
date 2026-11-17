"""Non-secret ingestion settings; credentials are read only by live clients."""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class IngestionSettings:
    index_name: str = "enterprise-it-knowledge"
    namespace: str = "it-support"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536
    chunk_size: int = 1000
    chunk_overlap: int = 150
    batch_size: int = 64
    cloud: str = "aws"
    region: str = "us-east-1"

    def __post_init__(self):
        if not 1 <= self.chunk_size <= 8000:
            raise ValueError("chunk_size must be between 1 and 8000 characters")
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("chunk_overlap must be non-negative and less than chunk_size")
        if not 1 <= self.batch_size <= 100:
            raise ValueError("batch_size must be between 1 and 100")
        if self.embedding_dimensions <= 0:
            raise ValueError("embedding_dimensions must be positive")
        if not all(s.strip() for s in (self.index_name, self.namespace, self.embedding_model)):
            raise ValueError("index_name, namespace and embedding_model cannot be empty")

    @classmethod
    def from_env(cls):
        return cls(
            index_name=os.getenv("PINECONE_INDEX_NAME", "enterprise-it-knowledge"),
            namespace=os.getenv("PINECONE_NAMESPACE", "it-support"),
            embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
            embedding_dimensions=int(os.getenv("OPENAI_EMBEDDING_DIMENSIONS", "1536")),
            chunk_size=int(os.getenv("INGESTION_CHUNK_SIZE", "1000")),
            chunk_overlap=int(os.getenv("INGESTION_CHUNK_OVERLAP", "150")),
            batch_size=int(os.getenv("INGESTION_BATCH_SIZE", "64")),
            cloud=os.getenv("PINECONE_CLOUD", "aws"),
            region=os.getenv("PINECONE_REGION", "us-east-1"),
        )
