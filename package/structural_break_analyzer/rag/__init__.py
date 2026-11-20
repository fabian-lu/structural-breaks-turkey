"""RAG components for contextual break explanations.

This subpackage provides:
- `RAGSystem`: high-level interface to ingest documents and retrieve context.
- `LocalEmbedder`: sentence-transformers based text embedder.
- `ChromaVectorStore`: persistent vector store backed by ChromaDB.
- `DocumentParser`: parsing for text/csv/pdf/markdown.
"""

from .rag_system import RAGSystem
from .embedder import LocalEmbedder
from .vector_store import ChromaVectorStore, Document
from .document_parser import DocumentParser

__all__ = [
    "RAGSystem",
    "LocalEmbedder",
    "ChromaVectorStore",
    "Document",
    "DocumentParser",
]

