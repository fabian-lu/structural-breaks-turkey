from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
from pathlib import Path

from .vector_store import Document, ChromaVectorStore
from .document_parser import DocumentParser
from .embedder import LocalEmbedder


class RAGSystem:
    """Production-ready RAG system."""

    def __init__(self,
                 embedding_model: str = 'all-MiniLM-L6-v2',
                 cache_dir: str = ".rag_cache"):
        """Initialize RAG system."""

        # Initialize embedder
        self.embedder = LocalEmbedder(
            model_name=embedding_model,
            cache_dir=cache_dir
        )

        # Initialize vector store with correct dimension
        self.vector_store = ChromaVectorStore(
            embedding_dimension=self.embedder.embedding_dim,
            persist_directory=cache_dir
        )

        self.parser = DocumentParser()

    def add_documents(self, paths: List[str]):
        """
        Add documents to the RAG system.

        Args:
            paths: List of file paths or directory paths
        """
        documents = []

        # Expand directories and collect files
        file_paths = self._collect_files(paths)

        print(f"Processing {len(file_paths)} files...")

        # Parse documents
        for file_path in file_paths:
            try:
                # Parse file
                parsed = self.parser.parse_file(file_path)

                # Create document ID
                doc_id = str(hash(file_path))

                # Create embedding
                embedding = self.embedder.embed_text(parsed['content'])

                # Create document
                doc = Document(
                    id=doc_id,
                    content=parsed['content'],
                    metadata=parsed['metadata'],
                    embedding=embedding
                )

                documents.append(doc)
                print(f"  ✓ {Path(file_path).name}")

            except Exception as e:
                print(f"  ✗ Error processing {file_path}: {e}")

        if documents:
            # Add to vector store
            self.vector_store.add_documents(documents)
            print(f"Added {len(documents)} documents to RAG system.")
            
            # Verify documents were actually added
            stats = self.get_stats()
            total_docs = stats['vector_store']['total_documents']
            print(f"Vector store now contains {total_docs} total documents.")
        else:
            print("Warning: No valid documents were processed for addition to RAG.")

    def get_context_for_break(self,
                              break_date: datetime,
                              description: str = "",
                              window_days: int = 30,
                              top_k: int = 3) -> str:
        """
        Get document context for a structural break.

        This is the main method called during break analysis.
        """
        # Define date range
        start_date = break_date - timedelta(days=window_days)
        end_date = break_date + timedelta(days=window_days)

        # Create query embedding from description
        if description:
            query_embedding = self.embedder.embed_text(description)

            # Use hybrid search (semantic + date)
            results = self.vector_store.search_hybrid(
                query_embedding=query_embedding,
                date_range=(start_date, end_date),
                top_k=top_k,
                date_weight=0.3  # 30% weight on date proximity
            )
        else:
            # No description, just use date range
            results = self.vector_store.search_by_date_range(
                start_date=start_date,
                end_date=end_date,
                limit=top_k
            )

        if not results:
            # Check if we have any documents at all
            total_docs = self.vector_store.get_stats()['total_documents']
            if total_docs == 0:
                return "No documents in RAG system. Please add documents first."
            else:
                return f"No relevant documents found for this time period. (Total documents in system: {total_docs})"

        # Format context for LLM
        context_parts = []
        for i, result in enumerate(results, 1):
            meta = result['metadata']

            # Truncate content to reasonable length
            content = result['content']
            if len(content) > 1000:
                content = content[:1000] + "..."

            context_parts.append(
                f"Document {i}: {meta.get('title', 'Untitled')}\n"
                f"Date: {meta.get('date', 'Unknown')}\n"
                f"Type: {meta.get('type', 'Unknown')}\n"
                f"Relevance: {result.get('combined_score', result.get('relevance_score', 0)):.2f}\n"
                f"Content: {content}\n"
            )

        return "\n---\n".join(context_parts)

    def search(self,
               query: str,
               break_date: Optional[datetime] = None,
               window_days: int = 30,
               top_k: int = 5) -> List[Dict[str, Any]]:
        """
        General search interface.

        Args:
            query: Search query
            break_date: Optional date to search around
            window_days: Days before/after break date
            top_k: Number of results

        Returns:
            List of relevant documents with metadata
        """
        if break_date:
            # Date-constrained search
            context = self.get_context_for_break(
                break_date=break_date,
                description=query,
                window_days=window_days,
                top_k=top_k
            )
            # Parse context back to list (simplified for now)
            return [{'content': context}]
        else:
            # Pure semantic search
            query_embedding = self.embedder.embed_text(query)
            return self.vector_store.search_semantic(query_embedding, top_k)

    def _collect_files(self, paths: List[str]) -> List[str]:
        """Collect all files from paths (expanding directories)."""
        file_paths = []

        for path in paths:
            path_obj = Path(path)

            if path_obj.is_file():
                file_paths.append(str(path_obj))
            elif path_obj.is_dir():
                # Collect all files in directory
                for ext in ['*.txt', '*.csv', '*.pdf', '*.md']:
                    file_paths.extend([str(f) for f in path_obj.glob(ext)])
            else:
                print(f"Warning: Path not found: {path}")

        return file_paths

    def clear(self):
        """Clear all documents from the system."""
        self.vector_store.delete_collection()
        print("RAG system cleared.")
    
    def delete_documents(self, document_ids: List[str]) -> bool:
        """Delete specific documents by their IDs."""
        return self.vector_store.delete_documents(document_ids)
    
    def delete_documents_by_metadata(self, metadata_filter: Dict) -> bool:
        """Delete documents matching metadata criteria."""
        return self.vector_store.delete_documents_by_metadata(metadata_filter)
    
    def delete_documents_by_date_range(self, start_date: datetime, end_date: datetime) -> bool:
        """Delete documents within a date range."""
        return self.vector_store.delete_documents_by_date_range(start_date, end_date)
    
    def clear_embedding_cache(self):
        """Clear the embedding cache (but keep documents)."""
        import os
        if hasattr(self.embedder, 'cache'):
            self.embedder.cache.clear()
            if hasattr(self.embedder, 'cache_dir') and self.embedder.cache_dir:
                cache_file = os.path.join(self.embedder.cache_dir, "embeddings.pkl")
                if os.path.exists(cache_file):
                    os.remove(cache_file)
                    print("Embedding cache cleared.")
    
    def clear_all_cache(self):
        """Clear both documents and embedding cache."""
        import shutil
        from pathlib import Path
        
        # Clear documents
        self.clear()
        
        # Clear embedding cache
        self.clear_embedding_cache()
        
        # Remove entire cache directory if it exists
        if hasattr(self.embedder, 'cache_dir') and self.embedder.cache_dir:
            cache_path = Path(self.embedder.cache_dir)
            if cache_path.exists():
                shutil.rmtree(cache_path)
                print(f"Removed cache directory: {cache_path}")
        
        print("All RAG cache cleared.")

    def get_stats(self) -> Dict:
        """Get statistics about the RAG system."""
        return {
            'vector_store': self.vector_store.get_stats(),
            'embedder': {
                'model': self.embedder.model_name,
                'dimension': self.embedder.embedding_dim,
                'cache_size': len(self.embedder.cache)
            }
        }