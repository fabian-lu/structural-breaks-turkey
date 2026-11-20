from dataclasses import dataclass
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime, timedelta
import uuid
import numpy as np


@dataclass
class Document:
    """Document in the store."""
    id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None


class ChromaVectorStore:
    """
    Production-ready vector store using ChromaDB.
    """

    def __init__(self,
                 collection_name: str = "structural_breaks",
                 persist_directory: str = ".chroma_db",
                 embedding_dimension: int = 384):
        """
        Initialize ChromaDB vector store.

        Args:
            collection_name: Name of the collection
            persist_directory: Where to store the database
            embedding_dimension: Dimension of embeddings (from your embedder)
        """
        try:
            # TODO: This is just a local issue with my sqlite version, remove this at some point
            # --- SQLite shim for ChromaDB (requires sqlite >= 3.35) ---
            import sys
            try:
                from sqlite3 import sqlite_version as _sqlite_version
                _need = (3, 35, 0)
                _have = tuple(map(int, _sqlite_version.split(".")))
                if _have < _need:
                    import pysqlite3 as sqlite3  # provides a bundled modern SQLite
                    sys.modules["sqlite3"] = sqlite3
            except Exception:
                # If anything goes wrong, fall back; Chroma will raise/skip as before
                pass
            # ----------------------------------------------------------
            import chromadb
            from chromadb.config import Settings
        except ImportError:
            raise ImportError(
                "ChromaDB not installed. Install with:\n"
                "pip install chromadb"
            )

        self.embedding_dimension = embedding_dimension

        # Create persistent client
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

        # Get or create collection
        try:
            self.collection = self.client.get_collection(collection_name)
            print(f"Loaded existing collection with {self.collection.count()} documents")
        except Exception as e:
            # If we get a readonly database or other error, try to reset and recreate
            if "readonly database" in str(e).lower() or "database is locked" in str(e).lower():
                print(f"Database appears to be locked/readonly, attempting to reset...")
                try:
                    # Try to reset the client
                    self.client.reset()
                    print("Database reset successful")
                except Exception:
                    # If reset fails, recreate the client entirely
                    print("Client reset failed, recreating client...")
                    import shutil
                    from pathlib import Path
                    cache_path = Path(persist_directory)
                    if cache_path.exists():
                        shutil.rmtree(cache_path)
                        print(f"Removed corrupted cache directory: {cache_path}")
                    # Recreate client
                    self.client = chromadb.PersistentClient(
                        path=persist_directory,
                        settings=Settings(
                            anonymized_telemetry=False,
                            allow_reset=True
                        )
                    )
            
            # Try to create collection
            try:
                self.collection = self.client.create_collection(
                    name=collection_name,
                    metadata={"hnsw:space": "cosine"}
                )
                print(f"Created new collection: {collection_name}")
            except Exception as create_error:
                print(f"Failed to create collection: {create_error}")
                raise

    def add_documents(self, documents: List[Document]):
        """Add documents with their embeddings to the store."""
        if not documents:
            return

        # Filter out documents without embeddings
        valid_docs = [d for d in documents if d.embedding is not None]
        if not valid_docs:
            print("Warning: No documents with embeddings to add")
            return

        # Prepare data for ChromaDB
        ids = []
        contents = []
        embeddings = []
        metadatas = []

        for doc in valid_docs:
            # Generate ID if not present
            doc_id = doc.id if doc.id else str(uuid.uuid4())
            ids.append(doc_id)
            contents.append(doc.content)
            embeddings.append(doc.embedding.tolist())  # Convert numpy to list

            # Clean metadata for ChromaDB (must be primitive types)
            metadata = {}
            for key, value in doc.metadata.items():
                if isinstance(value, datetime):
                    metadata[key] = value.isoformat()
                    metadata[f"{key}_timestamp"] = value.timestamp()  # For numeric comparisons
                elif isinstance(value, (str, int, float, bool)):
                    metadata[key] = value
                elif value is None:
                    metadata[key] = ""
                else:
                    metadata[key] = str(value)
            metadatas.append(metadata)

        # Add to collection
        self.collection.add(
            ids=ids,
            documents=contents,
            embeddings=embeddings,
            metadatas=metadatas
        )

        print(f"Added {len(valid_docs)} documents to vector store")

    def search_semantic(self,
                        query_embedding: np.ndarray,
                        top_k: int = 5,
                        metadata_filter: Optional[Dict] = None) -> List[Dict]:
        """
        Pure semantic search.

        Args:
            query_embedding: Query vector
            top_k: Number of results
            metadata_filter: Optional ChromaDB where clause

        Returns:
            List of documents sorted by relevance
        """
        if self.collection.count() == 0:
            return []

        # Ensure we don't request more than available
        n_results = min(top_k, self.collection.count())

        # Perform semantic search
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results,
            where=metadata_filter,
            include=["metadatas", "documents", "distances"]
        )

        # Format results
        formatted = []
        for i in range(len(results['ids'][0])):
            formatted.append({
                'id': results['ids'][0][i],
                'content': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i],
                'relevance_score': 1.0 - results['distances'][0][i]
            })

        return formatted

    def search_by_date_range(self,
                             start_date: datetime,
                             end_date: datetime,
                             limit: Optional[int] = None) -> List[Dict]:
        """
        Get all documents within a date range.

        Args:
            start_date: Start of range
            end_date: End of range
            limit: Maximum documents to return

        Returns:
            Documents in date range
        """
        if self.collection.count() == 0:
            return []

        # Use timestamp for numeric comparison (more reliable)
        where_clause = {
            "$and": [
                {"date_timestamp": {"$gte": start_date.timestamp()}},
                {"date_timestamp": {"$lte": end_date.timestamp()}}
            ]
        }

        # Get documents by metadata filter
        results = self.collection.get(
            where=where_clause,
            limit=limit,
            include=["metadatas", "documents", "embeddings"]
        )

        # Format results
        formatted = []
        for i in range(len(results['ids'])):
            doc_dict = {
                'id': results['ids'][i],
                'content': results['documents'][i],
                'metadata': results['metadatas'][i]
            }

            # Calculate date proximity if we have a date
            if 'date' in doc_dict['metadata']:
                try:
                    doc_date = datetime.fromisoformat(doc_dict['metadata']['date'])
                    doc_dict['date_object'] = doc_date
                except:
                    pass

            formatted.append(doc_dict)

        return formatted

    def search_hybrid(self,
                      query_embedding: np.ndarray,
                      date_range: Tuple[datetime, datetime],
                      top_k: int = 5,
                      date_weight: float = 0.3) -> List[Dict]:
        """
        Hybrid search combining semantic similarity and date relevance.

        Args:
            query_embedding: Query vector for semantic search
            date_range: (start_date, end_date) tuple
            top_k: Number of results
            date_weight: Weight for date relevance (0-1)

        Returns:
            Documents sorted by combined score
        """
        start_date, end_date = date_range

        # Build date filter
        where_clause = {
            "$and": [
                {"date_timestamp": {"$gte": start_date.timestamp()}},
                {"date_timestamp": {"$lte": end_date.timestamp()}}
            ]
        }

        # Get semantic results within date range
        semantic_results = self.search_semantic(
            query_embedding=query_embedding,
            top_k=top_k * 2,  # Get more to allow for date reranking
            metadata_filter=where_clause
        )

        if not semantic_results:
            # No results in date range, fall back to pure date search
            date_results = self.search_by_date_range(start_date, end_date, limit=top_k)
            for result in date_results:
                result['relevance_score'] = 0.5  # Default score
            return date_results[:top_k]

        # Calculate combined scores
        mid_date = start_date + (end_date - start_date) / 2

        for result in semantic_results:
            # Semantic score (already have this)
            semantic_score = result['relevance_score']

            # Date proximity score
            if 'date' in result['metadata']:
                try:
                    doc_date = datetime.fromisoformat(result['metadata']['date'])
                    days_from_mid = abs((doc_date - mid_date).days)
                    max_days = (end_date - start_date).days / 2
                    date_score = max(0, 1 - (days_from_mid / max_days))
                except:
                    date_score = 0.5
            else:
                date_score = 0.5

            # Combined score
            result['semantic_score'] = semantic_score
            result['date_score'] = date_score
            result['combined_score'] = (
                    (1 - date_weight) * semantic_score +
                    date_weight * date_score
            )

        # Sort by combined score
        semantic_results.sort(key=lambda x: x['combined_score'], reverse=True)

        return semantic_results[:top_k]

    def get_document_by_id(self, doc_id: str) -> Optional[Dict]:
        """Get a specific document by ID."""
        results = self.collection.get(
            ids=[doc_id],
            include=["metadatas", "documents", "embeddings"]
        )

        if results['ids']:
            return {
                'id': results['ids'][0],
                'content': results['documents'][0],
                'metadata': results['metadatas'][0]
            }
        return None

    def delete_documents(self, document_ids: List[str]) -> bool:
        """
        Delete specific documents by their IDs.
        
        Args:
            document_ids: List of document IDs to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Filter to only existing IDs
            existing_docs = self.collection.get(ids=document_ids, include=[])
            existing_ids = existing_docs['ids']
            
            if existing_ids:
                self.collection.delete(ids=existing_ids)
                print(f"Deleted {len(existing_ids)} documents")
                return True
            else:
                print("No matching documents found to delete")
                return False
        except Exception as e:
            print(f"Error deleting documents: {e}")
            return False
    
    def delete_documents_by_metadata(self, metadata_filter: Dict) -> bool:
        """
        Delete documents matching metadata criteria.
        
        Args:
            metadata_filter: ChromaDB where clause for filtering
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get documents matching filter
            results = self.collection.get(
                where=metadata_filter,
                include=[]
            )
            
            if results['ids']:
                self.collection.delete(ids=results['ids'])
                print(f"Deleted {len(results['ids'])} documents matching filter")
                return True
            else:
                print("No documents found matching the filter")
                return False
        except Exception as e:
            print(f"Error deleting documents by metadata: {e}")
            return False
    
    def delete_documents_by_date_range(self, start_date: datetime, end_date: datetime) -> bool:
        """
        Delete all documents within a date range.
        
        Args:
            start_date: Start of range
            end_date: End of range
            
        Returns:
            True if successful, False otherwise
        """
        where_clause = {
            "$and": [
                {"date_timestamp": {"$gte": start_date.timestamp()}},
                {"date_timestamp": {"$lte": end_date.timestamp()}}
            ]
        }
        return self.delete_documents_by_metadata(where_clause)

    def delete_collection(self):
        """Delete the entire collection."""
        self.client.delete_collection(self.collection.name)
        print(f"Deleted collection: {self.collection.name}")

    def get_stats(self) -> Dict:
        """Get statistics about the vector store."""
        return {
            'total_documents': self.collection.count(),
            'collection_name': self.collection.name,
            'embedding_dimension': self.embedding_dimension
        }