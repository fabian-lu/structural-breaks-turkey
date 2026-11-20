from typing import List, Optional
import numpy as np
import hashlib
import os
import pickle


class LocalEmbedder:
    """
    Local embedder using Sentence Transformers.
    Much better than TF-IDF, no API needed!
    """

    def __init__(self,
                 model_name: str = 'all-MiniLM-L6-v2',
                 cache_dir: Optional[str] = ".rag_cache"):
        """
        Initialize embedder with Sentence Transformers.

        Args:
            model_name: HuggingFace model name
                       'all-MiniLM-L6-v2' = 80MB, fast, good quality
                       'all-mpnet-base-v2' = 420MB, slower, best quality
            cache_dir: Directory for caching embeddings
        """
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.cache = {}

        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            self._load_cache()

        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            print(f"Loaded embedding model: {model_name} (dim={self.embedding_dim})")
        except ImportError:
            raise ImportError(
                "Sentence Transformers not installed. Install with:\n"
                "pip install sentence-transformers"
            )

    def embed_text(self, text: str) -> np.ndarray:
        """Create embedding for a single text."""
        # Check cache
        text_hash = hashlib.md5(text.encode()).hexdigest()
        if text_hash in self.cache:
            return self.cache[text_hash]

        # Create embedding
        embedding = self.model.encode(text, convert_to_numpy=True)

        # Normalize
        embedding = embedding / np.linalg.norm(embedding)

        # Cache
        self.cache[text_hash] = embedding
        self._save_cache()

        return embedding

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Embed multiple texts efficiently."""
        # Check cache for all texts
        uncached_texts = []
        uncached_indices = []
        results = [None] * len(texts)

        for i, text in enumerate(texts):
            text_hash = hashlib.md5(text.encode()).hexdigest()
            if text_hash in self.cache:
                results[i] = self.cache[text_hash]
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)

        # Batch encode uncached texts
        if uncached_texts:
            embeddings = self.model.encode(
                uncached_texts,
                convert_to_numpy=True,
                show_progress_bar=len(uncached_texts) > 10
            )

            # Normalize
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / norms

            # Store in results and cache
            for idx, embedding, text in zip(uncached_indices, embeddings, uncached_texts):
                results[idx] = embedding
                text_hash = hashlib.md5(text.encode()).hexdigest()
                self.cache[text_hash] = embedding

            self._save_cache()

        return np.array(results)

    def _save_cache(self):
        """Save embedding cache."""
        if self.cache_dir:
            cache_file = os.path.join(self.cache_dir, "embeddings.pkl")
            with open(cache_file, 'wb') as f:
                pickle.dump(self.cache, f)

    def _load_cache(self):
        """Load embedding cache."""
        if self.cache_dir:
            cache_file = os.path.join(self.cache_dir, "embeddings.pkl")
            if os.path.exists(cache_file):
                with open(cache_file, 'rb') as f:
                    self.cache = pickle.load(f)