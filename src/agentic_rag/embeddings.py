import numpy as np
from sentence_transformers import SentenceTransformer
from typing import Callable, List, Optional


class EmbeddingManager:
    """Handles document embedding generation using SentenceTransformer"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding manager
        
        Args:
            model_name: HuggingFace model name for sentence embeddings
        """
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load the SentenceTransformer model"""
        try:
            print(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            print(f"Model loaded successfully. Embedding dimension: {self.model.get_sentence_embedding_dimension()}")
        except Exception as e:
            print(f"Error loading model {self.model_name}: {e}")
            raise

    def generate_embeddings(
        self,
        texts: List[str],
        progress_callback: Optional[Callable[[int, int], None]] = None,
        batch_size: int = 64,
    ) -> np.ndarray:
        """
        Generate embeddings for a list of texts

        Args:
            texts: List of text strings to embed
            progress_callback: Optional, called as progress_callback(done, total) after
                               each batch (e.g. to show a progress bar in the UI)
            batch_size: Texts per batch when reporting progress

        Returns:
            numpy array of embeddings with shape (len(texts), embedding_dim)
        """
        if not self.model:
            raise ValueError("Model not loaded")

        print(f"Generating embeddings for {len(texts)} texts...")
        if progress_callback is None:
            embeddings = self.model.encode(texts, show_progress_bar=True)
        else:
            batches = []
            for start in range(0, len(texts), batch_size):
                batches.append(self.model.encode(texts[start:start + batch_size]))
                progress_callback(min(start + batch_size, len(texts)), len(texts))
            embeddings = np.vstack(batches) if batches else np.empty((0, self.dimension))
        print(f"Generated embeddings with shape: {embeddings.shape}")
        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate the embedding for a single search query

        Args:
            query: The question or search text

        Returns:
            numpy array of shape (embedding_dim,)
        """
        if not self.model:
            raise ValueError("Model not loaded")
        return self.model.encode(query)

    @property
    def dimension(self) -> int:
        """Size of each embedding vector (the vector store needs it)"""
        return self.model.get_sentence_embedding_dimension()


if __name__ == "__main__":
    embedding_manager = EmbeddingManager()
    print(embedding_manager.generate_embeddings(["hello world"]).shape)