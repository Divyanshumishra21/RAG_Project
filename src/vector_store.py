import os
import pickle
import faiss
import numpy as np
from typing import List, Tuple
from sentence_transformers import SentenceTransformer

class VectorStore:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.chunks = []
        self.embeddings = None

    def create_embeddings(self, chunks: List[str]) -> np.ndarray:
        """
        Generate sentence embeddings for all chunks.
        """
        return self.model.encode(chunks, show_progress_bar=True)

    def build_index(self, chunks: List[str]):
        """
        Create a FAISS index using cosine similarity.
        """
        self.chunks = chunks
        self.embeddings = self.create_embeddings(chunks)

        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)

        # Normalize for cosine similarity
        faiss.normalize_L2(self.embeddings)
        self.index.add(self.embeddings)

    def search(self, query: str, k: int = 3) -> List[Tuple[str, float]]:
        """
        Perform similarity search against the index.
        """
        if self.index is None:
            raise ValueError("Index not initialized. Call build_index first.")

        query_embedding = self.model.encode([query])
        faiss.normalize_L2(query_embedding)

        scores, indices = self.index.search(query_embedding, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.chunks):
                results.append((self.chunks[idx], float(score)))

        return results

    def save(self, vectordb_dir: str):
        """
        Save the index and metadata to disk.
        """
        os.makedirs(vectordb_dir, exist_ok=True)

        faiss.write_index(self.index, os.path.join(vectordb_dir, "index.faiss"))

        with open(os.path.join(vectordb_dir, "chunks.pkl"), "wb") as f:
            pickle.dump(self.chunks, f)

        with open(os.path.join(vectordb_dir, "embeddings.pkl"), "wb") as f:
            pickle.dump(self.embeddings, f)

    def load(self, vectordb_dir: str):
        """
        Load the index and metadata from disk.
        """
        self.index = faiss.read_index(os.path.join(vectordb_dir, "index.faiss"))

        with open(os.path.join(vectordb_dir, "chunks.pkl"), "rb") as f:
            self.chunks = pickle.load(f)

        with open(os.path.join(vectordb_dir, "embeddings.pkl"), "rb") as f:
            self.embeddings = pickle.load(f)
