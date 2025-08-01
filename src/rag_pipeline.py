from typing import List, Tuple, Iterator
from .vector_store import VectorStore
from .llm_generator import LLMGenerator

class RAGPipeline:
    def __init__(self, vector_store: VectorStore, llm_generator: LLMGenerator):
        self.vector_store = vector_store
        self.llm_generator = llm_generator

    def retrieve(self, query: str, k: int = 3) -> List[Tuple[str, float]]:
        """
        Search for the top-k relevant chunks using the vector store.
        Returns a list of (chunk, similarity_score) tuples.
        """
        return self.vector_store.search(query, k)

    def generate_streaming_response(self, query: str, k: int = 3) -> Tuple[Iterator[str], List[str]]:
        """
        Generate a streaming response for the given query.
        Falls back to a simple, pre-built response format.
        """
        retrieved = self.retrieve(query, k)
        chunks = [chunk for chunk, _ in retrieved]
        basic_response = self.llm_generator.generate_simple_response(query, chunks)

        def stream():
            for word in basic_response.split():
                yield word + " "

        return stream(), chunks

    def generate_response(self, query: str, k: int = 3) -> Tuple[str, List[str]]:
        """
        Generate a full response without streaming.
        Returns the final text response along with source chunks.
        """
        retrieved = self.retrieve(query, k)
        chunks = [chunk for chunk, _ in retrieved]
        response = self.llm_generator.generate_simple_response(query, chunks)
        return response, chunks
