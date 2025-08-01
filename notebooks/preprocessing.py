#!/usr/bin/env python3

"""
Script to preprocess documents and create a FAISS vector database.
"""

import os
import sys

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.document_processor import DocumentProcessor
from src.vector_store import VectorStore

def main():
    # Initialize processors
    processor = DocumentProcessor(chunk_size=200, chunk_overlap=50)
    vector_store = VectorStore()

    # Prepare paths
    project_root = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.join(project_root, "data")
    chunks_dir = os.path.join(project_root, "chunks")
    vectordb_dir = os.path.join(project_root, "vectordb")

    # Step 1: Process documents
    print("Extracting and chunking documents...")
    chunks = processor.process_documents(data_dir)
    print(f"Total chunks created: {len(chunks)}")

    # Save chunks to file (for reference)
    os.makedirs(chunks_dir, exist_ok=True)
    chunks_path = os.path.join(chunks_dir, "chunks.txt")
    with open(chunks_path, "w", encoding="utf-8") as f:
        for i, chunk in enumerate(chunks):
            f.write(f"Chunk {i + 1}:\n{chunk}\n\n{'=' * 50}\n\n")

    # Step 2: Create vector index
    print("Generating embeddings and building vector index...")
    vector_store.build_index(chunks)

    # Step 3: Save index to disk
    os.makedirs(vectordb_dir, exist_ok=True)
    vector_store.save(vectordb_dir)

    print(f"Vector store saved at: {vectordb_dir}")
    print("Preprocessing complete.")

if __name__ == "__main__":
    main()
