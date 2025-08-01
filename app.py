import streamlit as st
import os
import sys
import time
from typing import Iterator

# Add 'src' to path for local imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.vector_store import VectorStore
from src.llm_generator import LLMGenerator
from src.rag_pipeline import RAGPipeline

# Streamlit configuration
st.set_page_config(
    page_title="RAG Chatbot - By Divyanshu",
    layout="wide"
)

@st.cache_resource
def load_rag_system():
    """
    Load and cache the RAG pipeline components
    including vector store, LLM, and pipeline object.
    """
    try:
        vector_store = VectorStore()
        vectordb_dir = os.path.join(os.path.dirname(__file__), "vectordb")

        if os.path.exists(vectordb_dir) and os.listdir(vectordb_dir):
            vector_store.load(vectordb_dir)
            st.success("Vector database loaded successfully.")
        else:
            st.error("Vector database not found. Please run preprocessing.")
            return None, None, None

        llm_generator = LLMGenerator()
        rag_pipeline = RAGPipeline(vector_store, llm_generator)

        return rag_pipeline, vector_store, llm_generator

    except Exception as e:
        st.error(f"Failed to load RAG system: {str(e)}")
        return None, None, None

def stream_response(response_iterator: Iterator[str]) -> str:
    """
    Simulate real-time streaming by displaying the output word by word.
    """
    response_container = st.empty()
    full_response = ""

    for chunk in response_iterator:
        full_response += chunk
        response_container.markdown(full_response + "▌")
        time.sleep(0.05)

    response_container.markdown(full_response)
    return full_response

def main():
    st.title("🤖 RAG Chatbot - By Divyanshu")
    st.markdown("Fine-tuned RAG Chatbot with Streaming Response Support")

    rag_pipeline, vector_store, llm_generator = load_rag_system()
    if rag_pipeline is None:
        st.stop()

    # Sidebar Info
    with st.sidebar:
        st.header("System Info")
        st.info(f"Model: {llm_generator.model_name}")
        st.info(f"Indexed Chunks: {len(vector_store.chunks)}")
        st.info("Embedding Model: all-MiniLM-L6-v2")

        st.header("Settings")
        k_chunks = st.slider("Number of Chunks to Retrieve", 1, 5, 3)

        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.rerun()

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and "sources" in message:
                with st.expander("Source Chunks"):
                    for i, source in enumerate(message["sources"]):
                        st.markdown(f"Chunk {i + 1}:")
                        st.markdown(source)
                        st.markdown("---")

    # Handle user input
    if prompt := st.chat_input("Ask a question based on the document..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            try:
                response_stream, source_chunks = rag_pipeline.generate_streaming_response(prompt, k_chunks)
                full_response = stream_response(response_stream)

                if source_chunks:
                    with st.expander("Source Chunks"):
                        for i, chunk in enumerate(source_chunks):
                            st.markdown(f"Chunk {i + 1}:")
                            st.markdown(chunk)
                            st.markdown("---")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_response,
                    "sources": source_chunks
                })

            except Exception as e:
                error_msg = f"Error generating response: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

    # Sample queries for testing
    st.markdown("---")
    st.subheader("Example Queries")

    sample_queries = [
        "What are the payment terms?",
        "How can I terminate my account?",
        "What data do you collect from users?",
        "How do you protect my personal information?",
        "What happens if I violate the terms?"
    ]

    cols = st.columns(len(sample_queries))
    for i, query in enumerate(sample_queries):
        if cols[i].button(query, key=f"sample_{i}"):
            st.session_state.sample_query = query
            st.rerun()

    # Handle selected sample query
    if hasattr(st.session_state, "sample_query"):
        query = st.session_state.sample_query
        delattr(st.session_state, "sample_query")

        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("assistant"):
            try:
                response_stream, source_chunks = rag_pipeline.generate_streaming_response(query, k_chunks)
                full_response = stream_response(response_stream)

                if source_chunks:
                    with st.expander("Source Chunks"):
                        for i, chunk in enumerate(source_chunks):
                            st.markdown(f"Chunk {i + 1}:")
                            st.markdown(chunk)
                            st.markdown("---")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_response,
                    "sources": source_chunks
                })

            except Exception as e:
                error_msg = f"Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

if __name__ == "__main__":
    main()
