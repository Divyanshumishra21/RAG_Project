# RAG Chatbot with Streaming Responses 

## 🎯 Project Overview

This project implements a complete Retrieval-Augmented Generation (RAG) chatbot with streaming responses, designed to answer queries based on Terms & Conditions and Privacy Policy documents. The system combines semantic search with language generation to provide accurate, contextual responses.

## 🏗️ Architecture

```
User Query → Vector Search → Context Retrieval → LLM Generation → Streaming Response
```

### Components:
- **Document Processor**: Cleans and chunks documents into 100-300 word segments
- **Vector Store**: Uses FAISS for efficient similarity search with sentence-transformers embeddings
- **LLM Generator**: Employs open-source models (GPT-2/DialoGPT) for response generation
- **RAG Pipeline**: Orchestrates retrieval and generation with streaming support
- **Streamlit Interface**: Provides real-time chat with source attribution

## 📁 Project Structure

```
├── data/                # Document files
├── chunks/              # Processed text segments
├── vectordb/            # FAISS vector database
├── notebooks/           # Preprocessing scripts
├── src/                 # Core components
│   ├── document_processor.py
│   ├── vector_store.py
│   ├── llm_generator.py
│   └── rag_pipeline.py
├── app.py              # Streamlit application
├── requirements.txt    # Dependencies
└── README.md           # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Preprocessing

```bash
python notebooks/preprocessing.py
```

This will:
- Process documents in the `data/` folder
- Create text chunks (100-300 words)
- Generate embeddings using `all-MiniLM-L6-v2`
- Build and save FAISS vector index

### 3. Run the Chatbot

```bash
streamlit run app.py
```

## 🔧 Technical Details

### Document Processing
- **Chunking Strategy**: Recursive character splitting with sentence awareness
- **Chunk Size**: 200 words with 50-word overlap
- **Text Cleaning**: Removes extra whitespace and formatting artifacts

### Embedding & Retrieval
- **Model**: `all-MiniLM-L6-v2` (384-dimensional embeddings)
- **Vector DB**: FAISS with cosine similarity (IndexFlatIP)
- **Search**: Top-k retrieval with configurable k (default: 3)

### Language Generation
- **Primary Model**: Microsoft DialoGPT-medium
- **Fallback**: GPT-2 for compatibility
- **Optimization**: 16-bit precision, CPU/GPU adaptive
- **Streaming**: Token-by-token response generation

### Prompt Engineering
```
Based on the following context, please answer the question accurately and concisely.

Context:
[Retrieved chunks...]

Question: [User query]

Answer:
```

## 💡 Sample Queries

1. **"What are the payment terms?"**
   - Retrieves payment-related sections
   - Explains pricing, processing, and refund policies

2. **"How can I terminate my account?"**
   - Finds account termination procedures
   - Details user and company termination rights

3. **"What data do you collect from users?"**
   - Locates privacy policy sections
   - Lists collected information types

4. **"How do you protect my personal information?"**
   - Retrieves security measures
   - Explains data protection practices

5. **"What happens if I violate the terms?"**
   - Finds violation consequences
   - Details enforcement procedures

## 🎮 Features

### Streaming Interface
- Real-time token-by-token response generation
- Visual streaming indicator with cursor
- Responsive chat interface

### Source Attribution
- Expandable source chunks for each response
- Chunk relevance scoring
- Transparent retrieval process

### System Information
- Current model display
- Document chunk count
- Embedding model details
- Configurable retrieval parameters

### User Experience
- Sample query buttons
- Chat history persistence
- Clear chat functionality
- Responsive design

## 📊 Performance Characteristics

### Accuracy
- **Grounded Responses**: All answers based on retrieved context
- **Source Transparency**: Users can verify information sources
- **Hallucination Mitigation**: Responses limited to document content

### Limitations
- **Model Size**: Uses smaller models for accessibility
- **Context Window**: Limited by model's maximum input length
- **Response Quality**: Dependent on chunk relevance and model capabilities

### Success Cases
- ✅ Direct policy questions (payment, termination, data collection)
- ✅ Specific procedure inquiries
- ✅ Legal term explanations

### Failure Cases
- ❌ Questions outside document scope
- ❌ Complex multi-step reasoning
- ❌ Highly specific edge cases not covered in documents

## 🔄 Streaming Implementation

The streaming response system:
1. Generates response tokens iteratively
2. Updates UI with partial responses
3. Shows typing indicator during generation
4. Maintains chat context throughout

```python
def stream_response(response_iterator):
    response_placeholder = st.empty()
    full_response = ""
    
    for chunk in response_iterator:
        full_response += chunk
        response_placeholder.markdown(full_response + "▌")
        time.sleep(0.05)
    
    return full_response
```

## 🛠️ Development Notes

### Model Selection Rationale
- **DialoGPT**: Optimized for conversational responses
- **GPT-2 Fallback**: Ensures broad compatibility
- **Quantization**: Reduces memory usage for deployment

### Vector Store Choice
- **FAISS**: High-performance similarity search
- **Cosine Similarity**: Effective for semantic matching
- **Persistence**: Saves preprocessing time

### Chunking Strategy
- **Sentence-Aware**: Maintains semantic coherence
- **Overlap**: Ensures context continuity
- **Size Optimization**: Balances detail and relevance

## 📈 Future Enhancements

1. **Model Upgrades**: Integration with Mistral-7B or Llama-3
2. **Advanced Retrieval**: Hybrid search with keyword matching
3. **Response Quality**: Fine-tuning on domain-specific data
4. **Scalability**: Multi-document support and categorization
5. **Analytics**: Query logging and response quality metrics

## 🎯 Assignment Compliance

This implementation fulfills all requirements:
- ✅ RAG pipeline with vector database
- ✅ Open-source LLM integration
- ✅ Streaming response interface
- ✅ Document preprocessing and chunking
- ✅ Streamlit deployment with source attribution
- ✅ Complete project structure and documentation

## 📞 Contact

For questions about this implementation, please refer to the code comments and documentation provided throughout the project files.


<p align="center">
  <img src="Screenshot 2025-08-01 092449.png" width="1200"/>
</p>
