# 🤖 Context-Aware RAG Chatbot

---

## 1. Introduction

Large Language Models (LLMs) are powerful but have limitations such as hallucinations, lack of awareness of external or private data, and weak handling of conversational context. This project addresses these challenges by implementing a **Context-Aware Retrieval-Augmented Generation (RAG) Chatbot** that grounds responses in a trusted knowledge source while maintaining conversational memory.

The system is built using **LangChain v1.0+**, **OpenAI**, **FAISS**, and **Streamlit**, following modern best practices and avoiding deprecated APIs.

---

## 2. Problem Statement

Traditional LLM-based chatbots:

- Cannot reliably answer questions from specific external documents
- Often hallucinate responses
- Do not understand follow-up questions without additional context handling
- Expose poor UX and insecure API key handling

There is a need for a conversational system that retrieves answers from a verified knowledge base, maintains conversational context, and provides a secure, user-friendly interface.

---

## 3. Objective

The objectives of this project are to:

- Build a **context-aware chatbot** using Retrieval-Augmented Generation (RAG)
- Enable accurate question answering from an external knowledge source
- Maintain conversational history for multi-turn interactions
- Reduce hallucinations by grounding responses in retrieved documents
- Use **modern LangChain v1.0+ (LCEL)** APIs
- Provide a clean and secure frontend experience

---

## 4. System Architecture

### High-Level Components

1. **Frontend (Streamlit)**
   - API key input
   - Chat interface
   - Clear chat history control

2. **Document Loader**
   - WebBaseLoader (Wikipedia – Artificial Intelligence page)

3. **Text Processing**
   - RecursiveCharacterTextSplitter

4. **Vector Store**
   - FAISS for similarity search

5. **Embedding Model**
   - OpenAI Embeddings (`text-embedding-3-small`)

6. **LLM**
   - OpenAI Chat Model (`gpt-4o-mini`)

7. **RAG Pipeline (LCEL)**
   - Contextual question reformulation
   - Document retrieval
   - Grounded answer generation

---

## 5. Methodology / Workflow

1. **API Key Connection**
   - User enters OpenAI API key via frontend
   - Chat functionality enabled only after successful connection

2. **Data Ingestion**
   - Wikipedia page is loaded and parsed

3. **Chunking**
   - Text split into overlapping chunks for better retrieval

4. **Vectorization**
   - Text chunks converted into embeddings
   - Stored in FAISS vector database

5. **Query Processing**
   - User query reformulated into a standalone question if chat history exists

6. **Retrieval**
   - Relevant document chunks retrieved using semantic similarity

7. **Answer Generation**
   - LLM generates concise answers grounded in retrieved context

8. **Memory Handling**
   - Chat history stored using Streamlit session state

---

## 6. Key Features

- Context-aware multi-turn conversations
- Retrieval-Augmented Generation (RAG)
- Modern LangChain LCEL implementation
- Secure API key handling
- Clear chat history functionality
- Cached vector store for performance

---

## 7. Technology Stack

- **Programming Language:** Python 3.10+
- **Frontend:** Streamlit
- **LLM Framework:** LangChain v1.0+
- **Vector Database:** FAISS
- **LLM & Embeddings:** OpenAI

---

## 8. Installation & Setup

```bash
# Clone repository
git clone https://github.com/your-username/context-aware-rag-chatbot.git
cd context-aware-rag-chatbot

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Run application
streamlit run app.py
```

---

## 9. Results & Observations

- The chatbot accurately answers questions based on retrieved context
- Follow-up questions are handled effectively using conversational memory
- Retrieval grounding significantly reduces hallucinations
- Clean UI improves usability and security

---

## 10. Limitations

- Knowledge limited to the ingested dataset
- Requires an active OpenAI API key
- Single-source document ingestion in the current version

---

## 11. Future Enhancements

- Upload and query custom documents (PDF, DOCX, TXT)
- Multi-document support
- Streaming responses
- Agentic RAG (Planner–Retriever–Verifier)
- Conversation export functionality

---

## 12. Conclusion
This project demonstrates a production-ready implementation of a Context-Aware RAG Chatbot using modern LLM tooling. It provides a scalable foundation for building enterprise-grade conversational AI systems that are accurate, secure, and context-aware.




