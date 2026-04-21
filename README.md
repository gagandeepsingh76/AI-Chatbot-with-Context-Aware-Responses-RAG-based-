# 🤖 Context-Aware RAG Chatbot

A **context-aware AI chatbot** that uses **Retrieval-Augmented Generation (RAG)** to provide accurate, grounded responses from a trusted knowledge source.  

This system combines **LangChain, OpenAI, FAISS, and Streamlit** to enable **multi-turn conversations**, reduce hallucinations, and improve response reliability.

---

# 📌 Overview

Large Language Models (LLMs) are powerful but often suffer from:

- Hallucinated responses  
- Lack of access to external/private data  
- Poor handling of follow-up questions  

This project solves these issues by implementing a **RAG-based chatbot system** that:

- Retrieves relevant information from a knowledge base  
- Maintains conversational context  
- Generates accurate, grounded responses  

The system uses a **Wikipedia knowledge source** and provides a **clean, secure chat interface** using Streamlit.

---

# 🎯 Objectives

The primary goals of this project are:

- Build a **context-aware chatbot**
- Enable **accurate document-based question answering**
- Maintain **multi-turn conversation history**
- Reduce hallucinations using **retrieval grounding**
- Implement **modern LangChain v1.0+ (LCEL) pipeline**
- Provide a **secure and user-friendly interface**

---

# ⚙️ System Architecture

The system consists of the following main components:

- Streamlit frontend (chat UI + API key input)
- WebBaseLoader (data ingestion)
- RecursiveCharacterTextSplitter (text chunking)
- OpenAI Embeddings (`text-embedding-3-small`)
- FAISS vector database
- OpenAI LLM (`gpt-4o-mini`)
- LangChain LCEL RAG pipeline

The system retrieves relevant document chunks and generates responses grounded in context.

---

# 🧠 Working Principle

1. **API Key Authentication**
   - User enters OpenAI API key via UI.

2. **Data Ingestion**
   - Wikipedia content is loaded and processed.

3. **Text Chunking**
   - Data is split into overlapping chunks.

4. **Vector Embedding**
   - Text is converted into embeddings and stored in FAISS.

5. **Query Reformulation**
   - User query is converted into a standalone question (if needed).

6. **Context Retrieval**
   - Relevant document chunks are retrieved.

7. **Answer Generation**
   - LLM generates responses based on retrieved context.

8. **Conversation Memory**
   - Chat history is stored for multi-turn interactions.

---

# 🔌 System Components

| Component | Description |
|--------|--------|
| Streamlit | Frontend chat interface |
| LangChain | RAG pipeline orchestration |
| OpenAI API | LLM + embeddings |
| FAISS | Vector similarity search |
| WebBaseLoader | Loads Wikipedia data |
| Text Splitter | Splits data into chunks |

---

# 📊 Block Diagram

The following diagram illustrates the architecture of the RAG chatbot system.

<img width="800" height="436" alt="image" src="https://github.com/user-attachments/assets/bd3b4e13-9ce8-4488-a35a-6244a9631180" />


---

# 🔧 Pipeline Flow

The RAG pipeline consists of:

- Contextual query reformulation  
- Semantic document retrieval  
- Grounded answer generation  

This ensures responses are **accurate, relevant, and context-aware**.

---

# 🧮 Software Algorithm

The algorithm follows these steps:

1. API Key Validation  
2. Data Loading  
3. Text Chunking  
4. Embedding Generation  
5. Query Processing  
6. Document Retrieval  
7. Response Generation  
8. Memory Update  

---

# 📈 System Flowchart

This flowchart represents the complete workflow of the RAG chatbot.

<img width="1122" height="1402" alt="image" src="https://github.com/user-attachments/assets/3830405e-920c-4105-bdbd-a56f2cc731b4" />


---

# 💬 Chat Interface

The system provides:

- Interactive chat UI  
- API key input field  
- Clear chat history option  
- Real-time response generation  

---

# 📂 Project Structure

context-aware-rag-chatbot  
│  
├── app.py  
├── requirements.txt  
├── README.md  
│  
├── /data  
├── /vectorstore  
└── /utils  


---

# 🚀 Applications

- AI-powered knowledge assistants  
- Customer support chatbots  
- Document-based Q&A systems  
- Enterprise knowledge retrieval  
- Educational AI tools  

---

# 🔮 Future Improvements

Possible improvements for the system:

- Upload custom documents (PDF, DOCX, TXT)  
- Multi-document support  
- Streaming responses  
- Agentic RAG (Planner–Retriever–Verifier)  
- Chat export functionality  
- Deployment on cloud platforms  

---

# 👨‍💻 Author

**Gagandeep Singh**  

Computer Science Student  
Interested in Artificial Intelligence, LLMs, and System Design.

---

# ⭐ Support

If you find this project useful, please consider giving it a ⭐ on GitHub.
