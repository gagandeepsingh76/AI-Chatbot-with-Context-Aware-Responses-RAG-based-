
"""
Context-Aware RAG Chatbot (LangChain v1.0+ | Clean UI)

Tech Stack:
- Streamlit
- LangChain v1.0+ (LCEL)
- FAISS
- OpenAI (Chat + Embeddings)

Requirements:
pip install streamlit langchain langchain-openai langchain-community faiss-cpu beautifulsoup4 lxml

Python: 3.10+
"""

# =========================
# 1. IMPORTS
# =========================

import os
import re
import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx

os.environ.setdefault("USER_AGENT", "context-aware-rag-chatbot-demo/1.0")

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FakeEmbeddings

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


def is_quota_error(error: Exception) -> bool:
    """Detect OpenAI quota/rate-limit failures by message text."""
    message = str(error).lower()
    return any(token in message for token in ["insufficient_quota", "ratelimiterror", "error code: 429"])


def ensure_streamlit_runtime() -> None:
    """Stop execution early when script is started with python app.py."""
    if get_script_run_ctx() is None:
        raise SystemExit("Run this app with: streamlit run app.py")


ensure_streamlit_runtime()

# =========================
# 2. STREAMLIT CONFIG
# =========================

st.set_page_config(
    page_title="Context-Aware RAG Chatbot",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Context-Aware RAG Chatbot")
st.caption("Ask questions based on the Artificial Intelligence Wikipedia page")

# =========================
# 3. SESSION STATE
# =========================

if "api_connected" not in st.session_state:
    st.session_state.api_connected = False

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "local_mode" not in st.session_state:
    st.session_state.local_mode = True

# =========================
# 4. SIDEBAR — API KEY + CLEAR HISTORY
# =========================

with st.sidebar:
    st.header("🔑 OpenAI Configuration")

    with st.form("api_key_form", clear_on_submit=False):
        api_key = st.text_input(
            "Enter your OpenAI API Key (optional)",
            type="password",
            placeholder="sk-..."
        )
        submitted = st.form_submit_button("🔌 Connect")

    if submitted:
        if api_key.strip().startswith("sk-"):
            os.environ["OPENAI_API_KEY"] = api_key.strip()
            st.session_state.api_connected = True
            st.session_state.local_mode = False
            st.success("✅ API key connected successfully!")
        else:
            st.session_state.api_connected = False
            st.session_state.local_mode = True
            st.error("❌ Invalid API key")

    if st.button("Use Local Mode (No OpenAI)"):
        st.session_state.local_mode = True
        st.session_state.api_connected = False
        st.info("Local mode enabled. Responses will be generated from retrieved context only.")

    st.markdown("---")

    if st.session_state.api_connected:
        if st.button("🧹 Clear Chat History"):
            st.session_state.chat_history = []
            st.success("Chat history cleared!")

    # st.markdown("---")
    # st.info("This chatbot uses RAG with conversational memory.")
    # st.warning("⚠️ LangChain v1.0+ (chains deprecated)")

# =========================
# 5. APP MODE
# =========================

if st.session_state.local_mode:
    st.success("Demo mode is active. The chatbot is running without paid API usage.")

# =========================
# 6. LOAD & VECTORIZE DATASET
# =========================

@st.cache_resource
def load_vector_store():
    url = "https://en.wikipedia.org/wiki/Artificial_intelligence"

    with st.spinner("📚 Loading and processing dataset..."):
        loader = WebBaseLoader(url)
        try:
            documents = loader.load()
        except Exception as e:
            st.warning("Could not download the Wikipedia page. Using offline fallback context.")
            st.caption(f"Dataset fallback reason: {e}")
            documents = [
                Document(
                    page_content=(
                        "Artificial intelligence (AI) is a field of computer science focused on "
                        "building systems that can perform tasks requiring human intelligence, "
                        "such as perception, language understanding, reasoning, and decision-making."
                    )
                )
            ]

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = splitter.split_documents(documents)

        # In demo mode, skip OpenAI embeddings entirely to avoid quota/key issues.
        if st.session_state.local_mode:
            vector_store = FAISS.from_documents(chunks, FakeEmbeddings(size=1536))
        else:
            try:
                embeddings = OpenAIEmbeddings(
                    model="text-embedding-3-small"
                )
                vector_store = FAISS.from_documents(chunks, embeddings)
            except Exception:
                st.info("Switched to local embeddings to keep the app available.")
                vector_store = FAISS.from_documents(chunks, FakeEmbeddings(size=1536))

    return vector_store

vector_store = load_vector_store()

# =========================
# 7. RAG CHAIN (MODERN LCEL)
# =========================

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def build_rag_chain(vector_store):
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.3
    )

    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    contextualize_prompt = ChatPromptTemplate.from_messages([
        ("system", "Rephrase the user question into a standalone question."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Use the following context to answer the question. "
         "If unknown, say you don't know. Max 3 sentences.\n\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    def contextualized_question(x):
        if x["chat_history"]:
            return (contextualize_prompt | llm | StrOutputParser()).invoke(x)
        return x["input"]

    rag_chain = (
        RunnablePassthrough.assign(
            context=lambda x: format_docs(
                retriever.invoke(contextualized_question(x))
            )
        )
        | qa_prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


def extractive_context_answer(user_query, vector_store):
    """Generate a readable answer from retrieved context without LLM calls."""
    docs = vector_store.similarity_search(user_query, k=6)
    if not docs:
        return "I could not find enough relevant context to answer that question."

    stop_words = {
        "the", "a", "an", "is", "are", "to", "of", "and", "in", "on", "for", "with",
        "what", "how", "why", "when", "where", "who", "which", "between", "difference"
    }
    query_tokens = [t for t in re.findall(r"[a-zA-Z]+", user_query.lower()) if t not in stop_words]

    scored = []
    for doc in docs:
        text = " ".join(doc.page_content.split())
        sentences = re.split(r"(?<=[.!?])\s+", text)
        for s in sentences:
            if len(s) < 50:
                continue
            s_low = s.lower()
            score = sum(1 for tok in query_tokens if tok in s_low)
            if score > 0:
                scored.append((score, s))

    if not scored:
        fallback = " ".join(docs[0].page_content.split())[:450]
        return f"Best available context:\n\n{fallback}..."

    scored.sort(key=lambda x: x[0], reverse=True)
    top_sentences = []
    seen = set()
    for _, sentence in scored:
        key = sentence[:120]
        if key in seen:
            continue
        seen.add(key)
        top_sentences.append(sentence)
        if len(top_sentences) == 3:
            break

    return "\n\n".join(top_sentences)


rag_chain = None if st.session_state.local_mode else build_rag_chain(vector_store)

# =========================
# 8. DISPLAY CHAT HISTORY
# =========================

for msg in st.session_state.chat_history:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    else:
        with st.chat_message("assistant"):
            st.markdown(msg.content)

# =========================
# 9. USER INPUT
# =========================

user_input = st.chat_input(
    "Ask something about Artificial Intelligence...",
    disabled=False
)

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking..."):
            try:
                if rag_chain is None:
                    result = extractive_context_answer(user_input, vector_store)
                else:
                    result = rag_chain.invoke({
                        "input": user_input,
                        "chat_history": st.session_state.chat_history
                    })
                st.markdown(result)
            except Exception as e:
                if is_quota_error(e):
                    result = extractive_context_answer(user_input, vector_store)
                    st.info("Generated a local response for this question.")
                    st.markdown(result)
                else:
                    result = "I hit a temporary issue while answering. Please try again."
                    st.error("Temporary processing issue. Please retry.")

    st.session_state.chat_history.append(HumanMessage(content=user_input))
    st.session_state.chat_history.append(AIMessage(content=result))
