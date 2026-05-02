"""
Context-Aware RAG Chatbot

Streamlit interface for a LangChain + FAISS + OpenAI RAG workflow.
Run locally with:
    streamlit run app.py
"""

import os
import re

import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx

os.environ.setdefault("USER_AGENT", "context-aware-rag-chatbot-demo/1.0")

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import FakeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


KNOWLEDGE_SOURCE_URL = "https://en.wikipedia.org/wiki/Artificial_intelligence"
OPENAI_CHAT_MODEL = "gpt-4o-mini"
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"


def ensure_streamlit_runtime() -> None:
    """Stop execution early when the script is started with python app.py."""
    if get_script_run_ctx() is None:
        raise SystemExit("Run this app with: streamlit run app.py")


def get_hosted_api_key() -> str:
    """Read an OpenAI key from Streamlit secrets or the environment."""
    env_key = os.getenv("OPENAI_API_KEY", "").strip()
    if env_key:
        return env_key

    try:
        return str(st.secrets.get("OPENAI_API_KEY", "")).strip()
    except Exception:
        return ""


def should_fallback_to_local(error: Exception) -> bool:
    """Detect API failures where the extractive demo mode is a better response."""
    message = str(error).lower()
    fallback_tokens = [
        "insufficient_quota",
        "ratelimiterror",
        "rate limit",
        "error code: 429",
        "authenticationerror",
        "invalid api key",
        "incorrect api key",
        "error code: 401",
    ]
    return any(token in message for token in fallback_tokens)


ensure_streamlit_runtime()

st.set_page_config(
    page_title="Context-Aware RAG Chatbot",
    layout="centered",
)

st.title("Context-Aware RAG Chatbot")
st.caption("Ask questions grounded in the Artificial Intelligence Wikipedia page.")

hosted_api_key = get_hosted_api_key()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "user_api_key" not in st.session_state:
    st.session_state.user_api_key = ""

if "force_local_mode" not in st.session_state:
    st.session_state.force_local_mode = False


def get_active_api_key() -> str:
    if st.session_state.force_local_mode:
        return ""
    return st.session_state.user_api_key or hosted_api_key


with st.sidebar:
    st.header("Settings")

    if hosted_api_key and not st.session_state.user_api_key and not st.session_state.force_local_mode:
        st.success("Hosted OpenAI key is configured.")
    elif st.session_state.force_local_mode:
        st.info("Demo mode is active.")
    else:
        st.info("Add an OpenAI key for full AI responses.")

    with st.form("api_key_form", clear_on_submit=False):
        entered_api_key = st.text_input(
            "OpenAI API key",
            type="password",
            placeholder="sk-...",
            help="Optional. The app also works in demo mode without a key.",
        )
        submitted = st.form_submit_button("Connect")

    if submitted:
        cleaned_key = entered_api_key.strip()
        if cleaned_key.startswith("sk-"):
            st.session_state.user_api_key = cleaned_key
            st.session_state.force_local_mode = False
            st.success("API key connected.")
        else:
            st.error("Please enter a valid OpenAI API key.")

    controls = st.columns(2)
    with controls[0]:
        if st.button("Demo mode", use_container_width=True):
            st.session_state.force_local_mode = True
            st.session_state.user_api_key = ""
            st.rerun()
    with controls[1]:
        if st.button(
            "Hosted key",
            disabled=not hosted_api_key,
            use_container_width=True,
        ):
            st.session_state.force_local_mode = False
            st.session_state.user_api_key = ""
            st.rerun()

    st.markdown("---")

    if st.button("Clear chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()


active_api_key = get_active_api_key()
using_openai = bool(active_api_key)

if using_openai:
    st.success("AI mode is active.")
else:
    st.success("Demo mode is active. The app is running without paid API usage.")


@st.cache_resource(show_spinner=False)
def load_documents():
    loader = WebBaseLoader(KNOWLEDGE_SOURCE_URL)
    try:
        return loader.load()
    except Exception:
        return [
            Document(
                page_content=(
                    "Artificial intelligence (AI) is a field of computer science focused on "
                    "building systems that can perform tasks requiring human intelligence, "
                    "such as perception, language understanding, reasoning, and decision-making."
                )
            )
        ]


@st.cache_resource(show_spinner=False)
def load_vector_store(use_openai_embeddings: bool, api_key: str):
    documents = load_documents()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = splitter.split_documents(documents)

    if use_openai_embeddings:
        try:
            embeddings = OpenAIEmbeddings(
                model=OPENAI_EMBEDDING_MODEL,
                openai_api_key=api_key,
            )
            return FAISS.from_documents(chunks, embeddings), True
        except Exception:
            pass

    return FAISS.from_documents(chunks, FakeEmbeddings(size=1536)), False


with st.spinner("Loading knowledge base..."):
    vector_store, openai_embeddings_ready = load_vector_store(using_openai, active_api_key)

if using_openai and not openai_embeddings_ready:
    st.info("OpenAI embeddings were unavailable, so retrieval is using demo embeddings.")


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def build_rag_chain(vector_store, api_key: str):
    llm = ChatOpenAI(
        model=OPENAI_CHAT_MODEL,
        temperature=0.3,
        openai_api_key=api_key,
    )

    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    contextualize_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Rephrase the user question into a standalone question."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Use the following context to answer the question. "
                "If the answer is unknown, say you do not know. Max 3 sentences.\n\n{context}",
            ),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    def contextualized_question(values):
        if values["chat_history"]:
            return (contextualize_prompt | llm | StrOutputParser()).invoke(values)
        return values["input"]

    return (
        RunnablePassthrough.assign(
            context=lambda values: format_docs(
                retriever.invoke(contextualized_question(values))
            )
        )
        | qa_prompt
        | llm
        | StrOutputParser()
    )


def extractive_context_answer(user_query, vector_store):
    """Generate a readable answer from retrieved context without LLM calls."""
    docs = vector_store.similarity_search(user_query, k=6)
    if not docs:
        return "I could not find enough relevant context to answer that question."

    stop_words = {
        "the",
        "a",
        "an",
        "is",
        "are",
        "to",
        "of",
        "and",
        "in",
        "on",
        "for",
        "with",
        "what",
        "how",
        "why",
        "when",
        "where",
        "who",
        "which",
        "between",
        "difference",
    }
    query_tokens = [
        token
        for token in re.findall(r"[a-zA-Z]+", user_query.lower())
        if token not in stop_words
    ]

    scored = []
    for doc in docs:
        text = " ".join(doc.page_content.split())
        sentences = re.split(r"(?<=[.!?])\s+", text)
        for sentence in sentences:
            if len(sentence) < 50:
                continue
            sentence_lower = sentence.lower()
            score = sum(1 for token in query_tokens if token in sentence_lower)
            if score > 0:
                scored.append((score, sentence))

    if not scored:
        fallback = " ".join(docs[0].page_content.split())[:450]
        return f"Best available context:\n\n{fallback}..."

    scored.sort(key=lambda item: item[0], reverse=True)
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


rag_chain = build_rag_chain(vector_store, active_api_key) if using_openai else None

for message in st.session_state.chat_history:
    role = "user" if isinstance(message, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(message.content)

user_input = st.chat_input("Ask something about Artificial Intelligence...")

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                if rag_chain is None:
                    result = extractive_context_answer(user_input, vector_store)
                else:
                    result = rag_chain.invoke(
                        {
                            "input": user_input,
                            "chat_history": st.session_state.chat_history,
                        }
                    )
                st.markdown(result)
            except Exception as error:
                if should_fallback_to_local(error):
                    result = extractive_context_answer(user_input, vector_store)
                    st.info("Generated a demo response for this question.")
                    st.markdown(result)
                else:
                    result = "I hit a temporary issue while answering. Please try again."
                    st.error(result)

    st.session_state.chat_history.append(HumanMessage(content=user_input))
    st.session_state.chat_history.append(AIMessage(content=result))
