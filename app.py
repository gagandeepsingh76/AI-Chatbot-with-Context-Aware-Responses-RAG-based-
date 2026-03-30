
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
import streamlit as st

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

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

# =========================
# 4. SIDEBAR — API KEY + CLEAR HISTORY
# =========================

with st.sidebar:
    st.header("🔑 OpenAI Configuration")

    with st.form("api_key_form", clear_on_submit=False):
        api_key = st.text_input(
            "Enter your OpenAI API Key",
            type="password",
            placeholder="sk-..."
        )
        submitted = st.form_submit_button("🔌 Connect")

    if submitted:
        if api_key.strip().startswith("sk-"):
            os.environ["OPENAI_API_KEY"] = api_key.strip()
            st.session_state.api_connected = True
            st.success("✅ API key connected successfully!")
        else:
            st.session_state.api_connected = False
            st.error("❌ Invalid API key")

    st.markdown("---")

    if st.session_state.api_connected:
        if st.button("🧹 Clear Chat History"):
            st.session_state.chat_history = []
            st.success("Chat history cleared!")

    # st.markdown("---")
    # st.info("This chatbot uses RAG with conversational memory.")
    # st.warning("⚠️ LangChain v1.0+ (chains deprecated)")

# =========================
# 5. BLOCK APP UNTIL CONNECTED
# =========================

if not st.session_state.api_connected:
    st.warning("🔑 Please enter your OpenAI API key to start chatting.")
    st.stop()

# =========================
# 6. LOAD & VECTORIZE DATASET
# =========================

@st.cache_resource
def load_vector_store():
    url = "https://en.wikipedia.org/wiki/Artificial_intelligence"

    with st.spinner("📚 Loading and processing dataset..."):
        loader = WebBaseLoader(url)
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = splitter.split_documents(documents)

        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small"
        )

        vector_store = FAISS.from_documents(chunks, embeddings)

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

rag_chain = build_rag_chain(vector_store)

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
    disabled=not st.session_state.api_connected
)

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking..."):
            try:
                result = rag_chain.invoke({
                    "input": user_input,
                    "chat_history": st.session_state.chat_history
                })
                st.markdown(result)
            except Exception as e:
                result = "❌ Error processing your request."
                st.error(e)

    st.session_state.chat_history.append(HumanMessage(content=user_input))
    st.session_state.chat_history.append(AIMessage(content=result))
