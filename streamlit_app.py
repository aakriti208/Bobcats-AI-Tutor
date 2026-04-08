"""
Bobcat AI Tutor — Streamlit App

Combines SimpleRAG's Canvas ingestion pipeline with a Boko-Buddy-style
chat interface. Runs fully locally: LlamaIndex + ChromaDB + Ollama.

Run:
    streamlit run streamlit_app.py
"""

import sys
from pathlib import Path
from datetime import datetime

import chromadb as chromadb_client
import streamlit as st

# Add project root to path so src.* imports work
sys.path.insert(0, str(Path(__file__).parent))

from llama_index.core import Settings as LlamaSettings, StorageContext, load_index_from_storage
from llama_index.core.chat_engine import SimpleChatEngine
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore

from src.config import (
    CHROMA_DB_DIR,
    STORAGE_DIR,
    EMBEDDING_MODEL,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
)

CHROMA_COLLECTION_NAME = "course_content"

# ---------------------------------------------------------------------------
# Page config (must be the first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Bobcat AI Tutor",
    page_icon="🐱",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🐱 Bobcat AI Tutor")
    st.caption("Powered by Canvas + Ollama — fully private")
    st.markdown("---")

    st.header("Tutor Settings")
    mode = st.radio(
        "Tutoring Style",
        [
            "Supportive (All Course Materials)",
            "Strict (Course Materials Only)",
            "General AI (No RAG)",
        ],
        index=0,
    )

    temp = st.slider("Creativity / Temperature", 0.0, 1.0, 0.1, step=0.05)
    deep_search = st.toggle("Deep Search / Summary Mode", value=False)

    st.markdown("---")

    if st.button("Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.header("Study Tools")
    if st.session_state.messages:
        chat_export = "# Course Study Guide\n\n"
        for msg in st.session_state.messages:
            role = "Student" if msg["role"] == "user" else "Tutor"
            chat_export += f"### {role}\n{msg['content']}\n\n---\n\n"
        st.download_button(
            label="Download Study Guide (.md)",
            data=chat_export,
            file_name="study_guide.md",
            mime="text/markdown",
        )
    else:
        st.caption("Ask a question to start building your study guide.")

# ---------------------------------------------------------------------------
# LlamaIndex global settings (embeddings + LLM)
# These are set once; @st.cache_resource ensures one instance per session.
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading embedding model...")
def _init_llama_settings(embedding_model: str):
    LlamaSettings.embed_model = HuggingFaceEmbedding(model_name=embedding_model)
    LlamaSettings.chunk_size = 8192
    LlamaSettings.chunk_overlap = 100


_init_llama_settings(EMBEDDING_MODEL)


# ---------------------------------------------------------------------------
# Index loader — cached so ChromaDB is only opened once per session
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading course knowledge base...")
def _load_index():
    index_store = STORAGE_DIR / "index_store.json"
    if not index_store.exists():
        return None

    collection = chromadb_client.PersistentClient(
        path=str(CHROMA_DB_DIR)
    ).get_or_create_collection(
        CHROMA_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store,
        persist_dir=str(STORAGE_DIR),
    )
    return load_index_from_storage(storage_context)


index = _load_index()

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------
_TODAY = datetime.now().strftime("%A, %B %d, %Y")

_PROMPT_SUPPORTIVE = (
    f"You are an expert Teaching Assistant for this course at Texas State University. "
    f"Today is {_TODAY}. "
    "Your tone is professional, encouraging, and technically precise. "
    "1. Use all available course materials (pages, assignments, discussions, announcements) to answer. "
    "2. ALWAYS cite your sources at the end of your explanation using the title of the material. "
    "3. If code or examples are relevant, include them. "
    "4. If the answer is not in the course materials, say so clearly and offer to help the student "
    "formulate a question for the professor's office hours."
)

_PROMPT_STRICT = (
    f"You are an expert Teaching Assistant for this course at Texas State University. "
    f"Today is {_TODAY}. "
    "Answer using ONLY the provided course materials. "
    "Do not use any external or general knowledge. "
    "If the answer is not in the materials, say: "
    "'I could not find this in the course materials. Please check with your professor.'"
)


def _build_engine(mode: str, temperature: float, deep_search: bool):
    """Build a fresh chat engine for the given settings."""
    llm = Ollama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=temperature,
        request_timeout=180.0,
    )

    if mode == "General AI (No RAG)":
        return SimpleChatEngine.from_defaults(
            llm=llm,
            system_prompt=(
                f"You are a helpful AI assistant. Today is {_TODAY}. "
                "Answer using your general knowledge."
            ),
        )

    if index is None:
        return None

    system_text = _PROMPT_SUPPORTIVE if mode.startswith("Supportive") else _PROMPT_STRICT
    k = 15 if deep_search else 5

    return index.as_chat_engine(
        chat_mode="context",
        llm=llm,
        system_prompt=system_text,
        similarity_top_k=k,
        streaming=False,
    )


# ---------------------------------------------------------------------------
# Main page title
# ---------------------------------------------------------------------------
st.title("Bobcat AI Tutor")
st.caption("Ask anything about your Canvas course materials.")

# Warn if index is missing
if index is None:
    st.warning(
        "No course knowledge base found. "
        "Run `python scripts/ingest_data.py --full` first, then refresh this page.",
        icon="⚠️",
    )

# ---------------------------------------------------------------------------
# Chat history display
# ---------------------------------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------------
if prompt := st.chat_input("Ask about your course..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    today_str = datetime.now().strftime("%Y-%m-%d")
    enhanced_prompt = f"(Today is {today_str}) {prompt}"

    col_rag, col_general = st.columns(2)

    # --- LEFT: RAG answer (Canvas materials) ---
    with col_rag:
        st.subheader("Bobcat AI Tutor")
        st.caption("Grounded in your Canvas course materials")
        with st.chat_message("assistant", avatar="🐱"):
            if index is None:
                st.error("Knowledge base not loaded. Please run ingestion first.")
                rag_text = ""
            else:
                with st.spinner("Searching course materials..."):
                    rag_engine = _build_engine(mode, temp, deep_search)
                    rag_response = rag_engine.chat(enhanced_prompt)
                    rag_text = rag_response.response
                    st.markdown(rag_text)

                # Citations from retrieved nodes
                if hasattr(rag_response, "source_nodes") and rag_response.source_nodes:
                    with st.expander("Sources from Canvas"):
                        seen = set()
                        for node in rag_response.source_nodes[:5]:
                            meta = node.metadata
                            title = meta.get("title", "Untitled")
                            content_type = meta.get("content_type", "")
                            url = meta.get("url", "")
                            course = meta.get("course_name", "")

                            key = f"{title}_{content_type}"
                            if key in seen:
                                continue
                            seen.add(key)

                            label = f"**{title}**"
                            if content_type:
                                label += f"  •  {content_type.capitalize()}"
                            if course:
                                label += f"  •  {course}"
                            st.markdown(label)
                            if url:
                                st.markdown(f"[Open in Canvas]({url})")
                            st.markdown("---")

    # --- RIGHT: General LLM (no RAG) ---
    with col_general:
        st.subheader("General AI")
        st.caption("General knowledge only — no course materials")
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Thinking..."):
                general_engine = _build_engine("General AI (No RAG)", temp, False)
                gen_response = general_engine.chat(enhanced_prompt)
                gen_text = gen_response.response
                st.markdown(gen_text)

    # Save RAG answer to history
    if rag_text:
        st.session_state.messages.append({"role": "assistant", "content": rag_text})
