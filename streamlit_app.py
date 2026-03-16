import hashlib
import os
import pickle
from pathlib import Path

import faiss
import streamlit as st

from app.chunking import chunk_text
from app.embeddings import create_embeddings
from app.generator import REFUSAL_TEXT, generate_answer
from app.ingestion import extract_text_from_pdf
from app.memory import add_ai_message, add_user_message, clear_history
from app.query_rewriter import rewrite_query
from app.vector_store import (create_vector_store, get_confidence_label,
                              search, should_answer)

st.set_page_config(
    page_title="PaperPilot AI",
    page_icon="📘",
    layout="wide",
)

# -----------------------------
# Styling
# -----------------------------
CUSTOM_CSS = """
<style>
:root {
    --bg: #07111f;
    --panel: rgba(13, 22, 38, 0.88);
    --panel-2: rgba(17, 27, 46, 0.92);
    --border: rgba(125, 157, 203, 0.16);
    --text: #eaf2ff;
    --muted: #97a9c3;
    --accent: #5ee7ff;
    --accent-soft: rgba(94, 231, 255, 0.12);
    --success: rgba(34, 197, 94, 0.16);
    --warning: rgba(245, 158, 11, 0.16);
    --danger: rgba(239, 68, 68, 0.16);
}

.stApp {
    background:
        radial-gradient(circle at top left, rgba(94, 231, 255, 0.08), transparent 28%),
        radial-gradient(circle at top right, rgba(59, 130, 246, 0.08), transparent 24%),
        linear-gradient(180deg, #050d18 0%, #081220 45%, #091321 100%);
    color: var(--text);
}

.block-container {
    max-width: 1080px;
    padding-top: 1.7rem;
    padding-bottom: 2rem;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a1424 0%, #0d1728 100%);
    border-right: 1px solid rgba(94, 231, 255, 0.10);
}

.hero {
    background: linear-gradient(135deg, rgba(13, 22, 38, 0.96), rgba(16, 28, 48, 0.92));
    border: 1px solid rgba(94, 231, 255, 0.16);
    border-radius: 26px;
    padding: 1.35rem 1.4rem;
    margin-bottom: 1rem;
    box-shadow: 0 16px 34px rgba(2, 8, 23, 0.30);
}

.hero-title {
    font-size: 2rem;
    font-weight: 800;
    margin-bottom: 0.3rem;
    letter-spacing: 0.2px;
}

.hero-subtitle {
    color: var(--muted);
    font-size: 0.98rem;
    line-height: 1.5;
}

.pills {
    display: flex;
    gap: 0.55rem;
    flex-wrap: wrap;
    margin-top: 0.95rem;
}

.pill {
    background: var(--accent-soft);
    border: 1px solid rgba(94, 231, 255, 0.16);
    color: #c9f7ff;
    border-radius: 999px;
    padding: 0.34rem 0.7rem;
    font-size: 0.82rem;
    font-weight: 600;
}

.panel {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 0.9rem 1rem;
    margin-bottom: 0.75rem;
}

.metric-label {
    color: var(--muted);
    font-size: 0.78rem;
    margin-bottom: 0.15rem;
}

.metric-value {
    color: var(--text);
    font-size: 1rem;
    font-weight: 700;
}

.answer-card {
    background: linear-gradient(180deg, rgba(12, 20, 34, 0.95), rgba(15, 24, 41, 0.92));
    border: 1px solid rgba(148, 163, 184, 0.12);
    border-radius: 22px;
    padding: 1rem 1.05rem;
    margin-top: 0.42rem;
    line-height: 1.65;
}

.empty-card {
    background: rgba(120, 33, 33, 0.16);
    border: 1px solid rgba(248, 113, 113, 0.24);
    border-radius: 18px;
    padding: 1rem 1.05rem;
    margin-top: 0.42rem;
    color: #ffd2d2;
}

.chip {
    display: inline-block;
    margin-bottom: 0.55rem;
    padding: 0.28rem 0.66rem;
    border-radius: 999px;
    font-size: 0.79rem;
    font-weight: 700;
    border: 1px solid transparent;
}

.chip-high {
    background: rgba(34, 197, 94, 0.14);
    color: #bbf7d0;
    border-color: rgba(34, 197, 94, 0.22);
}

.chip-medium {
    background: rgba(245, 158, 11, 0.14);
    color: #fde68a;
    border-color: rgba(245, 158, 11, 0.22);
}

.chip-low {
    background: rgba(239, 68, 68, 0.14);
    color: #fecaca;
    border-color: rgba(239, 68, 68, 0.22);
}

.source-card {
    background: rgba(8, 14, 24, 0.82);
    border: 1px solid rgba(94, 231, 255, 0.10);
    border-radius: 16px;
    padding: 0.85rem 0.9rem;
    margin-bottom: 0.75rem;
}

.source-title {
    font-weight: 700;
    color: #f2f7ff;
    margin-bottom: 0.18rem;
}

.source-meta {
    color: var(--muted);
    font-size: 0.82rem;
    margin-bottom: 0.45rem;
}

.small-muted {
    color: var(--muted);
    font-size: 0.88rem;
}

.section-title {
    font-size: 0.95rem;
    color: #dceaff;
    font-weight: 700;
    margin-bottom: 0.35rem;
}

.chat-spacer {
    height: 0.3rem;
}

div[data-testid="stChatInput"] textarea {
    background: rgba(13, 22, 38, 0.95) !important;
    color: #f8fbff !important;
    border-radius: 16px !important;
}

div[data-testid="stExpander"] details {
    background: rgba(10, 17, 29, 0.68);
    border: 1px solid rgba(148, 163, 184, 0.10);
    border-radius: 14px;
    padding: 0.2rem 0.2rem;
}

hr {
    border-color: rgba(148, 163, 184, 0.10);
}

.welcome-card {
    background: linear-gradient(135deg, rgba(13, 22, 38, 0.96), rgba(16, 28, 48, 0.92));
    border: 1px solid rgba(94, 231, 255, 0.18);
    border-radius: 20px;
    padding: 1.4rem 1.5rem;
    margin-top: 0.6rem;
    margin-bottom: 0.6rem;
    font-size: 1.05rem;
}

.welcome-title {
    font-size: 1.25rem;
    font-weight: 700;
    margin-bottom: 0.35rem;
}

.welcome-text {
    color: #9fb3d1;
    font-size: 0.95rem;
}

</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# -----------------------------
# Cache
# -----------------------------
CACHE_DIR = Path(".rag_cache")
CACHE_DIR.mkdir(exist_ok=True)


def get_data_signature(data_folder="data"):
    records = []

    for file in sorted(os.listdir(data_folder)):
        if file.endswith(".pdf"):
            path = os.path.join(data_folder, file)
            stat = os.stat(path)
            records.append(f"{file}:{stat.st_mtime}:{stat.st_size}")

    raw = "|".join(records)
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


@st.cache_resource(show_spinner=False)
def load_rag_pipeline(data_folder="data"):
    signature = get_data_signature(data_folder)
    cache_path = CACHE_DIR / signature
    cache_path.mkdir(parents=True, exist_ok=True)

    chunks_file = cache_path / "chunks.pkl"
    index_file = cache_path / "faiss.index"
    meta_file = cache_path / "meta.pkl"

    pdf_files = []
    pages = []

    for file in sorted(os.listdir(data_folder)):
        if file.endswith(".pdf"):
            pdf_files.append(file)
            pdf_path = os.path.join(data_folder, file)
            extracted_pages = extract_text_from_pdf(pdf_path)

            for page in extracted_pages:
                page["source"] = file

            pages.extend(extracted_pages)

    if chunks_file.exists() and index_file.exists() and meta_file.exists():
        with open(chunks_file, "rb") as f:
            chunks = pickle.load(f)

        index = faiss.read_index(str(index_file))

        with open(meta_file, "rb") as f:
            meta = pickle.load(f)

        return {
            "chunks": chunks,
            "index": index,
            "pdf_files": pdf_files,
            "page_count": meta["page_count"],
            "chunk_count": meta["chunk_count"],
        }

    chunks = chunk_text(pages)
    embeddings = create_embeddings(chunks)
    index = create_vector_store(embeddings)

    with open(chunks_file, "wb") as f:
        pickle.dump(chunks, f)

    faiss.write_index(index, str(index_file))

    with open(meta_file, "wb") as f:
        pickle.dump(
            {
                "page_count": len(pages),
                "chunk_count": len(chunks),
            },
            f,
        )

    return {
        "chunks": chunks,
        "index": index,
        "pdf_files": pdf_files,
        "page_count": len(pages),
        "chunk_count": len(chunks),
    }

def clean_document_name(filename: str) -> str:
    """
    Convert ugly PDF filename into cleaner display name.
    Example:
    MACHINE LEARNING(R17A0534).pdf -> Machine Learning
    """
    if not filename:
        return "Document"

    name = filename.rsplit(".", 1)[0]

    # remove bracket/codes like (R17A0534)
    import re
    name = re.sub(r"\([^)]*\)", "", name)

    # replace underscores / hyphens
    name = name.replace("_", " ").replace("-", " ")

    # collapse spaces
    name = re.sub(r"\s+", " ", name).strip()

    # title case if fully uppercase-ish
    if name.isupper():
        name = name.title()

    return name if name else "Document"


def truncate_text(text: str, max_len: int = 28) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len - 3] + "..."


def get_dynamic_prompt_suggestions(doc_name: str):
    """
    Generic but adaptable starter prompts.
    Safe for changing PDFs.
    """
    clean_name = clean_document_name(doc_name)

    return [
        f"What is the main topic of {clean_name}?",
        "Summarize the key concepts in this document.",
        "Explain an important concept from this PDF.",
        "Compare two related topics mentioned in this document.",
    ]


# -----------------------------
# UI helpers
# -----------------------------
def confidence_class(label):
    return {
        "High": "chip chip-high",
        "Medium": "chip chip-medium",
        "Low": "chip chip-low",
    }.get(label, "chip chip-low")


def build_source_cards(results, debug=False):
    source_items = []

    for item in results:
        preview = item["text"][:260].strip()
        if len(item["text"]) > 260:
            preview += "..."

        data = {
            "label": f"{item['source']}",
            "page": f"Page {item['page']}",
            "preview": preview,
        }

        if debug:
            data["score"] = f"{item.get('score', 0):.4f}"
            data["dense"] = f"{item.get('dense_score', 0):.4f}"
            data["sparse"] = f"{item.get('sparse_score', 0):.4f}"
            if "rerank_score" in item:
                data["rerank"] = f"{item['rerank_score']:.4f}"

        source_items.append(data)

    return source_items


def render_sources(sources, debug=False):
    with st.expander("Sources", expanded=False):
        for item in sources:
            st.markdown(
                f"""
                <div class="source-card">
                    <div class="source-title">{item['label']}</div>
                    <div class="source-meta">{item['page']}</div>
                    <div>{item['preview']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if debug and "score" in item:
                extra = (
                    f"Final: {item['score']} | Dense: {item['dense']} | Sparse: {item['sparse']}"
                )
                if "rerank" in item:
                    extra += f" | Rerank: {item['rerank']}"
                st.caption(extra)


def render_sidebar(stats):
    document_name = stats["pdf_files"][0] if stats["pdf_files"] else "No document"
    display_name = clean_document_name(document_name)
    short_display_name = truncate_text(display_name, max_len=24)

    with st.sidebar:
        st.markdown("## 📄 Current Document")
        st.markdown(
            f"""
            <div class="panel" title="{document_name}">
                <div class="metric-label">Active PDF</div>
                <div class="metric-value">{short_display_name}</div>
            </div>
            """,
            unsafe_allow_html=True,
)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                f"""
                <div class="panel">
                    <div class="metric-label">Pages</div>
                    <div class="metric-value">{stats['page_count']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                f"""
                <div class="panel">
                    <div class="metric-label">Chunks</div>
                    <div class="metric-value">{stats['chunk_count']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("### 🛠 Options")
        with st.expander("Advanced options", expanded=False):
            debug_mode = st.checkbox("Debug mode", value=False)
            if debug_mode:
                st.caption(f"Full filename: {document_name}")

        if st.button("🗑 Clear chat", use_container_width=True):
            st.session_state.chat_history = []
            clear_history()
            st.rerun()

        st.markdown(
            "<div class='small-muted' style='margin-top:0.75rem;'>Single-document grounded Q&A with local retrieval and citations.</div>",
            unsafe_allow_html=True,
        )

    return debug_mode


def render_hero():
    st.markdown(
        """
        <div class="hero">
            <div class="hero-title">📘 PaperPilot AI</div>
            <div class="hero-subtitle">
                Ask grounded questions from your document and get concise answers with source-backed evidence.
            </div>
            <div class="pills">
                <span class="pill">Local</span>
                <span class="pill">Grounded</span>
                <span class="pill">Single-PDF</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def render_welcome_state(doc_name: str):
    suggestions = get_dynamic_prompt_suggestions(doc_name)

    st.markdown(
    """
        <div class="welcome-card">
            <div class="welcome-title">👋 Welcome</div>
            <div class="welcome-text">
                Ask grounded questions from your document. Answers will stay limited to the PDF content and include supporting sources.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section-title" style="font-size:1.05rem; margin-top:0.4rem; margin-bottom:0.55rem;">
            Try asking
        </div>
        """,
        unsafe_allow_html=True,
    )

    cols = st.columns(2)
    for i, suggestion in enumerate(suggestions):
        with cols[i % 2]:
            if st.button(suggestion, key=f"starter_{i}", use_container_width=True):
                st.session_state.prefill_question = suggestion

# -----------------------------
# Session state
# -----------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -----------------------------
# Load pipeline
# -----------------------------
try:
    stats = load_rag_pipeline("data")
except Exception as e:
    st.error(f"Error loading document: {e}")
    st.stop()

debug_mode = render_sidebar(stats)
render_hero()

document_name = stats["pdf_files"][0] if stats["pdf_files"] else "document"

if not st.session_state.chat_history:
    render_welcome_state(document_name)

# -----------------------------
# Render chat history
# -----------------------------
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            st.markdown(
                f"<span class='{confidence_class(message.get('confidence', 'Low'))}'>{message.get('confidence', 'Low')} confidence</span>",
                unsafe_allow_html=True,
            )

            if message["content"].strip() == REFUSAL_TEXT:
                st.markdown(
                    """
                    <div class="empty-card">
                        <strong>No grounded answer found in this document.</strong><br>
                        Try asking in a more specific way or refer to a concept clearly mentioned in the PDF.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"<div class='answer-card'>{message['content']}</div>",
                    unsafe_allow_html=True,
                )

            if debug_mode and message.get("rewritten_query"):
                st.caption(f"Rewritten query: {message['rewritten_query']}")

            if message.get("sources"):
                render_sources(message["sources"], debug=debug_mode)
        else:
            st.markdown(message["content"])

        st.markdown("<div class='chat-spacer'></div>", unsafe_allow_html=True)

# -----------------------------
# Chat input
# -----------------------------
if "prefill_question" not in st.session_state:
    st.session_state.prefill_question = ""

question = st.chat_input(
    "Ask a question about your PDF...",
    key="main_chat_input",
)

if not question and st.session_state.prefill_question:
    question = st.session_state.prefill_question
    st.session_state.prefill_question = ""

if question:
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the document..."):
            rewritten_query = rewrite_query(question)
            results = search(
                rewritten_query,
                stats["chunks"],
                stats["index"],
                top_k=7,
            )

        confidence = get_confidence_label(results)
        sources = build_source_cards(results, debug=debug_mode)

        if confidence == "Low" or not should_answer(results):
            answer = REFUSAL_TEXT
            confidence = "Low"
        else:
            with st.spinner("Generating answer..."):
                answer = generate_answer(rewritten_query, results)

        st.markdown(
            f"<span class='{confidence_class(confidence)}'>{confidence} confidence</span>",
            unsafe_allow_html=True,
        )

        if answer.strip() == REFUSAL_TEXT:
            st.markdown(
                """
                <div class="empty-card">
                    <strong>No grounded answer found in this document.</strong><br>
                    Try asking in a more specific way or refer to a concept clearly mentioned in the PDF.
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<div class='answer-card'>{answer}</div>",
                unsafe_allow_html=True,
            )

        if debug_mode:
            st.caption(f"Rewritten query: {rewritten_query}")

        if sources:
            render_sources(sources, debug=debug_mode)

    add_user_message(question)
    add_ai_message(answer)

    st.session_state.chat_history.append(
        {
            "role": "user",
            "content": question,
        }
    )
    st.session_state.chat_history.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "confidence": confidence,
            "rewritten_query": rewritten_query,
        }
    )