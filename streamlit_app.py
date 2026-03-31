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
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Inter:wght@300;400;500;600&display=swap');

:root {
    --bg-deep:       #040c18;
    --bg-mid:        #071020;
    --bg-surface:    #0b1628;
    --bg-raised:     #0f1e35;
    --border:        rgba(99, 179, 237, 0.10);
    --border-bright: rgba(99, 179, 237, 0.22);
    --text:          #e8f4ff;
    --text-muted:    #6b8cae;
    --text-dim:      #3d5a7a;
    --accent:        #38bdf8;
    --accent-glow:   rgba(56, 189, 248, 0.15);
    --accent-soft:   rgba(56, 189, 248, 0.08);
    --green:         #34d399;
    --green-soft:    rgba(52, 211, 153, 0.12);
    --amber:         #fbbf24;
    --amber-soft:    rgba(251, 191, 36, 0.12);
    --red:           #f87171;
    --red-soft:      rgba(248, 113, 113, 0.12);
    --radius-sm:     10px;
    --radius-md:     16px;
    --radius-lg:     22px;
    --radius-xl:     28px;
}

* { box-sizing: border-box; }

html, body, .stApp {
    font-family: 'Inter', sans-serif;
    background: var(--bg-deep);
    color: var(--text);
}

.stApp {
    background:
        radial-gradient(ellipse 80% 50% at 10% -10%, rgba(56,189,248,0.06) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 90% 100%, rgba(99,102,241,0.05) 0%, transparent 55%),
        linear-gradient(180deg, #040c18 0%, #060f1c 50%, #050d1a 100%);
}

.block-container {
    max-width: 960px !important;
    padding: 1.5rem 1.8rem 3rem !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #060e1d 0%, #080f1e 100%) !important;
    border-right: 1px solid var(--border) !important;
}

[data-testid="stSidebar"] > div {
    padding: 1.4rem 1rem !important;
}

/* Hide default Streamlit file uploader label ugliness */
[data-testid="stFileUploader"] {
    background: transparent !important;
}

[data-testid="stFileUploader"] > div {
    background: var(--accent-soft) !important;
    border: 1.5px dashed var(--border-bright) !important;
    border-radius: var(--radius-md) !important;
    padding: 1rem !important;
    transition: all 0.2s ease !important;
}

[data-testid="stFileUploader"] > div:hover {
    border-color: var(--accent) !important;
    background: rgba(56,189,248,0.12) !important;
}

[data-testid="stFileUploader"] label {
    display: none !important;
}

/* Upload drop zone text */
[data-testid="stFileUploaderDropzone"] {
    background: transparent !important;
}

/* Hide ALL default uploader instruction text */
[data-testid="stFileUploaderDropzoneInstructions"] {
    font-size: 0 !important;
}

/* Rebuild only the text you want */
[data-testid="stFileUploaderDropzoneInstructions"] div {
    font-size: 0 !important;
}

[data-testid="stFileUploaderDropzoneInstructions"]::before {
    content: "Drop your PDF here";
    display: block;
    text-align: center;
    font-size: 0.92rem;
    color: var(--text-muted);
    font-family: 'Inter', sans-serif;
    margin-bottom: 0.3rem;
}

[data-testid="stFileUploaderDropzoneInstructions"]::after {
    content: "PDF only";
    display: block;
    text-align: center;
    font-size: 0.75rem;
    color: var(--text-dim);
    font-family: 'Inter', sans-serif;
    margin-top: 0.2rem;
}

/* Browse files button */
[data-testid="stFileUploader"] button {
    background: var(--bg-raised) !important;
    border: 1px solid var(--border-bright) !important;
    color: var(--accent) !important;
    border-radius: var(--radius-sm) !important;
    font-size: 0.8rem !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    padding: 0.4rem 1rem !important;
    transition: all 0.2s !important;
}

[data-testid="stFileUploader"] button:hover {
    background: var(--accent-glow) !important;
    border-color: var(--accent) !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border-bright); border-radius: 4px; }

/* ── Chat input ── */
[data-testid="stChatInput"] {
    background: var(--bg-raised) !important;
    border: 1px solid var(--border-bright) !important;
    border-radius: var(--radius-lg) !important;
    backdrop-filter: blur(12px) !important;
}

[data-testid="stChatInput"] textarea {
    background: transparent !important;
    color: var(--text) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.94rem !important;
}

[data-testid="stChatInput"] textarea::placeholder {
    color: var(--text-dim) !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    background: var(--bg-raised) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
}

[data-testid="stExpander"] summary {
    color: var(--text-muted) !important;
    font-size: 0.85rem !important;
}

/* ── Buttons ── */
.stButton > button {
    background: var(--bg-raised) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-muted) !important;
    border-radius: var(--radius-sm) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.83rem !important;
    font-weight: 500 !important;
    transition: all 0.18s ease !important;
    padding: 0.5rem 1rem !important;
}

.stButton > button:hover {
    border-color: var(--border-bright) !important;
    color: var(--text) !important;
    background: rgba(56,189,248,0.06) !important;
    transform: translateY(-1px) !important;
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 0.2rem 0 !important;
}

/* ── Spinner ── */
.stSpinner > div {
    border-top-color: var(--accent) !important;
}

/* ── Custom components ── */

.sidebar-brand {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin-bottom: 1.4rem;
    padding-bottom: 1.2rem;
    border-bottom: 1px solid var(--border);
}

.sidebar-brand-icon {
    font-size: 1.4rem;
    line-height: 1;
}

.sidebar-brand-name {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 1rem;
    color: var(--text);
    letter-spacing: 0.3px;
}

.sidebar-brand-version {
    font-size: 0.68rem;
    color: var(--text-dim);
    font-weight: 400;
    margin-top: 1px;
}

.sidebar-section-label {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    color: var(--text-dim);
    margin: 1.2rem 0 0.6rem 0;
}

.upload-label {
    font-size: 0.78rem;
    font-weight: 600;
    color: var(--text-muted);
    margin-bottom: 0.5rem;
    letter-spacing: 0.3px;
}

.doc-card {
    background: linear-gradient(135deg, var(--bg-raised), rgba(15,30,55,0.9));
    border: 1px solid var(--border-bright);
    border-radius: var(--radius-md);
    padding: 0.85rem 1rem;
    margin: 0.75rem 0;
    position: relative;
    overflow: hidden;
}

.doc-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 3px; height: 100%;
    background: linear-gradient(180deg, var(--accent), rgba(56,189,248,0.2));
    border-radius: 3px 0 0 3px;
}

.doc-label {
    font-size: 0.68rem;
    color: var(--text-dim);
    text-transform: uppercase;
    letter-spacing: 0.8px;
    font-weight: 600;
    margin-bottom: 0.3rem;
}

.doc-name {
    font-family: 'Syne', sans-serif;
    font-size: 0.92rem;
    font-weight: 700;
    color: var(--text);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.stats-row {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.5rem;
    margin: 0.5rem 0;
}

.stat-box {
    background: var(--bg-surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 0.6rem 0.75rem;
    text-align: center;
}

.stat-num {
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    font-weight: 800;
    color: var(--accent);
    line-height: 1;
}

.stat-lbl {
    font-size: 0.67rem;
    color: var(--text-dim);
    text-transform: uppercase;
    letter-spacing: 0.6px;
    margin-top: 0.2rem;
}

.hero {
    background: linear-gradient(135deg,
        rgba(11,22,40,0.97) 0%,
        rgba(13,26,48,0.95) 100%);
    border: 1px solid var(--border-bright);
    border-radius: var(--radius-xl);
    padding: 1.6rem 1.8rem 1.4rem;
    margin-bottom: 1rem;
    position: relative;
    overflow: hidden;
}

.hero::before {
    content: '';
    position: absolute;
    top: -40px; right: -40px;
    width: 180px; height: 180px;
    background: radial-gradient(circle, rgba(56,189,248,0.08) 0%, transparent 70%);
    pointer-events: none;
}

.hero::after {
    content: '';
    position: absolute;
    bottom: -30px; left: -30px;
    width: 120px; height: 120px;
    background: radial-gradient(circle, rgba(99,102,241,0.06) 0%, transparent 70%);
    pointer-events: none;
}

.hero-eyebrow {
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 0.5rem;
    opacity: 0.8;
}

.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.9rem;
    font-weight: 800;
    color: var(--text);
    line-height: 1.15;
    margin-bottom: 0.5rem;
    letter-spacing: -0.3px;
}

.hero-title span {
    background: linear-gradient(135deg, var(--accent), #818cf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.hero-sub {
    color: var(--text-muted);
    font-size: 0.9rem;
    line-height: 1.55;
    max-width: 520px;
    margin-bottom: 1rem;
}

.pills {
    display: flex;
    gap: 0.45rem;
    flex-wrap: wrap;
}

.pill {
    background: var(--accent-soft);
    border: 1px solid rgba(56,189,248,0.18);
    color: rgba(56,189,248,0.85);
    border-radius: 999px;
    padding: 0.28rem 0.75rem;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.2px;
}

.welcome-card {
    background: linear-gradient(135deg, var(--bg-surface), var(--bg-raised));
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 1.3rem 1.5rem;
    margin-bottom: 0.9rem;
}

.welcome-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: var(--text);
    margin-bottom: 0.35rem;
}

.welcome-text {
    color: var(--text-muted);
    font-size: 0.88rem;
    line-height: 1.55;
}

.try-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 1.1px;
    text-transform: uppercase;
    color: var(--text-dim);
    margin: 1rem 0 0.55rem;
}

.answer-wrap {
    margin-top: 0.35rem;
}

.confidence-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.25rem 0.7rem;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.3px;
    margin-bottom: 0.55rem;
    text-transform: uppercase;
}

.conf-high {
    background: var(--green-soft);
    border: 1px solid rgba(52,211,153,0.25);
    color: var(--green);
}

.conf-high::before { content: '●'; font-size: 0.55rem; }

.conf-medium {
    background: var(--amber-soft);
    border: 1px solid rgba(251,191,36,0.25);
    color: var(--amber);
}

.conf-medium::before { content: '●'; font-size: 0.55rem; }

.conf-low {
    background: var(--red-soft);
    border: 1px solid rgba(248,113,113,0.25);
    color: var(--red);
}

.conf-low::before { content: '●'; font-size: 0.55rem; }

.answer-card {
    background: linear-gradient(160deg, var(--bg-surface), var(--bg-raised));
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 1.05rem 1.2rem;
    line-height: 1.7;
    font-size: 0.92rem;
    color: var(--text);
    position: relative;
}

.answer-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 100%; height: 1px;
    background: linear-gradient(90deg, var(--accent), transparent 60%);
    opacity: 0.3;
    border-radius: var(--radius-lg) var(--radius-lg) 0 0;
}

.empty-card {
    background: var(--red-soft);
    border: 1px solid rgba(248,113,113,0.2);
    border-radius: var(--radius-md);
    padding: 0.9rem 1.1rem;
    font-size: 0.88rem;
    color: #fca5a5;
    line-height: 1.55;
}

.empty-card strong {
    display: block;
    color: var(--red);
    font-size: 0.82rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 0.3rem;
}

.source-section-label {
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: var(--text-dim);
    margin: 0.9rem 0 0.45rem;
}

.source-card {
    background: var(--bg-surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 0.8rem 0.95rem;
    margin-bottom: 0.5rem;
    transition: border-color 0.15s;
}

.source-card:hover {
    border-color: var(--border-bright);
}

.source-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 0.3rem;
}

.source-name {
    font-size: 0.8rem;
    font-weight: 600;
    color: var(--text);
}

.source-page {
    font-size: 0.7rem;
    color: var(--accent);
    background: var(--accent-soft);
    border: 1px solid rgba(56,189,248,0.15);
    border-radius: 999px;
    padding: 0.1rem 0.55rem;
    font-weight: 600;
}

.source-preview {
    font-size: 0.8rem;
    color: var(--text-muted);
    line-height: 1.5;
}

.upload-prompt {
    background: linear-gradient(135deg, var(--bg-surface), var(--bg-raised));
    border: 1px dashed var(--border-bright);
    border-radius: var(--radius-lg);
    padding: 2rem 1.5rem;
    text-align: center;
    margin-top: 0.75rem;
}

.upload-prompt-icon {
    font-size: 2rem;
    margin-bottom: 0.6rem;
    opacity: 0.6;
}

.upload-prompt-text {
    color: var(--text-muted);
    font-size: 0.9rem;
    margin-bottom: 0.3rem;
}

.upload-prompt-sub {
    color: var(--text-dim);
    font-size: 0.78rem;
}

.divider {
    height: 1px;
    background: var(--border);
    margin: 0.75rem 0;
}

.debug-tag {
    font-size: 0.72rem;
    color: var(--text-dim);
    font-family: 'JetBrains Mono', monospace;
    background: var(--bg-surface);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 0.2rem 0.5rem;
    margin-top: 0.4rem;
    display: inline-block;
}

.spacer { height: 0.4rem; }

</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# -----------------------------
# Cache
# -----------------------------
CACHE_DIR = Path(".rag_cache")
CACHE_DIR.mkdir(exist_ok=True)


def get_data_signature(file_bytes, filename):
    content_hash = hashlib.sha256(file_bytes).hexdigest()
    return content_hash[:32]


@st.cache_resource(show_spinner=False)
def load_rag_pipeline_from_bytes(file_bytes, filename):
    signature = get_data_signature(file_bytes, filename)
    cache_path = CACHE_DIR / signature
    cache_path.mkdir(parents=True, exist_ok=True)

    chunks_file = cache_path / "chunks.pkl"
    index_file  = cache_path / "faiss.index"
    meta_file   = cache_path / "meta.pkl"

    temp_path = cache_path / filename
    with open(temp_path, "wb") as f:
        f.write(file_bytes)

    if chunks_file.exists() and index_file.exists() and meta_file.exists():
        with open(chunks_file, "rb") as f:
            chunks = pickle.load(f)
        index = faiss.read_index(str(index_file))
        with open(meta_file, "rb") as f:
            meta = pickle.load(f)
        return {
            "chunks": chunks, "index": index,
            "pdf_files": [filename],
            "page_count": meta["page_count"],
            "chunk_count": meta["chunk_count"],
        }

    pages = extract_text_from_pdf(str(temp_path))
    for page in pages:
        page["source"] = filename

    chunks     = chunk_text(pages)
    embeddings = create_embeddings(chunks)
    index      = create_vector_store(embeddings)

    with open(chunks_file, "wb") as f: pickle.dump(chunks, f)
    faiss.write_index(index, str(index_file))
    with open(meta_file, "wb") as f:
        pickle.dump({"page_count": len(pages), "chunk_count": len(chunks)}, f)

    return {
        "chunks": chunks, "index": index,
        "pdf_files": [filename],
        "page_count": len(pages),
        "chunk_count": len(chunks),
    }


# -----------------------------
# Helpers
# -----------------------------
def clean_document_name(filename: str) -> str:
    import re
    if not filename:
        return "Document"
    name = filename.rsplit(".", 1)[0]
    name = re.sub(r"\([^)]*\)", "", name)
    name = name.replace("_", " ").replace("-", " ")
    name = re.sub(r"\s+", " ", name).strip()
    if name.isupper():
        name = name.title()
    return name or "Document"


def truncate_text(text: str, max_len: int = 26) -> str:
    return text if len(text) <= max_len else text[:max_len - 1] + "…"


def get_suggestions(doc_name: str):
    n = clean_document_name(doc_name)
    return [
        f"What is the main topic of {n}?",
        "Summarize the key concepts in this document.",
        "Explain an important concept from this PDF.",
        "Compare two related topics mentioned here.",
    ]


def confidence_class(label):
    return {
        "High":   "conf-high",
        "Medium": "conf-medium",
        "Low":    "conf-low",
    }.get(label, "conf-low")


def build_source_cards(results, debug=False):
    items = []
    for item in results:
        preview = item["text"][:240].strip()
        if len(item["text"]) > 240:
            preview += "…"
        d = {
            "label":   item["source"],
            "page":    item["page"],
            "preview": preview,
        }
        if debug:
            d["score"]  = f"{item.get('score', 0):.4f}"
            d["dense"]  = f"{item.get('dense_score', 0):.4f}"
            d["sparse"] = f"{item.get('sparse_score', 0):.4f}"
            if "rerank_score" in item:
                d["rerank"] = f"{item['rerank_score']:.4f}"
        items.append(d)
    return items


def render_sources(sources, debug=False):
    st.markdown(
        "<div class='source-section-label'>Sources</div>",
        unsafe_allow_html=True,
    )
    for item in sources:
        debug_line = ""
        if debug and "score" in item:
            extra = f"score {item['score']} · dense {item['dense']} · sparse {item['sparse']}"
            if "rerank" in item:
                extra += f" · rerank {item['rerank']}"
            debug_line = f"<div class='debug-tag'>{extra}</div>"

        st.markdown(
            f"""
            <div class="source-card">
                <div class="source-header">
                    <span class="source-name">📄 {item['label']}</span>
                    <span class="source-page">Page {item['page']}</span>
                </div>
                <div class="source-preview">{item['preview']}</div>
                {debug_line}
            </div>
            """,
            unsafe_allow_html=True,
        )


# -----------------------------
# Session state
# -----------------------------
if "chat_history"      not in st.session_state: st.session_state.chat_history      = []
if "prefill_question"  not in st.session_state: st.session_state.prefill_question  = ""
if "current_pdf"       not in st.session_state: st.session_state.current_pdf       = None

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:

    # Brand
    st.markdown(
        """
        <div class="sidebar-brand">
            <span class="sidebar-brand-icon">📘</span>
            <div>
                <div class="sidebar-brand-name">PaperPilot AI</div>
                <div class="sidebar-brand-version">v1.0 · Powered by Groq</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Upload
    st.markdown("<div class='upload-label'>📂 Upload Document</div>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "upload",
        type="pdf",
        key="pdf_uploader",
        label_visibility="collapsed",
    )

    # If PDF loaded — show stats
    if uploaded_file is not None:
        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
        st.markdown("<div class='sidebar-section-label'>Active Document</div>", unsafe_allow_html=True)

        display_name = clean_document_name(uploaded_file.name)
        short_name   = truncate_text(display_name, max_len=22)

        st.markdown(
            f"""
            <div class="doc-card">
                <div class="doc-label">Active PDF</div>
                <div class="doc-name" title="{uploaded_file.name}">{short_name}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Stats placeholder — filled after pipeline loads
        stats_placeholder = st.empty()

        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
        st.markdown("<div class='sidebar-section-label'>Settings</div>", unsafe_allow_html=True)

        with st.expander("Advanced options", expanded=False):
            debug_mode = st.checkbox("Debug mode", value=False)
            max_tokens = st.slider(
                "Answer length",
                min_value=80,
                max_value=400,
                value=160,
                step=40,
                help="Controls how long answers can be"
            )

        if st.button("🗑 Clear chat", use_container_width=True):
            st.session_state.chat_history = []
            clear_history()
            st.rerun()

        st.markdown(
            "<div style='margin-top:1rem; font-size:0.72rem; color:var(--text-dim); line-height:1.5;'>"
            "Grounded Q&A · Source citations · Hallucination control"
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        debug_mode = False
        max_tokens = 160

# -----------------------------
# No PDF — show upload prompt
# -----------------------------
if uploaded_file is None:
    st.markdown(
        """
        <div class="hero">
            <div class="hero-eyebrow">AI Document Assistant</div>
            <div class="hero-title">Ask anything from<br><span>your PDF</span></div>
            <div class="hero-sub">
                Upload a research paper, textbook, or any PDF and get
                precise, source-backed answers in seconds.
            </div>
            <div class="pills">
                <span class="pill">⚡ Groq LLaMA 3.3</span>
                <span class="pill">🔍 Hybrid Retrieval</span>
                <span class="pill">🛡 Grounded Answers</span>
                <span class="pill">📑 Source Citations</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="upload-prompt">
            <div class="upload-prompt-icon">📂</div>
            <div class="upload-prompt-text">Upload a PDF from the sidebar to get started</div>
            <div class="upload-prompt-sub">Supports research papers, textbooks, reports · Max 50MB</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

# -----------------------------
# PDF uploaded — check if new
# -----------------------------
if st.session_state.current_pdf != uploaded_file.name:
    st.session_state.chat_history = []
    clear_history()
    st.session_state.current_pdf = uploaded_file.name

# -----------------------------
# Load pipeline
# -----------------------------
file_bytes = uploaded_file.read()

try:
    with st.spinner("Processing document…"):
        stats = load_rag_pipeline_from_bytes(file_bytes, uploaded_file.name)
except Exception as e:
    st.error(f"Error processing document: {e}")
    st.stop()

# Fill stats in sidebar
with stats_placeholder:
    st.markdown(
        f"""
        <div class="stats-row">
            <div class="stat-box">
                <div class="stat-num">{stats['page_count']}</div>
                <div class="stat-lbl">Pages</div>
            </div>
            <div class="stat-box">
                <div class="stat-num">{stats['chunk_count']}</div>
                <div class="stat-lbl">Chunks</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# -----------------------------
# Hero
# -----------------------------
st.markdown(
    """
    <div class="hero">
        <div class="hero-eyebrow">AI Document Assistant</div>
        <div class="hero-title">📘 <span>PaperPilot AI</span></div>
        <div class="hero-sub">
            Ask grounded questions from your document and get concise,
            source-backed answers with confidence scoring.
        </div>
        <div class="pills">
            <span class="pill">⚡ Groq LLaMA 3.3</span>
            <span class="pill">🔍 Hybrid Retrieval</span>
            <span class="pill">🛡 Grounded</span>
            <span class="pill">📄 Single-PDF</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Welcome state
# -----------------------------
document_name = uploaded_file.name

if not st.session_state.chat_history:
    suggestions = get_suggestions(document_name)

    st.markdown(
        """
        <div class="welcome-card">
            <div class="welcome-title">👋 Welcome</div>
            <div class="welcome-text">
                Your document is ready. Ask any question — answers stay
                strictly within the PDF content with supporting sources.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div class='try-label'>Try asking</div>", unsafe_allow_html=True)

    cols = st.columns(2)
    for i, s in enumerate(suggestions):
        with cols[i % 2]:
            if st.button(s, key=f"starter_{i}", use_container_width=True):
                st.session_state.prefill_question = s

# -----------------------------
# Render chat history
# -----------------------------
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            conf  = message.get("confidence", "Low")
            cname = confidence_class(conf)

            st.markdown(
                f"<div class='confidence-badge {cname}'>{conf} Confidence</div>",
                unsafe_allow_html=True,
            )

            if message["content"].strip() == REFUSAL_TEXT:
                st.markdown(
                    """
                    <div class="empty-card">
                        <strong>No answer found</strong>
                        This question does not appear to be covered in the document.
                        Try rephrasing or asking about a concept clearly mentioned in the PDF.
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
                st.markdown(
                    f"<div class='debug-tag'>↺ rewritten: {message['rewritten_query']}</div>",
                    unsafe_allow_html=True,
                )

            if message.get("sources"):
                render_sources(message["sources"], debug=debug_mode)

        else:
            st.markdown(message["content"])

        st.markdown("<div class='spacer'></div>", unsafe_allow_html=True)

# -----------------------------
# Chat input
# -----------------------------
question = st.chat_input(
    "Ask a question about your PDF…",
    key="main_chat_input",
)

if not question and st.session_state.prefill_question:
    question = st.session_state.prefill_question
    st.session_state.prefill_question = ""

if question:
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching document…"):
            rewritten_query = rewrite_query(question)
            results = search(
                rewritten_query,
                stats["chunks"],
                stats["index"],
                top_k=7,
            )

        confidence = get_confidence_label(results)
        sources    = build_source_cards(results, debug=debug_mode)

        if confidence == "Low" or not should_answer(results):
            answer     = REFUSAL_TEXT
            confidence = "Low"
        else:
            with st.spinner("Generating answer…"):
                answer = generate_answer(rewritten_query, results, max_tokens=max_tokens)

        cname = confidence_class(confidence)
        st.markdown(
            f"<div class='confidence-badge {cname}'>{confidence} Confidence</div>",
            unsafe_allow_html=True,
        )

        if answer.strip() == REFUSAL_TEXT:
            st.markdown(
                """
                <div class="empty-card">
                    <strong>No answer found</strong>
                    This question does not appear to be covered in the document.
                    Try rephrasing or asking about a concept clearly mentioned in the PDF.
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
            st.markdown(
                f"<div class='debug-tag'>↺ rewritten: {rewritten_query}</div>",
                unsafe_allow_html=True,
            )

        if sources:
            render_sources(sources, debug=debug_mode)

    add_user_message(question)
    add_ai_message(answer)

    st.session_state.chat_history.append({"role": "user", "content": question})
    st.session_state.chat_history.append({
        "role":            "assistant",
        "content":         answer,
        "sources":         sources,
        "confidence":      confidence,
        "rewritten_query": rewritten_query,
    })
