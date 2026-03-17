# 📘 PaperPilot AI — Local RAG Document Assistant

🚀 **Live Demo:** [paperpilot-ai.streamlit.app](https://paperpilot-ai.streamlit.app)

> Ask grounded questions from your PDF documents using local AI — no internet required.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)
![FAISS](https://img.shields.io/badge/VectorDB-FAISS-green)
![Ollama](https://img.shields.io/badge/LLM-Ollama-purple)

---

## 🎯 What is DocuMind?

DocuMind is a locally deployable AI-powered document assistant 
built on a custom RAG (Retrieval-Augmented Generation) pipeline.

Built from scratch — without LangChain or LlamaIndex — to deeply 
understand every component of a RAG system.

---

## ✨ Features

- 📥 PDF ingestion with PyMuPDF
- ✂️ Sentence-aware overlap chunking
- 🔍 Hybrid retrieval (dense + keyword)
- 🔁 Cross-encoder reranking
- 🧠 Query rewriting for follow-up questions
- 🚫 Hallucination control via answer gating
- 📊 Confidence scoring (High / Medium / Low)
- 📑 Source attribution with page numbers
- 💬 Conversational memory
- 🎨 Clean Streamlit dark UI

---

## 🧰 Tech Stack

| Component | Tool |
|---|---|
| Language | Python 3.10+ |
| PDF Processing | PyMuPDF (fitz) |
| Embeddings | SentenceTransformers (all-MiniLM-L6-v2) |
| Vector Database | FAISS |
| Reranking | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| LLM | Ollama — Qwen2.5 3B |
| UI | Streamlit |

---

## 🧠 System Architecture
```
PDF
↓
Text Extraction (PyMuPDF)
↓
Sentence-aware Chunking
↓
Embeddings (MiniLM)
↓
FAISS Vector Index
↓
User Question
↓
Query Rewriting
↓
Hybrid Retrieval
↓
Cross-Encoder Reranking
↓
Answer Gating
↓
Grounded Generation (Qwen2.5 3B)
↓
Confidence Score + Sources
↓
Streamlit UI
```

---

## 📂 Project Structure
```
DocuMind/
│
├── app/
│   ├── chunking.py
│   ├── embeddings.py
│   ├── generator.py
│   ├── ingestion.py
│   ├── memory.py
│   ├── query_rewriter.py
│   └── vector_store.py
│
├── streamlit_app.py
├── main.py
├── requirements.txt
└── README.md
```

---

## ⚙️ How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/DocuMind.git
cd DocuMind
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Install and start Ollama
Download from: https://ollama.com
```bash
ollama pull qwen2.5:3b
ollama run qwen2.5:3b
```

### 4. Run the app
```bash
streamlit run streamlit_app.py
```

---

## ⚠️ Hardware Requirements

Optimized for CPU-only machines.

| Spec | Minimum |
|---|---|
| RAM | 8 GB |
| GPU | Not required |
| OS | Windows / Linux / Mac |

---

## 🔮 Upcoming Features

- [ ] Multi-PDF document support
- [ ] Document upload via UI
- [ ] Semantic caching
- [ ] Deployment on Streamlit Cloud
- [ ] Evaluation metrics (RAGAS)

---

## 👨‍💻 Author

**Omkar Khurdal**
AI & Data Science Engineering
[GitHub](https://github.com/omkar-khurdal)
[LinkedIn](https://www.linkedin.com/in/omkar-khurdal-738716252?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app)