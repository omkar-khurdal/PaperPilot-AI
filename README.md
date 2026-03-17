# 📘 PaperPilot AI — Local RAG Document Assistant

🚀 **Live Demo:** [paperpilot-ai.streamlit.app](https://paperpilot-ai.streamlit.app)

> Ask grounded questions from your PDF documents using AI — upload any PDF and get source-backed answers instantly.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)
![FAISS](https://img.shields.io/badge/VectorDB-FAISS-green)
![Groq](https://img.shields.io/badge/LLM-Groq-orange)

---

## 🎯 What is PaperPilot AI?

PaperPilot AI is a deployable AI-powered document assistant
built on a custom RAG (Retrieval-Augmented Generation) pipeline.

Built from scratch — without LangChain or LlamaIndex — to deeply
understand every component of a RAG system.

---

## ✨ Features

- 📥 PDF upload directly from UI — no hardcoded files
- ✂️ Sentence-aware overlap chunking
- 🔍 Hybrid retrieval (dense + keyword scoring)
- 🔁 Cross-encoder reranking for better relevance
- 🧠 Query rewriting for conversational follow-ups
- 🚫 Hallucination control via answer gating
- 📊 Confidence scoring (High / Medium / Low)
- 📑 Source attribution with page numbers
- 💬 Conversational memory
- 🎨 Clean Streamlit dark UI
- ☁️ Deployed free on Streamlit Cloud

---

## 🧰 Tech Stack

| Component | Tool |
|---|---|
| Language | Python 3.10+ |
| PDF Processing | PyMuPDF (fitz) |
| Embeddings | SentenceTransformers (all-MiniLM-L6-v2) |
| Vector Database | FAISS |
| Reranking | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| LLM | Groq API — LLaMA 3.3 70B |
| UI | Streamlit |

---

## 🧠 System Architecture
```
PDF Upload (UI)
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
Hybrid Retrieval (Dense + Keyword)
↓
Cross-Encoder Reranking
↓
Answer Gating
↓
Grounded Generation (LLaMA 3.3 70B via Groq)
↓
Confidence Score + Sources
↓
Streamlit UI
```

---

## 📂 Project Structure
```
PaperPilot-AI/
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
git clone https://github.com/omkar-khurdal/PaperPilot-AI.git
cd PaperPilot-AI
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add your Groq API key
Create a `.env` file in the project folder:
```
GROQ_API_KEY=your_groq_api_key_here
```
Get a free API key at: https://console.groq.com

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
- [ ] Semantic caching
- [ ] Evaluation metrics (RAGAS)
- [ ] Section-aware retrieval
- [ ] Metadata filtering

---

## 👨‍💻 Author

**Omkar Khurdal**  
AI & Data Science Engineering

[GitHub](https://github.com/omkar-khurdal) • [LinkedIn](https://www.linkedin.com/in/omkar-khurdal-738716252)