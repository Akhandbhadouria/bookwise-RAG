# 📚 SmartBook QA — RAG-based Book Question Answering System

Upload any PDF book, ask questions, and get accurate answers **strictly from the book's content** — with page numbers.

> **100% Free** — No paid APIs. Uses Groq (free tier) for LLM and HuggingFace (local) for embeddings.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?logo=streamlit&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-0.2+-1C3C3C?logo=langchain&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-Meta-0467DF?logo=meta&logoColor=white)

---

## ✨ Features

- 📄 **PDF Upload** — Drag-and-drop any book in PDF format.
- 🔪 **Smart Chunking** — Text is split into overlapping 1 000-char chunks.
- 🧠 **Semantic Search** — FAISS finds the most relevant passages via local embeddings.
- 🤖 **Llama 3.3 70B (via Groq)** — Generates answers only from retrieved context.
- 📖 **Page Numbers** — Every answer cites its source pages.
- 💬 **Chat History** — Follow-up questions understand prior context (last 5 exchanges).
- 🚫 **No Hallucination** — If the answer isn't in the book, it says so.
- 💰 **Completely Free** — No paid APIs required.

---

## 🏗️ Architecture

```
PDF Upload → Page Extraction → Text Chunking → Embedding (local) → FAISS Index
                                                                        ↓
User Question → Embed Question → Similarity Search → Top-4 Chunks
                                                                        ↓
                                      Constrained Prompt + Chat History
                                                                        ↓
                                        Llama 3.3 70B (Groq) → Answer + Pages
```

### How it works

1. **Upload** — User uploads a PDF book via Streamlit.
2. **Parse** — PyMuPDF extracts text page-by-page, preserving page numbers.
3. **Chunk** — Text is split into 1 000-char chunks with 200-char overlap.
4. **Embed** — Each chunk is converted to a 384-dim vector using HuggingFace `all-MiniLM-L6-v2` (runs locally, no API).
5. **Index** — Vectors are stored in a FAISS index for fast similarity search.
6. **Query** — User's question is embedded and matched against the index.
7. **Generate** — Top-4 chunks are sent to Llama 3.3 70B (via Groq) with a strict prompt that prevents hallucination.
8. **Display** — Answer + source page numbers are shown in the chat UI.

---

## 📁 Project Structure

```
smartbook-qa/
├── app.py               → Streamlit UI (chat interface, PDF upload)
├── rag_pipeline.py      → Core RAG logic (prompt + LLM call)
├── pdf_parser.py        → PDF reading and text chunking
├── vector_store.py      → FAISS embedding and retrieval
├── requirements.txt     → Dependencies
├── .gitignore           → Git ignore rules
├── .env                 → API keys (not committed)
└── README.md            → This file
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- A **free** [Groq API key](https://console.groq.com) (no credit card required)

### 1. Clone the Repository

```bash
git clone https://github.com/Akhandbhadouria/bookwise-RAG.git
cd bookwise-RAG
```

### 2. Create a Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Your API Key

Create a `.env` file in the project root:

```bash
echo 'GROQ_API_KEY=gsk_your_key_here' > .env
```

Or export it directly:

```bash
export GROQ_API_KEY="gsk_your_key_here"
```

> 💡 Get your free key at [console.groq.com](https://console.groq.com) — sign up, go to API Keys, and create one.

### 5. Run the App

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`. Upload a PDF and start asking questions!

---

## 🛠️ Tech Stack

| Component        | Technology                                         | Cost   |
|-----------------|----------------------------------------------------|--------|
| Language         | Python 3.10+                                       | Free   |
| LLM              | Llama 3.3 70B via [Groq](https://groq.com)         | Free   |
| Embeddings       | HuggingFace `all-MiniLM-L6-v2` (local)             | Free   |
| Vector DB        | FAISS (local, in-memory)                            | Free   |
| Framework        | LangChain                                           | Free   |
| PDF Parsing      | PyMuPDF (fitz)                                      | Free   |
| UI               | Streamlit                                           | Free   |
| Text Splitting   | RecursiveCharacterTextSplitter                      | Free   |

---

## 🎯 Example Questions

Once you upload a book, try questions like:

- *"What are the main themes of this book?"*
- *"Who is the protagonist and what is their goal?"*
- *"Summarize chapter 3."*
- *"What happens on page 42?"*
- *"How does the story end?"*

---

## 📝 License

MIT — free to use and modify.
