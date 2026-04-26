# 📚 SmartBook QA — RAG-based Book Question Answering System

Upload any PDF book, ask questions, and get accurate answers **strictly from the book's content** — with page numbers.

## ✨ Features

- 📄 **PDF Upload** — Drag-and-drop any book in PDF format.
- 🔪 **Smart Chunking** — Text is split into overlapping 1 000-char chunks.
- 🧠 **Semantic Search** — FAISS finds the most relevant passages via embeddings.
- 🤖 **GPT-3.5-turbo** — Generates answers only from retrieved context.
- 📖 **Page Numbers** — Every answer cites its source pages.
- 💬 **Chat History** — Follow-up questions understand prior context.
- 🚫 **No Hallucination** — If the answer isn't in the book, it says so.

## 🏗️ Architecture

```
PDF Upload → Page Extraction → Text Chunking → Embedding → FAISS Index
                                                                  ↓
User Question → Embed Question → Similarity Search → Top-K Chunks
                                                                  ↓
                                    Constrained Prompt + Chat History
                                                                  ↓
                                             GPT-3.5-turbo → Answer + Pages
```

## 📁 Project Structure

```
smartbook-qa/
├── app.py               → Streamlit UI
├── rag_pipeline.py      → Core RAG logic (prompt + LLM)
├── pdf_parser.py        → PDF reading and chunking
├── vector_store.py      → FAISS embedding and retrieval
├── requirements.txt     → Dependencies
└── README.md            → This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- An [OpenAI API key](https://platform.openai.com/api-keys)

### Installation

```bash
# Clone or navigate to the project directory
cd smartbook-qa

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

### Set Your API Key

**Option A — Environment variable:**
```bash
export OPENAI_API_KEY="sk-..."
```

**Option B — `.env` file (recommended):**
```bash
echo 'OPENAI_API_KEY=sk-...' > .env
```

### Run the App

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

## 🛠️ Tech Stack

| Component        | Technology                     |
|-----------------|-------------------------------|
| Language         | Python                         |
| LLM              | OpenAI GPT-3.5-turbo           |
| Embeddings       | OpenAI text-embedding-ada-002  |
| Vector DB        | FAISS (local, in-memory)       |
| Framework        | LangChain                      |
| PDF Parsing      | PyMuPDF (fitz)                 |
| UI               | Streamlit                      |
| Text Splitting   | RecursiveCharacterTextSplitter |

## 📝 License

MIT — free to use and modify.
