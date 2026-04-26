"""
app.py — Streamlit UI for SmartBook QA

This is the entry point. Run with:
    streamlit run app.py

Responsibilities:
  - Accept a PDF upload from the user.
  - Parse, chunk, and embed the PDF (once per upload).
  - Provide a chat interface for asking questions.
  - Display answers with source page numbers.
  - Maintain chat history in Streamlit session state.
"""

import streamlit as st
from dotenv import load_dotenv

from pdf_parser import extract_text_from_pdf, chunk_documents
from vector_store import create_vector_store
from rag_pipeline import get_answer

# ── Load environment variables (.env file for GROQ_API_KEY) ──
load_dotenv()


# ── Page configuration ─────────────────────────────────────────
st.set_page_config(
    page_title="SmartBook QA",
    page_icon="📚",
    layout="centered",
    initial_sidebar_state="expanded",
)


# ── Custom CSS for a clean, modern look ────────────────────────
st.markdown(
    """
    <style>
    /* ── Global ─────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── Sidebar ────────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    section[data-testid="stSidebar"] * {
        color: #e0e0e0 !important;
    }

    /* ── Header ─────────────────────────────────────── */
    .main-header {
        text-align: center;
        padding: 1.5rem 0 0.5rem;
    }
    .main-header h1 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.4rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }
    .main-header p {
        color: #888;
        font-size: 1rem;
    }

    /* ── Chat bubbles ───────────────────────────────── */
    .chat-user {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: #fff;
        padding: 0.85rem 1.1rem;
        border-radius: 1rem 1rem 0.25rem 1rem;
        margin: 0.5rem 0;
        max-width: 85%;
        margin-left: auto;
        font-size: 0.95rem;
    }
    .chat-bot {
        background: #1e1e2f;
        color: #e0e0e0;
        padding: 0.85rem 1.1rem;
        border-radius: 1rem 1rem 1rem 0.25rem;
        margin: 0.5rem 0;
        max-width: 85%;
        font-size: 0.95rem;
        border: 1px solid #2a2a3d;
    }
    .source-badge {
        display: inline-block;
        background: #667eea22;
        color: #667eea;
        padding: 0.2rem 0.6rem;
        border-radius: 0.5rem;
        font-size: 0.78rem;
        margin-top: 0.4rem;
        font-weight: 600;
    }

    /* ── Upload area ────────────────────────────────── */
    .upload-area {
        border: 2px dashed #667eea55;
        border-radius: 1rem;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    }

    /* ── Status box ─────────────────────────────────── */
    .status-box {
        background: #667eea15;
        border-left: 4px solid #667eea;
        padding: 0.8rem 1rem;
        border-radius: 0 0.5rem 0.5rem 0;
        margin: 0.5rem 0;
        font-size: 0.88rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Session state initialization ───────────────────────────────

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of {"role", "content"} dicts

if "book_name" not in st.session_state:
    st.session_state.book_name = None

if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0


# ── Header ─────────────────────────────────────────────────────

st.markdown(
    """
    <div class="main-header">
        <h1>📚 SmartBook QA</h1>
        <p>Upload a book. Ask anything. Get answers with page numbers.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ── Sidebar: PDF upload & info ─────────────────────────────────

with st.sidebar:
    st.markdown("## 📖 Upload Your Book")

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        help="Upload the book you want to ask questions about.",
    )

    if uploaded_file is not None:
        # Only process if it's a new file (avoid re-processing on rerun)
        if st.session_state.book_name != uploaded_file.name:
            with st.spinner("🔍 Reading and processing the book..."):
                # Step 1 — Extract text from PDF
                documents = extract_text_from_pdf(uploaded_file)

                # Step 2 — Chunk the text
                chunks = chunk_documents(documents)

                # Step 3 — Build FAISS index
                vector_store = create_vector_store(chunks)

                # Save to session state
                st.session_state.vector_store = vector_store
                st.session_state.book_name = uploaded_file.name
                st.session_state.chunk_count = len(chunks)
                st.session_state.chat_history = []  # reset history for new book

            st.success("✅ Book processed successfully!")

        # Show book info
        st.markdown("---")
        st.markdown("### 📊 Book Stats")
        st.markdown(
            f"""
            <div class="status-box">
                <strong>Book:</strong> {st.session_state.book_name}<br>
                <strong>Chunks:</strong> {st.session_state.chunk_count}
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown(
        "### 💡 Tips\n"
        "- Ask specific questions for best results.\n"
        "- You can ask follow-up questions.\n"
        "- Answers include source page numbers.\n"
    )


# ── Main area: Chat interface ─────────────────────────────────

if st.session_state.vector_store is None:
    # No book uploaded yet — show a helpful prompt
    st.markdown(
        """
        <div class="upload-area">
            <h3>👈 Upload a PDF book to get started</h3>
            <p style="color: #888;">
                Once uploaded, you can ask any question about the book
                and get accurate answers with page references.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    # ── Display chat history ──────────────────────────────────
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="chat-user">{msg["content"]}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="chat-bot">{msg["content"]}</div>',
                unsafe_allow_html=True,
            )

    # ── Chat input ────────────────────────────────────────────
    user_question = st.chat_input("Ask a question about the book...")

    if user_question:
        # Show the user's message immediately
        st.markdown(
            f'<div class="chat-user">{user_question}</div>',
            unsafe_allow_html=True,
        )

        with st.spinner("🤔 Searching the book and generating answer..."):
            result = get_answer(
                vector_store=st.session_state.vector_store,
                question=user_question,
                chat_history=st.session_state.chat_history,
            )

        answer = result["answer"]
        pages = result["source_pages"]
        pages_str = ", ".join(str(p) for p in pages)

        # Display the answer
        st.markdown(
            f'<div class="chat-bot">{answer}<br>'
            f'<span class="source-badge">📄 Source pages: {pages_str}</span>'
            f"</div>",
            unsafe_allow_html=True,
        )

        # Update chat history (keep last 5 exchanges = 10 messages)
        st.session_state.chat_history.append(
            {"role": "user", "content": user_question}
        )
        st.session_state.chat_history.append(
            {"role": "assistant", "content": answer}
        )
        # Trim to last 10 messages (5 user + 5 assistant)
        if len(st.session_state.chat_history) > 10:
            st.session_state.chat_history = st.session_state.chat_history[-10:]
