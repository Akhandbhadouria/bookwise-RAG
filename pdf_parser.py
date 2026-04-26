"""
pdf_parser.py — PDF Reading & Text Chunking

This module handles two responsibilities:
1. Extracting raw text from an uploaded PDF file, page by page.
2. Splitting that text into smaller, overlapping chunks suitable
   for embedding and semantic search.

Why PyMuPDF (fitz)?
  - It's fast, lightweight, and handles most PDF layouts well.
  - It gives us page-level access so we can attach page numbers
    as metadata to every chunk.
"""

import fitz  # PyMuPDF
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def extract_text_from_pdf(pdf_file) -> list[Document]:
    """
    Read an uploaded PDF and extract text page-by-page.

    Args:
        pdf_file: A file-like object (from Streamlit's file_uploader).

    Returns:
        A list of LangChain Document objects. Each Document contains:
          - page_content : the full text of one PDF page
          - metadata     : {"page": <1-based page number>}

    Why return Documents (not plain strings)?
        Because LangChain's text splitter can propagate metadata
        from the parent Document to every child chunk. This lets
        us trace every chunk back to its source page.
    """

    # Read the raw bytes from the uploaded file
    pdf_bytes = pdf_file.read()

    # Open the PDF from memory (no temp file needed)
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    documents = []

    for page_number in range(len(doc)):
        page = doc[page_number]
        text = page.get_text()  # plain-text extraction

        # Skip blank pages (e.g. separator pages in some books)
        if text.strip():
            documents.append(
                Document(
                    page_content=text,
                    metadata={"page": page_number + 1},  # 1-based
                )
            )

    doc.close()
    return documents


def chunk_documents(
    documents: list[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[Document]:
    """
    Split page-level Documents into smaller, overlapping chunks.

    Args:
        documents   : The list of page-level Documents from extract_text_from_pdf.
        chunk_size  : Maximum number of characters per chunk.
        chunk_overlap: Number of characters shared between consecutive chunks.

    Returns:
        A list of smaller Document objects, each still carrying the
        original page number in its metadata.

    Why overlap?
        If a sentence sits right at a chunk boundary, overlap ensures
        the full sentence appears in at least one chunk. Without it,
        the retriever might miss relevant context split across two chunks.

    Why RecursiveCharacterTextSplitter?
        It tries to split on natural boundaries (paragraphs → sentences
        → words) before falling back to raw character counts. This
        produces more coherent chunks than a naive fixed-window split.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,        # count characters, not tokens
        separators=["\n\n", "\n", ". ", " ", ""],  # split hierarchy
    )

    # split_documents preserves and copies metadata to child chunks
    chunks = splitter.split_documents(documents)

    return chunks
