# ingest_docs.py
"""
Ingest docs into FAISS.
- Downloads a few WHO guideline PDFs (if reachable).
- Extracts text from PDFs or .txt files in medical_sources/.
- Chunks text, creates embeddings via sentence-transformers, and saves a FAISS index and metadata.
"""

import os
from pathlib import Path
import json
import re
import requests
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

# For PDF reading (fallback to PyPDF2 if pypdfium2 unavailable)
try:
    from pypdfium2 import PdfDocument
    def read_pdf_text(path):
        text_pages = []
        pdf = PdfDocument(str(path))
        for p in range(len(pdf)):
            page = pdf.get_page(p)
            text_pages.append(page.get_textpage().get_text_range())
            page.close()
        pdf.close()
        return "\n".join(text_pages)
except Exception:
    # fallback
    import PyPDF2
    def read_pdf_text(path):
        text = []
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for p in reader.pages:
                text.append(p.extract_text() or "")
        return "\n".join(text)

# ----------------- Config -----------------
DATA_DIR = Path("medical_sources")
INDEX_DIR = Path("vector_index")
EMB_MODEL = os.getenv("EMB_MODEL", "sentence-transformers/all-mpnet-base-v2")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 700))       # chars per chunk
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 150)) # overlap
INDEX_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

# Sample WHO PDFs to attempt to download (replace/add your own links)
WHO_PDF_URLS = [
    # Example WHO docs (these are example links; replace if any are unavailable)
    "https://www.minams.edu.pk/cPanel/ebooks/miscellaneous/Common%20Symptom%20Answer%20Guide.pdf" # NOTE: sometimes redirects or HTML pages
    # Add direct PDF links you trust. If a link is not a direct PDF, download will skip it gracefully.
]

# Utility: clean whitespace
def clean_text(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def download_sample_pdfs():
    print("Attempting to download sample PDFs (if any configured)...")
    for i, url in enumerate(WHO_PDF_URLS):
        try:
            r = requests.get(url, timeout=20, allow_redirects=True)
            content_type = r.headers.get("content-type", "")
            if "pdf" in content_type.lower() or url.lower().endswith(".pdf"):
                outp = DATA_DIR / f"who_sample_{i}.pdf"
                outp.write_bytes(r.content)
                print("Downloaded PDF to", outp)
            else:
                # If it's not a direct PDF, skip (you can manually download PDFs and place them in medical_sources/)
                print(f"URL does not look like a PDF (content-type={content_type}). Skipping: {url}")
        except Exception as e:
            print("Failed to download", url, ":", e)

def collect_files():
    files = []
    for p in DATA_DIR.iterdir():
        if p.is_file() and p.suffix.lower() in {".pdf", ".txt"}:
            files.append(p)
    return files

# Add to ingest_docs.py (replace the simple chunk_text function)

def chunk_text_advanced(text, chunk_size=700, overlap=150):
    """Better chunking that respects sentence boundaries"""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        sentence_length = len(sentence)
        
        if current_length + sentence_length > chunk_size and current_chunk:
            # Save current chunk
            chunk_text = ' '.join(current_chunk)
            chunks.append(chunk_text)
            
            # Start new chunk with overlap
            overlap_sentences = current_chunk[-2:] if len(current_chunk) > 2 else current_chunk
            current_chunk = overlap_sentences.copy()
            current_length = sum(len(s) for s in overlap_sentences) + len(overlap_sentences) - 1
        else:
            current_chunk.append(sentence)
            current_length += sentence_length + 1  # +1 for space

    # Add the last chunk
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def ingest_all():
    # 1) Optionally download sample PDFs
    download_sample_pdfs()

    files = collect_files()
    if not files:
        print("No files found in medical_sources/. Add .pdf or .txt files and re-run.")
        return

    print(f"Found {len(files)} source files. Loading and chunking...")
    docs = []
    for f in files:
        try:
            if f.suffix.lower() == ".pdf":
                txt = read_pdf_text(f)
            else:
                txt = f.read_text(encoding="utf-8", errors="ignore")
            txt = clean_text(txt)
            if len(txt) < 50:
                print("Skipping short file:", f)
                continue
            chunks = chunk_text_advanced(txt)
            for i, c in enumerate(chunks):
                docs.append({"source": f.name, "chunk_id": f"{f.name}__{i}", "text": c})
        except Exception as e:
            print("Failed to process", f, ":", e)

    if not docs:
        print("No document chunks produced. Check source files.")
        return

    # 2) Embed with sentence-transformers
    print("Loading embedding model:", EMB_MODEL)
    embedder = SentenceTransformer(EMB_MODEL)
    texts = [d["text"] for d in docs]
    print(f"Computing embeddings for {len(texts)} chunks...")
    embeddings = embedder.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    dim = embeddings.shape[1]
    print("Embedding dimension:", dim)

    # 3) Save FAISS index
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings, dtype="float32"))
    faiss.write_index(index, str(INDEX_DIR / "faiss.index"))
    print("Saved FAISS index to", INDEX_DIR / "faiss.index")

    # 4) Save metadata
    meta = [{"source": d["source"], "chunk_id": d["chunk_id"], "text": d["text"]} for d in docs]
    with open(INDEX_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print("Saved metadata to", INDEX_DIR / "metadata.json")
    print("Ingestion complete. Chunks:", len(meta))

if __name__ == "__main__":
    ingest_all()
