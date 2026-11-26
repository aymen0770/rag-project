import json
from pathlib import Path

import faiss
import pymupdf
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 900
OVERLAP = 150


def extract_pages(pdf_path):
    pages = []
    with pymupdf.open(pdf_path) as pdf:
        for number, page in enumerate(pdf, start=1):
            text = page.get_text("text").strip()
            if text:
                pages.append({"text": text, "source": Path(pdf_path).name, "page": number})
    return pages


def split_text(text, size, overlap):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start = end - overlap
    return chunks


def build_chunks(pages, size, overlap):
    chunks = []
    for page in pages:
        for i, text in enumerate(split_text(page["text"], size, overlap)):
            chunks.append({
                "text": text,
                "source": page["source"],
                "page": page["page"],
                "chunk_id": i,
            })
    return chunks


if __name__ == "__main__":
    pages = []
    for pdf_path in sorted(Path("data/documents").glob("*.pdf")):
        pages.extend(extract_pages(pdf_path))
    print("Pages:", len(pages))

    chunks = build_chunks(pages, CHUNK_SIZE, OVERLAP)
    print("Chunks:", len(chunks))

    model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = model.encode(
        [c["text"] for c in chunks],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype("float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, "rag_index.faiss")

    with open("chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print("Indexed vectors:", index.ntotal)
