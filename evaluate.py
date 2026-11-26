import json
import re
import sys
import time

import faiss
from sentence_transformers import SentenceTransformer

from build_index import EMBEDDING_MODEL, build_chunks, extract_pages

CONFIGS = [
    {"name": "Baseline", "chunk_size": 900, "overlap": 150, "top_k": 4},
    {"name": "A", "chunk_size": 600, "overlap": 100, "top_k": 4},
    {"name": "B", "chunk_size": 900, "overlap": 150, "top_k": 6},
    {"name": "C", "chunk_size": 1200, "overlap": 200, "top_k": 4},
]


def precision_at_k(retrieved_ids, relevant_ids, k):
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    return sum(x in relevant_ids for x in top_k) / len(top_k)


def recall_at_k(retrieved_ids, relevant_ids, k):
    if not relevant_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    return sum(x in relevant_ids for x in top_k) / len(relevant_ids)


def reciprocal_rank(retrieved_ids, relevant_ids):
    for rank, x in enumerate(retrieved_ids, start=1):
        if x in relevant_ids:
            return 1.0 / rank
    return 0.0


def normalize(text):
    return re.sub(r"\s+", " ", text.lower()).strip()


def relevant_positions(chunks, question):
    keywords = [normalize(k) for k in question["keywords"]]
    return {
        i for i, c in enumerate(chunks)
        if any(k in normalize(c["text"]) for k in keywords)
    }


def search(model, index, chunks, question, top_k):
    vector = model.encode(
        [question], convert_to_numpy=True, normalize_embeddings=True
    ).astype("float32")
    scores, positions = index.search(vector, top_k)
    return [(int(p), float(s)) for s, p in zip(scores[0], positions[0]) if p != -1]


def build(model, pages, config):
    chunks = build_chunks(pages, config["chunk_size"], config["overlap"])
    embeddings = model.encode(
        [c["text"] for c in chunks], convert_to_numpy=True, normalize_embeddings=True
    ).astype("float32")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return chunks, index


def mean(values):
    return sum(values) / len(values) if values else 0.0


def run(model, pages, questions, config):
    chunks, index = build(model, pages, config)
    k = config["top_k"]
    precisions, recalls, rrs, times = [], [], [], []

    for q in questions:
        if not q["answerable"]:
            continue
        relevant = relevant_positions(chunks, q)
        if not relevant:
            print("No matching chunk for:", q["question"])
            continue
        start = time.time()
        results = search(model, index, chunks, q["question"], k)
        times.append(time.time() - start)
        ids = [p for p, _ in results]
        precisions.append(precision_at_k(ids, relevant, k))
        recalls.append(recall_at_k(ids, relevant, k))
        rrs.append(reciprocal_rank(ids, relevant))

    return {
        "config": config["name"],
        "chunk_size": config["chunk_size"],
        "overlap": config["overlap"],
        "top_k": k,
        "chunks": len(chunks),
        "precision": round(mean(precisions), 3),
        "recall": round(mean(recalls), 3),
        "mrr": round(mean(rrs), 3),
        "retrieval_ms": round(1000 * mean(times), 1),
    }


def generate_answers(model, pages, questions):
    from ask import generate

    config = CONFIGS[0]
    chunks, index = build(model, pages, config)
    records = []
    for q in questions:
        results = search(model, index, chunks, q["question"], config["top_k"])
        retrieved = []
        for pos, score in results:
            item = dict(chunks[pos])
            item["score"] = score
            retrieved.append(item)
        answer = generate(q["question"], retrieved)
        records.append({
            "question": q["question"],
            "answer": answer,
            "sources": [f"{r['source']} p.{r['page']} ({r['score']:.3f})" for r in retrieved],
        })
        print("\nQ:", q["question"])
        print("A:", answer)
    with open("evaluation_answers.json", "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    with open("eval_questions.json", encoding="utf-8") as f:
        questions = json.load(f)

    pages = []
    from pathlib import Path
    for pdf_path in sorted(Path("data/documents").glob("*.pdf")):
        pages.extend(extract_pages(pdf_path))

    model = SentenceTransformer(EMBEDDING_MODEL)

    results = [run(model, pages, questions, c) for c in CONFIGS]

    lines = [
        "| Config | Chunk size | Overlap | top_k | Chunks | Precision@k | Recall@k | MRR | Retrieval (ms) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r['config']} | {r['chunk_size']} | {r['overlap']} | {r['top_k']} | {r['chunks']} "
            f"| {r['precision']} | {r['recall']} | {r['mrr']} | {r['retrieval_ms']} |"
        )
    table = "\n".join(lines)
    print("\n" + table)

    with open("evaluation_results.md", "w", encoding="utf-8") as f:
        f.write(table + "\n")

    if "generate" in sys.argv:
        generate_answers(model, pages, questions)
