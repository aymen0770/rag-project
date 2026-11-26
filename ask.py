import json
import sys

import faiss
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
TOP_K = 4

embedder = SentenceTransformer(EMBEDDING_MODEL)
index = faiss.read_index("rag_index.faiss")
with open("chunks.json", encoding="utf-8") as f:
    chunks = json.load(f)

tokenizer = AutoTokenizer.from_pretrained(LLM_NAME)
model = AutoModelForCausalLM.from_pretrained(LLM_NAME, torch_dtype="auto", device_map="auto")


def retrieve(question, top_k=TOP_K):
    vector = embedder.encode(
        [question], convert_to_numpy=True, normalize_embeddings=True
    ).astype("float32")
    scores, positions = index.search(vector, top_k)
    results = []
    for score, pos in zip(scores[0], positions[0]):
        if pos == -1:
            continue
        item = dict(chunks[pos])
        item["score"] = float(score)
        results.append(item)
    return results


def generate(question, retrieved):
    parts = []
    for i, item in enumerate(retrieved, start=1):
        parts.append(f"[{i}] Source: {item['source']}, page {item['page']}\n{item['text']}")
    context = "\n\n".join(parts)

    messages = [
        {"role": "system", "content":
            "Answer using only the supplied context. If the context does not "
            "contain the answer, say there is not enough information. "
            "Refer to source numbers when possible."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=300, do_sample=False)
    new_tokens = output[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def ask(question):
    retrieved = retrieve(question)
    answer = generate(question, retrieved)
    print("\nQuestion:", question)
    print("\nAnswer:\n" + answer)
    print("\nSources:")
    for i, item in enumerate(retrieved, start=1):
        print(f"[{i}] {item['source']}, page {item['page']} (score {item['score']:.3f})")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        ask(" ".join(sys.argv[1:]))
    else:
        while True:
            question = input("\nQuestion (empty to quit): ").strip()
            if not question:
                break
            ask(question)
