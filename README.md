# RAG System for PDF Question Answering

**Project title:** RAG System for PDF Question Answering

**Student name:** Mokhtari Aymen Az Islem

## Description

This project is a Retrieval-Augmented Generation (RAG) system. It reads PDF files, splits them into chunks, converts the chunks into embeddings, stores them in a FAISS index, and answers questions using the most relevant chunks. Each answer comes with the source file and page number.

The document used in this project is a blockchain technology assignment (`data/documents/BLOCKCHAIN_ASSIGNMENT_MOKHTARI.pdf`).

## Architecture

```
PDF -> text extraction -> chunks -> embeddings -> FAISS index

Question -> embedding -> top-k chunks -> prompt -> language model -> answer + sources
```

| Component | Tool |
|---|---|
| PDF extraction | PyMuPDF |
| Chunking | 900 characters, 150 overlap |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Vector search | FAISS IndexFlatIP |
| Generation | Qwen/Qwen2.5-1.5B-Instruct |

## Files

- `build_index.py`: builds `rag_index.faiss` and `chunks.json`
- `ask.py`: answers questions
- `evaluate.py`: computes Precision@k, Recall@k and MRR
- `eval_questions.json`: evaluation questions
- `requirements.txt`: Python packages

## Installation

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

```
python build_index.py
python ask.py "What is a hash pointer?"
python evaluate.py
python evaluate.py generate
```

## Evaluation

The evaluation uses 10 questions: 9 answerable questions and 1 question that cannot be answered from the document. A chunk is relevant when it contains the expected text of the answer.

Configurations:

| Config | Chunk size | Overlap | top_k |
|---|---|---|---|
| Baseline | 900 | 150 | 4 |
| A | 600 | 100 | 4 |
| B | 900 | 150 | 6 |
| C | 1200 | 200 | 4 |

### Results

The evaluation used 9 answerable questions (the unanswerable question is not scored in the retrieval metrics).

| Config | Chunk size | Overlap | top_k | Chunks | Precision@k | Recall@k | MRR | Retrieval (ms) |
|---|---|---|---|---|---|---|---|---|
| Baseline | 900 | 150 | 4 | 89 | 0.278 | 1.0 | 1.0 | 14.8 |
| A | 600 | 100 | 4 | 122 | 0.306 | 1.0 | 0.889 | 6.4 |
| B | 900 | 150 | 6 | 89 | 0.185 | 1.0 | 1.0 | 6.6 |
| C | 1200 | 200 | 4 | 69 | 0.25 | 1.0 | 0.944 | 6.7 |

### Discussion

All four configurations reached Recall@k = 1.0, so the relevant chunk was always found in the top k results. The differences are in precision and ranking.

The baseline (900/150, top_k = 4) gave the best ranking: MRR = 1.0, which means the relevant chunk was always the first result. Configuration A (600/100) had the highest precision (0.306) but a lower MRR (0.889), because for two questions the first relevant chunk was ranked second. Configuration C (1200/200) had MRR = 0.944 and precision 0.25. Configuration B (top_k = 6) kept the same recall but its precision dropped to 0.185, because the two extra passages were mostly irrelevant. A larger top_k did not help here.

Precision is low in general because most questions have only one or two relevant chunks, so Precision@4 cannot be much higher than 0.25 to 0.3. The baseline is the best overall choice because it has perfect recall and MRR with a good precision.

Limits of this evaluation: it uses only 9 questions, and relevance was labelled with keywords, so one question changes the results noticeably. The first retrieval time (14.8 ms) includes model warm-up; later values are about 6 to 7 ms.

### Answer quality

I read the 10 answers generated with the baseline configuration and checked them against the document.

- Correct and supported by the sources (5): hash pointer, 10-node PBFT, minimum number of nodes (13), Checks-Effects-Interactions pattern, sensor measurements on a public blockchain.
- Mostly correct but with mistakes (4):
  - Recalculating hashes: the answer says an attacker can change transactions without changing the hashes, and says Proof-of-Stake systems need a longer chain with more accumulated work. Both are wrong; accumulated work belongs to Proof of Work.
  - Hashing vs encryption: the answer calls the hash output "ciphertext" and does not say that hashing is one-way and has no key.
  - Medical records: the answer adds a statement about keys held by "unauthorized individuals like doctors", which is not in the document.
  - Blockchain trilemma: the answer says more computational power (block size) improves security, which is not what the document says. The document says a larger block size increases throughput but makes it harder to run full nodes.
- Unanswerable question (1): the system correctly answered that there is not enough information.

The model did not cite source numbers inside the answers. The sources are printed separately by `ask.py` with file name and page number. The small 1.5B model can add statements that are not in the retrieved text, so answers must be verified against the cited pages.

## Limitations

- The retrieved passages may be similar to the question without answering it.
- The document may not contain the answer.
- A small language model can misunderstand the context.
- Broad questions such as "What is this document about?" retrieve poor passages.
- Tables, equations and scanned pages can be extracted badly.
- Answers must be verified against the cited pages.
