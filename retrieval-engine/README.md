# Minimal RAG & Agent

README generated with LLM assistance.

A learning project built in plain Python: retrieve passages from documentation,
use them to answer a question with citations. No LangChain, vector database
or agent framework.

The goal is to understand RAG and a minimal tool-calling agent loop.

[Pipeline](#pipeline) · [Experiment](#experiment) · [Agent](#agent) ·
[Result files](#result-files) · [Run](#run) · [Current limits](#current-limits)

## Pipeline

```text
Indexing:   documents -> overlapping chunks -> embeddings -> in-memory index
Answering:  question -> top-k search -> context + question -> LLM -> cited answer
```

| Component | Implementation |
|---|---|
| Chunking | Word windows with configurable overlap |
| Embeddings | `paraphrase-multilingual-MiniLM-L12-v2`, running on CPU |
| Retrieval | Exact cosine similarity (all vs one) |
| Prompt | Sources separated from instructions; answer from context or acknowledge missing information |
| Generation | Gemma 4 31B through OpenRouter |
| Citations | Match `[document_id:chunk_id]` to passages supplied to the model |

The default configuration uses 100-word chunks, 20-word overlap and top-3
retrieval. Search can select the best chunks overall or one best chunk per
document.

`RAGResponse` separates the answer, all retrieved chunks, the chunks actually
cited, and unknown citation.

## Experiment

The corpus contains 30 documentation excerpts covering Python and scientific
Python libraries. The test set has 16 questions: 6 factual, 4 explanation or
synthesis, 3 exact-detail and 3 without an answer in the corpus.

We compared RAG with the same LLM answering without retrieved context.
Manual review of the 13 questions covered by the corpus found:

| Outcome | RAG | Baseline |
|---|---:|---:|
| Main question answered correctly | 6 | 12 |
| Partial answer | 3 | 0 |
| Refusal | 4 | 0 |
| Incorrect answer | 0 | 1 |

Baseline was stronger on familiar Python questions. RAG often retrieved the
right document but missed the specific passage containing the answer. Under
its context-only instruction, the model then declined to answer.

RAG did help with one exact detail: baseline incorrectly denied the Windows
limit of 61 workers for `ProcessPoolExecutor`; RAG found it and answered correctly.
RAG declined all three questions without corpus answers.

Questions and references: [evaluation set](data/rag_eval_questions.json).
See the [error analysis](experiments/rag_analysis.md) for individual cases.

## Agent

`SimpleAgent` uses `search_docs(query, top_k)` to retrieve chunks and
`get_document(document_id)` to read full documents. It can refine searches;
tool errors return to the model for correction. Limit: 5 model calls per question.

On 16 questions: 42 model calls (2.63/question), 23 searches, 3 document reads;
no tool errors or limit stops. Opening the full JSON document recovered the
UTF-8 answer missed by chunk searches. Of 3 unanswerable questions, 2 received
correct refusals; 1 used outside knowledge. Citation grounding remains unchecked.

## Result files

| JSON | Contents |
|---|---|
| [agent_evaluation.json](experiments/agent_evaluation.json) | Agent answers, errors and full message/tool traces |
| [rag_evaluation.json](experiments/rag_evaluation.json) | Fixed RAG answers, retrieved chunks and citations |
| [baseline_evaluation.json](experiments/baseline_evaluation.json) | LLM answers without retrieval |
| [results.json](experiments/results.json) | Earlier retrieval-only experiments |

## Run

Python 3.14; commands below use Bash from the project directory.
Set `OPENROUTER_API_KEY` in your environment.

```bash
python -m venv .venv
source .venv/Scripts/activate
python -m pip install -r requirements.txt

python main.py            # Agent evaluation
python baseline_eval.py   # Baseline evaluation
python -m pytest -q       # Chunking and retrieval tests
```

## Current limits

- The encoder can truncate long chunks.
- The citation parser skips grouped references such as `[doc:0, doc:1]`.
- There is no automatic retry, resume or semantic verification of citations.

Earlier retrieval-only experiments remain in `experiments/eval.py` and
`experiments/results.json`. Corpus provenance is in `data/corpus_sources.json`;
license notices are in `data/SOURCE_LICENSE.txt` and `data/licenses/`.
