# Retrieval Engine

Exact semantic search over text documents using NumPy and Sentence Transformers.

## Setup

Use Python 3.14. From the project directory in Git Bash:

```bash
python -m venv .venv
source .venv/Scripts/activate
python -m pip install -r requirements.txt
```

If `.venv` already exists, start with the activation command.

## Run

```bash
python main.py
```

Enter a query to receive the three best matching documents with their representative
chunks and similarity scores. The first run downloads the embedding model from
Hugging Face; later runs reuse the local cache. Computation runs on CPU.

Edit `CHUNK_SIZE`, `OVERLAP` and `TOP_K` at the top of `main.py` to change the
settings. Chunk size and overlap are measured in words. The model truncates input
to its token limit, which is distinct from the word limit used for chunking.

## Tests

```bash
python -m pytest -q
python -m pip check
```

Unit tests use fixed vectors and do not download the model.

## Corpus

`data/documents/` contains 30 English documents assembled from official Python,
SciPy, NumPy, pandas and scikit-learn documentation. Excerpts retain the original
text and code; HTML navigation is removed. Some documents combine related
sections, with every source listed in their headers. Each filename without its
extension is the document ID.

Sources and checksums are listed in `data/corpus_sources.json`. Python license
terms are in `data/SOURCE_LICENSE.txt`; additional BSD licenses are in
`data/licenses/`. Five documents from the original corpus are preserved in
`data/archive/` and are not indexed by `main.py`.

## Evaluation

`data/eval_questions.json` contains 10 queries with manually selected
`relevant_document_ids`, alongside the user's original wording. The corpus was
revised to cover these topics before running the evaluation. This is a small,
curated benchmark, not a measure of general Python question coverage.

Question 1 concerns leading and trailing whitespace, not whitespace inside a
string. Question 7 concerns concurrent I/O using async/await. Question 9 concerns
aliasing a reused mutable list; a shallow copy separates row lists but does not
recursively copy nested mutable objects. Non-public attributes in question 4 are
an encapsulation convention, not a security boundary.

Some queries have multiple relevant documents. For each query, Recall@3 is the
number of relevant documents retrieved in the top three divided by the total
number of labeled relevant documents. Average these values across the 10 queries.

Run the comparison from the project directory:

```bash
python -m experiments.eval
```

The model is loaded once, and a separate index is built for each configuration.
Per-query retrieved document IDs and Recall@3 values are saved in
`experiments/results.json`.

| Configuration | Chunk size (words) | Overlap (words) | Mean Recall@3 |
|---|---:|---:|---:|
| A | 100 | 20 | 0.85 |
| B | 250 | 50 | 0.90 |

The difference comes entirely from question q09 about reused lists: configuration
B retrieves both labeled documents, while A retrieves one of the two. Other
per-query recall values are unchanged. Questions q01 and q10 retrieve only one
of their two labeled relevant documents under both configurations.

One hypothesis is that changing chunk boundaries places related explanations of
list assignment and copying together, or closer to the beginning of a chunk.
This has not been isolated experimentally. The model accepts at most 128 tokens,
so a 250-word chunk is not necessarily encoded in full; the improvement cannot
simply be attributed to retaining all of the larger chunk's context. Ten curated
queries are insufficient to conclude that configuration B is generally better.
