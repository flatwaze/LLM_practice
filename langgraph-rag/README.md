# LangGraph RAG Agent

Built with help from an LLM assistant, including implementation, debugging, and this README.

A LangChain/LangGraph version of the agent in [retrieval-engine](../retrieval-engine).
It reads the same corpus without copying it. Gemma (`google/gemma-4-31b-it`)
runs through OpenRouter; local `paraphrase-multilingual-MiniLM-L12-v2`
embeddings feed an `InMemoryVectorStore`.

```text
START → model → tool calls → search_docs / get_document → model
            └→ final answer or call limit → END
```

`search_docs(q, top_k)` returns chunks; `get_document(document_id)` returns
a full source. Messages accumulate in graph state. Each question starts a
new history, with a budget of five model calls. Chunks use 200-character
windows with a target overlap of 30 characters and `[document_id:chunk_id]` citations.

## Saved run

[Results](experiments/agent_evaluation_20261030_154333_038449.json):
16 questions, 40 model calls, 21 searches, 3 full-document reads;
no recorded errors or call-limit stops.

**Document citation recall: 93.3% (14/15).** Across the 13 answerable questions,
count each expected document once per question and check whether its ID appears
in a numeric chunk citation in the final answer. Grouped citations are included;
the three unanswerable questions are excluded. Macro recall is 92.3%.

`rag08` has no citations. Some other references have malformed formatting.
This metric checks document IDs, not chunk validity, factual accuracy, or whether
the cited text supports the claim. This is one run on the existing development set.

## Run

From the repository root, using Git Bash on Windows:

```bash
cd langgraph-rag
source .venv/Scripts/activate
python -m pip install -r requirements.txt
export OPENROUTER_API_KEY="your-key"
python main.py
```

Create the environment first with `python -m venv .venv` if needed.
Each run writes a new JSON under `experiments/`, saving answers, errors,
model-call counts, and message traces after each question.
