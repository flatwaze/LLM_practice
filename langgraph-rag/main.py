import json
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.documents import Document
from langchain_openrouter import ChatOpenRouter

from graph import SimpleAgent
from lang_retriever import build_retriever
from tools import Tools


PROJECT_DIR = Path(__file__).resolve().parent
REPO_DIR = PROJECT_DIR.parent

CHUNK_SIZE = 200
OVERLAP = 30
MAX_CALLS = 5
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
LLM_MODEL = "google/gemma-4-31b-it"


def main():
    # Reading the same corpus as the original project.
    docs_dir = REPO_DIR / "retrieval-engine" / "data" / "documents"
    documents = []

    for path in sorted(docs_dir.iterdir()):
        documents.append(
            Document(
                page_content=path.read_text(encoding="utf-8"),
                metadata={
                    "document_id": path.stem,
                    "source": path.relative_to(REPO_DIR).as_posix(),
                },
            )
        )

    # Reading the same evaluation questions.
    questions_path = REPO_DIR / "retrieval-engine" / "data" / "rag_eval_questions.json"
    questions = json.loads(questions_path.read_text(encoding="utf-8"))

    # Creating the model and agent once for the entire run.
    model = ChatOpenRouter(
        model=LLM_MODEL,
        temperature=0,
        max_tokens=1000,
        timeout=60_000,
        max_retries=0,
    )

    retriever = build_retriever(
        documents,
        model_name=EMBEDDING_MODEL,
        chunk_size=CHUNK_SIZE,
        overlap=OVERLAP,
    )
    tools = Tools(retriever, documents)
    agent = SimpleAgent(model, tools, max_calls=MAX_CALLS)

    # A separate output file for each run; old results stay intact.
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    results_path = PROJECT_DIR / "experiments" / f"agent_evaluation_{run_id}.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results = []
    print(f"Results: {results_path}")

    for question in questions:
        print(f"\n{question['query_id']}: {question['query']}")
        result = {
            **question,
            "model": LLM_MODEL,
            "embedding_model": EMBEDDING_MODEL,
            "chunk_size": CHUNK_SIZE,
            "overlap": OVERLAP,
            "max_calls": MAX_CALLS,
            "answer": None,
            "error": None,
            "model_calls": None,
            "stop_reason": None,
            "trace": [],
        }

        try:
            state = agent.run(question["query"])
            last_message = state["messages"][-1]

            result["model_calls"] = state["model_calls"]
            result["trace"] = [
                message.model_dump(mode="json")
                for message in state["messages"]
            ]

            if last_message.tool_calls:
                result["answer"] = "Step limit reached without a final answer."
                result["stop_reason"] = "model_call_limit"
            else:
                result["answer"] = last_message.content
                result["stop_reason"] = "answer"

            print(f"\nAnswer:\n{result['answer']}")

        # Record failed questions and continue the evaluation batch.
        except Exception as error:
            result["error"] = f"{type(error).__name__}: {error}"
            result["stop_reason"] = "error"
            print(f"Error: {result['error']}")

        # Saving after each question, as in the original main.py.
        results.append(result)
        results_path.write_text(
            json.dumps(results, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print("-----------------")


if __name__ == "__main__":
    main()
