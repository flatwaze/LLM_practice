from pathlib import Path
import requests
import json

from src.embeddings import EmbeddingModel
from src.retrieval import ExactRetriever
from src.llm_client import LLMClient
from src.agent import SimpleAgent
from src.tools import Tools


CHUNK_SIZE = 100
OVERLAP = 20
MAX_STEPS = 5
LLM_MODEL = "google/gemma-4-31b-it"


def main():
    #reading documents
    docs_dir = Path(__file__).resolve().parent / "data" / "documents"
    documents = []

    for path in sorted(docs_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in {".txt", ".md"}:
            continue
        text = path.read_text(encoding="utf-8")

        documents.append({
            "document_id": path.stem,
            "text": text,
        })

    #reading questions
    q_path = Path(__file__).resolve().parent / "data" / "rag_eval_questions.json"
    with q_path.open(encoding='utf-8') as file:
        questions = json.load(file)

    #Creating client
    try:
        llm_client = LLMClient(
            model=LLM_MODEL,
            api_key_env="OPENROUTER_API_KEY",
            max_tokens=1000,
        )
    except ValueError as exc:
        print(f"Error: {exc}")
        return

    #Eval.
    results = []
    results_path = Path(__file__).resolve().parent / "experiments" / "agent_evaluation.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)

    with llm_client.session:
        #Creating agent
        model = EmbeddingModel()
        retriever = ExactRetriever(model, documents, chunk_size=CHUNK_SIZE, overlap=OVERLAP)
        tools = Tools(retriever, documents)
        agent = SimpleAgent(llm_client, tools, max_steps=MAX_STEPS)

        for q in questions:
            print(f"\n{q['query_id']}: {q['query']}")
            result = {**q, "answer": None, "error": None}

            try:
                result["answer"] = agent.run(q["query"])
            except (requests.RequestException, RuntimeError, ValueError) as exc:
                result["error"] = str(exc)
                print(f"Error: {exc}")
            else:
                print(f"\nAnswer:\n{result['answer']}")

            #Saving answer, error and agent trace after each question
            result["trace"] = agent.history.copy()
            results.append(result)
            with results_path.open("w", encoding="utf-8") as file:
                json.dump(results, file, ensure_ascii=False, indent=2)

            print('-----------------')


if __name__ == "__main__":
    main()
