from pathlib import Path
from dataclasses import asdict
import requests
import json

from src.embeddings import EmbeddingModel
from src.retrieval import ExactRetriever
from src.llm_client import LLMClient
from src.rag import RAG


CHUNK_SIZE = 100
OVERLAP = 20
TOP_K = 3
SEARCH_MODE = "chunks"
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

    #Creating RAGPipeline
    try:
        llm_client = LLMClient(model=LLM_MODEL, api_key_env="OPENROUTER_API_KEY")
    except ValueError as exc:
        print(f"Error: {exc}")
        return

    model = EmbeddingModel()
    retriever = ExactRetriever(model, documents, chunk_size=CHUNK_SIZE, overlap=OVERLAP)
    rag = RAG(retriever, llm_client)

    #Eval.
    results = []
    results_path = Path(__file__).resolve().parent / "experiments" / "rag_evaluation.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)

    with llm_client.session:
        for q in questions:
            print(f"\n{q['query_id']}: {q['query']}")
            try:
                response = rag.answer(q["query"], top_k=TOP_K, search_mode=SEARCH_MODE)
            except (requests.RequestException, RuntimeError, ValueError) as exc:
                print(f"Error: {exc}")
                results.append({
                    **q,
                    "rag": None,
                    "error": str(exc)
                })
            else:

                
                print(f"\nAnswer:\n{response.answer}")
                print("\nSources:")
    
                for source in response.sources:
                    print(f"[{source.document_id}:{source.chunk_id}]")

                if response.invalid_citations:
                    print("\nUnknown citations:", ", ".join(response.invalid_citations))

                #log.
                results.append({
                    **q,
                    "rag": asdict(response)
                })
                

                print('-----------------')


            with results_path.open("w", encoding="utf-8") as file:
                json.dump(results, file, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
