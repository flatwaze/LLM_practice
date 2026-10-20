import json
from pathlib import Path

from src.embeddings import EmbeddingModel
from src.retrieval import ExactRetriever


def main():
    main_dir = Path(__file__).resolve().parent.parent
    docs_dir = main_dir / "data" / "documents"
    q_dir = main_dir / "data" / "eval_questions.json"

    documents = []

    for path in sorted(docs_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in {".txt", ".md"}:
            continue
        text = path.read_text(encoding="utf-8")
        documents.append({
            "document_id": path.stem,
            "text": text,
        })

    queries = json.loads(q_dir.read_text(encoding="utf-8"))
    model = EmbeddingModel()
    results = []

    for chunk_size, overlap in [(100, 20), (250, 50)]:
        recall, query_results = compute_recall(model, queries, documents, chunk_size, overlap)
        results.append({
            "chunk_size": chunk_size,
            "overlap": overlap,
            "recall_at_3": recall,
            "queries": query_results,
        })
        print(f"chunk_size={chunk_size}, overlap={overlap}: Recall@3={recall:.3f}")

    results_path = Path(__file__).resolve().parent / "results.json"
    results_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


def compute_recall(model, queries, documents, chunk_size, overlap):
    retriever = ExactRetriever(model, documents, chunk_size=chunk_size, overlap=overlap)
    query_results = []

    for q in queries:
        results = retriever.search_documents(q["query"], top_k=3)
        retrieved = [result.document_id for result in results]
        relevant = set(q["relevant_document_ids"])
        recall = len(relevant.intersection(retrieved)) / len(relevant)
        query_results.append({
            "query_id": q["query_id"],
            "retrieved_document_ids": retrieved,
            "recall_at_3": recall,
        })

    recall = sum(result["recall_at_3"] for result in query_results) / len(query_results)
    return recall, query_results


if __name__ == "__main__":
    main()
