from pathlib import Path

from src.embeddings import EmbeddingModel
from src.retrieval import ExactRetriever

CHUNK_SIZE = 100
OVERLAP = 20
TOP_K = 3


def main():
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

    query = input("Query: ").strip()
    if not query:
        return

    model = EmbeddingModel()
    retriever = ExactRetriever(model, documents, chunk_size=CHUNK_SIZE, overlap=OVERLAP)
    results = retriever.search_documents(query, top_k=TOP_K)

    for result in results:
        print(f"\n{result.document_id} | chunk {result.chunk_id} | score {result.score:.4f}")
        print(result.text)


if __name__ == "__main__":
    main()
