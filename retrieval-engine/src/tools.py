from .retrieval import ExactRetriever
from .retrieval import SearchResult

class Tools:
    def __init__(self, retriever: ExactRetriever, documents: list[dict]):
        self.retriever = retriever
        self.documents = documents

    def search_docs(self, query: str, top_k: int = 3) -> list[SearchResult]:
        return self.retriever.search_chunks(query, top_k) 

    def get_document(self, document_id) -> dict:
        for d in self.documents:
            if d['document_id'] == document_id:
                return d
        raise ValueError(f"Invalid document_id {document_id}")
