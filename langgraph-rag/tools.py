from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever

class Tools:
    def __init__(self, retriever: VectorStoreRetriever, documents: list[Document]):
        self.retriever = retriever
        self.docs = {
            doc.metadata["document_id"]: doc
            for doc in documents
        }

    def search_docs(self, q: str, top_k: int = 3) -> str:
        """Search for relevant chunks among all avaliable documents."""
        
        chunks = self.retriever.vectorstore.similarity_search(q, k=top_k)
        
        res = []
        for chunk in chunks:
            res.append(
                f"[{chunk.metadata['document_id']}:{chunk.metadata['chunk_id']}]\n"
                f"Document ID: {chunk.metadata['document_id']}\n"
                f"Source: {chunk.metadata['source']}\n"
                f"Content: {chunk.page_content}"
            )

        return '\n\n'.join(res)

    def get_document(self, document_id: str) -> str:
        """Retrieve a complete document by its document ID."""

        document = self.docs.get(document_id)

        if document is None:
            return f"Invalid document ID: {document_id}."

        return (
            f"Document ID: {document_id}\n"
            f"Source: {document.metadata['source']}\n"
            f"Content: {document.page_content}"
        )

    
