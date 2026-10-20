from dataclasses import dataclass
from .chunking import chunk_text
import numpy as np


@dataclass
class SearchResult:
    document_id: str
    chunk_id: int
    text: str
    score: float


class ExactRetriever:
    def __init__(self, embed_model, documents=None, chunk_size=50, overlap=20):
        self.model = embed_model
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.index(documents if documents is not None else [])

    def index(self, documents: list[dict]) -> None:
        chunk_text('', self.chunk_size, self.overlap)
        chunks_index = []
        chunks_all = []
        chunks_by_doc_id = {}

        for document in documents:
            doc_id = document['document_id'] if 'document_id' in document else document['id']
            text = document['text']
            if doc_id in chunks_by_doc_id:
                raise ValueError('Document IDs must be unique.')

            chunks = chunk_text(text, self.chunk_size, self.overlap)
            start = len(chunks_all)
            chunks_by_doc_id[doc_id] = np.arange(start, start + len(chunks))
            chunks_all += chunks

            for i, chunk in enumerate(chunks):
                chunks_index.append({'document_id': doc_id, 'chunk_id': i, 'text': chunk})

        embeds = (
            self.model.encode(chunks_all)
            if chunks_all else np.empty((0, 0), dtype=np.float32)
        )
        self.chunks_index = chunks_index
        self.chunks_all = chunks_all
        self.chunks_by_doc_id = chunks_by_doc_id
        self.embeds = embeds

    def search_chunks(self, query: str, top_k: int = 5) -> list[SearchResult]:
        if top_k <= 0:
            raise ValueError('top_k must be positive.')
        if not self.chunks_all:
            return []

        q_embed = self.model.encode([query])[0]
        scores = self.embeds @ q_embed
        top_indxs = np.argsort(-scores, kind='stable')[:top_k]
        top_chunks = []

        for i in top_indxs:
            chunk = self.chunks_index[i]
            sr = SearchResult(chunk['document_id'], chunk['chunk_id'],
                              chunk['text'], float(scores[i]))
            top_chunks.append(sr)

        return top_chunks

    def search_documents(self, query: str, top_k: int = 5) -> list[SearchResult]:
        if top_k <= 0:
            raise ValueError('top_k must be positive.')
        if not self.chunks_all:
            return []

        q_embed = self.model.encode([query])[0]
        scores = self.embeds @ q_embed
        top_chunks = []

        for doc_range in self.chunks_by_doc_id.values():
            if len(doc_range):
                top_idx = doc_range[np.argmax(scores[doc_range])]
                top_chunks.append(top_idx)

        top_docs_idx = np.argsort(-scores[top_chunks], kind='stable')[:top_k]
        top_docs = []

        for i in top_docs_idx:
            idx = top_chunks[i]
            chunk = self.chunks_index[idx]
            sr = SearchResult(chunk['document_id'], chunk['chunk_id'],
                              chunk['text'], float(scores[idx]))
            top_docs.append(sr)

        return top_docs

    def search_docs(self, query: str, top_k: int = 5) -> list[SearchResult]:
        return self.search_documents(query, top_k)
