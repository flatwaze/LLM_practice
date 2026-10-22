from dataclasses import dataclass
from .retrieval import ExactRetriever, SearchResult
from .prompt_build import build_prompt
from .llm_client import LLMClient

import re

@dataclass
class RAGResponse:
    answer: str
    sources: list[SearchResult]
    retrieved_chunks: list[SearchResult]
    invalid_citations: list[str]


class RAG:
    def __init__(self, retriever: ExactRetriever, llm_client: LLMClient):
        self.retriever = retriever
        self.llm_client = llm_client

    def answer(self, q: str, top_k: int = 3, search_mode: str = 'chunks') -> RAGResponse:
        if not q.strip():
            raise ValueError('Question must not be empty.')
        if top_k <= 0:
            raise ValueError('top_k must be positive.')

        search_mode = search_mode.lower()
        if search_mode != 'chunks' and search_mode != 'documents':
            raise ValueError(f'search_mode must be "chunks" or "documents", now it is {search_mode}.')

        if search_mode == 'chunks':
            search_results = self.retriever.search_chunks(q, top_k)
        else:
            search_results = self.retriever.search_documents(q, top_k)

        if not search_results:
            return RAGResponse('Недостаточно информации: источники не найдены.', [], [], [])

        prompt = build_prompt(q, search_results)

        answer = self.llm_client.generate(prompt)

        #extract citations
        valid_sources = {
            (search_item.document_id, search_item.chunk_id): search_item
            for search_item in search_results
            }

        citations = re.findall(r"\[([^\[\]\s:]+):(\d+)\]", answer)
        invalid_citations = []
        sources = []
        seen = set()
        
        for document_id, chunk_id in citations:
            cit = (document_id, int(chunk_id))
            if cit in seen:
                continue
            seen.add(cit)

            if cit in valid_sources:
                sources.append(valid_sources[cit])
            else:
                invalid_citations.append(f'{document_id}:{chunk_id}')

        response = RAGResponse(answer, sources, search_results, invalid_citations)

        return response
