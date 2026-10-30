from langchain_core.documents import Document
from langchain_core.vectorstores import (
    InMemoryVectorStore,
    VectorStoreRetriever,
)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

def build_retriever(docs: list[Document], 
                    model_name: str,
                    chunk_size: int = 200,
                    overlap: int = 30,
                    k: int = 5 ) -> VectorStoreRetriever:

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                               chunk_overlap=overlap, 
                                               add_start_index=True
                                            )
    chunks = []
    for doc in docs:
        document_chunks = splitter.split_documents([doc])
        for chunk_id, chunk in enumerate(document_chunks):
            chunk.metadata["chunk_id"] = chunk_id
        chunks.extend(document_chunks)

    embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

    retriever = InMemoryVectorStore(
        embedding=embeddings
        )

    retriever.add_documents(chunks)

    retriever = retriever.as_retriever(search_type='similarity', 
                                       search_kwargs={'k' : k}
                                    )


    return retriever
