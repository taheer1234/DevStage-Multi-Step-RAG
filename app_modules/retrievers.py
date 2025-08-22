"""
This module contains the Retriever(s) that could be used during the Multi-Step RAG's execution.
"""

import os
from pinecone.pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

print("Initializing Embedding Model...")
embedding_model = HuggingFaceEmbeddings(model_name=os.environ["EMBEDDING_MODEL_NAME"])
print("Initialized Embedding Model!")

pc = Pinecone()


def pinecone_mmr_retriever(index_name: str, k: int, lambda_mult: float):
    """
    This function returns a retriver that uses the
    Pinecone Vector Store and the mmr search type for searching the Vector Store.
    """
    index = pc.Index(index_name)
    vector_store = PineconeVectorStore(index=index, embedding=embedding_model)
    return vector_store.as_retriever(
        search_type="mmr", search_kwargs={"k": k, "lambda_mult": lambda_mult}
    )


def pinecone_similarity_retriever(index_name: str, k: int):
    """
    This function returns a retriver that uses the
    Pinecone Vector Store and the similarity search type for searching the Vector Store.
    """
    index = pc.Index(index_name)
    vector_store = PineconeVectorStore(index=index, embedding=embedding_model)
    return vector_store.as_retriever(search_type="similarity", search_kwargs={"k": k})
