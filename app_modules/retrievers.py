"""
This module contains the Retriever(s) that could be used during the Multi-Step RAG's execution.
"""
import os
from pinecone.pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
load_dotenv()

embedding_model = HuggingFaceEmbeddings(model_name=os.environ["EMBEDDING_MODEL_NAME"])


def pinecone_mmr_retriever():
    pass

