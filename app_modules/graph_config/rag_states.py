"""This module contains the states for the Multi-Step RAG's LangGraph."""

from typing import TypedDict, Annotated, List
import operator
from langchain_core.messages import BaseMessage
from langchain_core.documents import Document
from langgraph.graph import add_messages


class RAGStates(TypedDict):
    """The states corresponding to the Multi-Step RAG's LangGraph."""

    query: str
    rewritten_query: str
    ontopic_classification: bool
    documents: List[Document]
    chat_history: Annotated[List[BaseMessage], add_messages]
    retriever_invoke_number: Annotated[int, operator.add]
    response: str
