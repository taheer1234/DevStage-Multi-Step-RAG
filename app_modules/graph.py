"""This module contsins the class that builds the Multi-Step RAG's LangGraph"""

from rag_nodes import MultiStepRAGNodes
from rag_states import RAGStates
from langgraph.graph import StateGraph, END

# Defining All Node Names
QUESTION_REWRITER_NODE = "question_rewriter_node"
QUESTION_CLASSIFIER_NODE = "question_classifier_node"
OFFTOPIC_NODE = "offtopic"
RETRIEVER_NODE = "retriever_node"
CANCEL_RETRIEVAL_NODE = "cancel_node"
REPHRASE_QUESTION_NODE = "rephrase_node"
GENERATE_RESPONSE_NODE = "generate_response_node"


class MultiStepRAGGraphBuilder:
    """This class creates the Multi-Step RAG's graph"""

    nodes = MultiStepRAGNodes()
    graph_builder = StateGraph(RAGStates)
