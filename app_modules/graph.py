"""This module contsins the class that builds the Multi-Step RAG's LangGraph."""

from rag_nodes import MultiStepRAGNodes
from rag_states import RAGStates
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# Defining All Node Names
QUESTION_REWRITER_NODE = "question_rewriter_node"
QUESTION_CLASSIFIER_NODE = "question_classifier_node"
OFFTOPIC_NODE = "offtopic"
RETRIEVER_NODE = "retriever_node"
CANCEL_RETRIEVAL_NODE = "cancel_node"
REPHRASE_QUESTION_NODE = "rephrase_node"
GENERATE_RESPONSE_NODE = "generate_response_node"


class MultiStepRAGGraphBuilder:
    """This class creates the Multi-Step RAG's graph."""

    def create_graph(self):
        """This function sets up the nodes, workflow, and entry point for the Multi-Step RAG's graph."""
        nodes = MultiStepRAGNodes()
        graph_builder = StateGraph(RAGStates)
        memory = MemorySaver()

        # Adding the nodes to the graph.
        graph_builder.add_node(QUESTION_REWRITER_NODE, nodes.question_rewriter_node)
        graph_builder.add_node(QUESTION_CLASSIFIER_NODE, nodes.question_classifier_node)
        graph_builder.add_node(OFFTOPIC_NODE, nodes.offtopic_node)
        graph_builder.add_node(RETRIEVER_NODE, nodes.retriever_node)
        graph_builder.add_node(CANCEL_RETRIEVAL_NODE, nodes.cancel_node)
        graph_builder.add_node(REPHRASE_QUESTION_NODE, nodes.rephrase_question_node)
        graph_builder.add_node(GENERATE_RESPONSE_NODE, nodes.generate_response_node)

        # Creating the flow of the graph.
        graph_builder.add_edge(QUESTION_REWRITER_NODE, QUESTION_CLASSIFIER_NODE)
        graph_builder.add_conditional_edges(
            QUESTION_CLASSIFIER_NODE, nodes.question_classifier_router
        )

        graph_builder.add_edge(OFFTOPIC_NODE, END)
        graph_builder.add_conditional_edges(RETRIEVER_NODE, nodes.retrieval_router)

        graph_builder.add_edge(REPHRASE_QUESTION_NODE, RETRIEVER_NODE)
        graph_builder.add_edge(CANCEL_RETRIEVAL_NODE, END)
        graph_builder.add_edge(GENERATE_RESPONSE_NODE, END)

        # Setting the entry point to the graph.
        graph_builder.set_entry_point(QUESTION_REWRITER_NODE)

        return graph_builder.compile(checkpointer=memory)
