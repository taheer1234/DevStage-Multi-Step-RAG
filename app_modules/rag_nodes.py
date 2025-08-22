"""
This module contain the Node(s) Class implementation to be used by the Multi-Step RAG's LangGraph.
"""

from rag_states import RAGStates
from chains import QR_chain, QC_chain, RDC_chain, RQ_chain, GR_chain
from langchain_core.messages import HumanMessage, AIMessage
from models import model

# Defining All Node Names
# QUESTION_REWRITER_NODE = "question_rewriter_node"
# QUESTION_CLASSIFIER_NODE = "question_classifier_node"
OFFTOPIC_NODE = "offtopic"
RETRIEVER_NODE = "retriever_node"
# CANCEL_RETRIEVAL_NODE = "cancel_node"
# REPHRASE_QUESTION_NODE = "rephrase_node"
# GENERATE_RESPONSE_NODE = "generate_response_node"


class MultiStepRAGNodes:
    """This class contains nodes used by the Multi-Step RAG's LangGraph."""

    def question_rewriter_node(self, state: RAGStates):
        """
        This node implements the question rewriter node.
        The node rewrites the user query in a way that it can
        better retrieve information form the vector database.
        """
        response = QR_chain.invoke(
            {"query": state["query"], "chat_history": state["chat_history"]}
        ).content

        print("\nRewritten Question/Statement: ", response)

        return {
            "rewritten_query": response,
            "chat_history": [HumanMessage(state["query"])],
        }

    def question_classifier_node(self, state: RAGStates):
        """
        This node implements the question classifier node.
        The node classifies the rewritten query as either on topic or off topic.
        """
        return {
            "ontopic_classification": QC_chain.invoke(
                {"rewritten_query": state["rewritten_query"]}
            ).classification
        }

    def question_classifier_router(self, state: RAGStates):
        """
        This is a router node. It redirects to either the retriever node or off topic node
        based on the classification of the question classifier node.
        """
        if state["ontopic_classification"]:
            return RETRIEVER_NODE
        return OFFTOPIC_NODE
    
    def offtopic(self, state: RAGStates):

        return {
            "response": model.invoke(
                "Generate a response to the user telling them thier query is not related to A Room with a View by E.M.forster and give a relevant question next time. DON'T ASK ABOUT WHAT ELSE YOU CAN HELP WITH."
            ).content
        }
