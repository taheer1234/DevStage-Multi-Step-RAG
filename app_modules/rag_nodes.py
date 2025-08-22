"""
This module contain the Node(s) Class implementation to be used by the Multi-Step RAG's LangGraph.
"""

from rag_states import RAGStates
from chains import QR_chain, QC_chain, RDC_chain, RQ_chain, GR_chain
from langchain_core.messages import HumanMessage, AIMessage


class MultiStepRAGNodes:
    """This class contains nodes used by the Multi-Step RAG's LangGraph."""

    def question_rewriter_node(self, state: RAGStates):
        """
        This node implements the question rewriter node.
        The node rewrites the user query inn a way that it can
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
