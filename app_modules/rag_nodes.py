"""
This module contain the Node(s) Class implementation to be used by the Multi-Step RAG's LangGraph.
"""

from rag_states import RAGStates
from chains import QR_chain, QC_chain, RDC_chain, RQ_chain, GR_chain
from models import model
from retrievers import pinecone_mmr_retriever
from langchain_core.messages import HumanMessage, AIMessage


# Defining All Node Names
# QUESTION_REWRITER_NODE = "question_rewriter_node"
# QUESTION_CLASSIFIER_NODE = "question_classifier_node"
OFFTOPIC_NODE = "offtopic"
RETRIEVER_NODE = "retriever_node"
CANCEL_RETRIEVAL_NODE = "cancel_node"
REPHRASE_QUESTION_NODE = "rephrase_node"
GENERATE_RESPONSE_NODE = "generate_response_node"


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

    def offtopic_node(self):
        """
        This node implements the off topic node. This node tells the
        user their query isn't related to the data in the vector database.
        """
        return {
            "response": model.invoke(
                "Generate a response to the user telling them thier query is not related to "
                "A Room with a View by E.M.forster and give a relevant question next time. "
                "DON'T ASK ABOUT WHAT ELSE YOU CAN HELP WITH."
            ).content
        }

    def retriever_node(self, state: RAGStates):
        """
        This node implements the retriever node.
        The node retrieves documents from the vector database related to the rewritten query.
        """
        print("\nRetrieving Documents...")
        docs = pinecone_mmr_retriever("my-index").invoke(state["rewritten_query"])
        useable_docs = []
        for doc in docs:
            if RDC_chain.invoke(
                {"document": doc.page_content, "query": state["rewritten_query"]}
            ).classification:
                useable_docs.append(doc)
        print(
            "Useable Docs: ",
            useable_docs,
            "\nNumber of Useable Docs: ",
            len(useable_docs),
        )
        return {"documents": useable_docs, "retriever_invoke_number": 1}

    def retrieval_router(self, state: RAGStates):
        """
        This is a router node. It redirects to either the rephrase question node or
        generate response node depending upon if any useable documents were found by the retriever.
        If no useable were documents were found after 2 rephrases,
        this node redirects to the cancel node.
        """
        print(
            "\nIN RETRIEVAL ROUTER\nRetriever Invoke Number = ",
            state["retriever_invoke_number"],
        )
        if len(state["documents"]) > 0:
            return GENERATE_RESPONSE_NODE
        if state["retriever_invoke_number"] >= 3:
            return CANCEL_RETRIEVAL_NODE
        return REPHRASE_QUESTION_NODE

    def cancel_node(self, state: RAGStates):
        """
        This node implements the cancel node. This node simply tells the user
        it couldnt find anything related to thier query in the database.
        """
        print("\nIN CANCEL NODE")
        return {
            "response": model.invoke(
                "Generate a response to the user telling them the retriever didn't find anything related to what they were asking. DON'T ASK ABOUT WHAT ELSE YOU CAN HELP WITH."
            ).content,
            "retriever_invoke_number": -3,
        }
