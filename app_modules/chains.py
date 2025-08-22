"""
This module contains the Chain(s) that could be invoked during the Multi-Step RAG's execution.
"""

from models import model
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

# QUESTION REWRITER IMPLEMENTATION
# The template, prompt, and chain for the Multi-Step RAG's [Question Rewriter] Node.
question_rewriter_template = [
    (
        "system",
        "YOU ARE AN EXPERT REWRITER. Rewrite the user's question to make it "
        "more relevant to the context of the chat. The new question you generate"
        "will be sent to a RAG retriever so make sure your "
        "question is all encompassing to the context.\n"
        "Also ENSURE that it is worded in a way that "
        "makes it good for searching vector databases.\n"
        "DONT MENTION THE BOOK NAME OR AUTHOR NAME CAUSE THAT IS USELESS!!!\n"
        "Context: {chat_history}\n"
        "User's question: {query}",
    )
]

question_rewriter_prompt = ChatPromptTemplate.from_messages(question_rewriter_template)

QR_chain = question_rewriter_prompt | model


# QUESTION CLASSIFIER IMPLEMENTATION
# The structured output, template, prompt, and chain
# for the Multi-Step RAG's [Question Classifier] Node.
class QuestionClassifierOutput(BaseModel):
    """This class creates the output schema to be used by the question classifier chain."""

    classification: bool = Field(
        description="If the user query is related to A Room with a View by E.M.forster return True"
        ", if it is not related return False."
    )


question_classifier_template = [
    (
        "system",
        "YOU ARE AN EXPERT ON THE BOOK A ROOM WITH A VIEW BY E.M. FORSTER. "
        "Check and make sure the users query is related to "
        "A Room with a View by E.M.forster."
        "If the user query is related to A Room with a View by E.M.forster return 'True', "
        "if it is not related return 'False'."
        "ONLY REPLY 'TRUE' TO THINGS THAT ARE ABOUT THE BOOK A ROOM WITH A VIEW BY E.M. FORSTER. "
        "DON'T REPLY 'TRUE' TO ANYTHING ELSE. EVERYTHING ELSE IS 'FALSE'."
        "User Query: {rewritten_query}",
    )
]

question_classifier_prompt = ChatPromptTemplate.from_messages(
    question_classifier_template
)

QC_chain = question_classifier_prompt | model.with_structured_output(
    QuestionClassifierOutput
)


# RETRIEVED DOCUMENT CLASSIFIER IMPLEMENTATION
# The structured output, template, prompt, and chain for the Multi-Step RAG's [Retriever] Node.
class RetrievedDocClassifierOutput(BaseModel):
    """This class creates the output schema to be
    used by the retrieved document classifier chain."""

    classification: bool = Field(
        description="The classification if 'True' if the document is relevant "
        "to the user query otherwise the classification is 'False'."
    )


retrieved_doc_classifier_template = [
    "system",
    "READ THE RETRIEVED DOCUMENT AND RESPOND WITH EITHER TRUE OR FALSE. "
    "IF THE RETRIEVED DOCUMENT ANSWERS THE USER QUERY RETURN TRUE. "
    "IF THE RETRIEVED DOCUMENT DOESN'T ANSWER THE USER QUERY RETURN FALSE. "
    "BE LENIENT WHEN CLASSIFYING. "
    "RETRIEVED DOCUMENT: {document}\n"
    "USER QUERY: {query}",
]

retrieved_doc_classifier_prompt = ChatPromptTemplate.from_messages(
    retrieved_doc_classifier_template
)

RDC_chain = retrieved_doc_classifier_prompt | model.with_structured_output(
    RetrievedDocClassifierOutput
)


# REPHRASE QUESTION IMPLEMENTATION
# The template, prompt, and chain for the Multi-Step RAG's [Rephrase Question] Node.
rephrase_question_template = [
    (
        "system",
        "Using the user query and context. Rephrase the user's "
        "question to get more information from the RAG retriever, "
        "relating to what the user wants to know. "
        "REPHRASE IN A WAY TO BEST GET ALL OF THE RELATED INFO FROM THE RETRIEVER. "
        "JUST MAKE THE NEW QUERY PHRASED INTO A STATEMENT INSTEAD OF "
        "A QUESTION THAT COULD GET RELEVANT DOCUMENTS FROM THE RAG.\n"
        "ALSO DONT MENTION THE BOOK NAME OR AUTHOR NAME CAUSE THAT IS USELESS!!!\n"
        "User query: {query}"
        "Context: {chat_history}",
    )
]

rephrase_question_prompt = ChatPromptTemplate.from_messages(rephrase_question_template)

RQ_chain = rephrase_question_prompt | model


# GENERATE RESPONSE IMPLEMENTATION
# The template, prompt, and chain for the Multi-Step RAG's [Generate Response] Node.
generate_response_template = [
    (
        "system",
        "Given the user's query and the documents fetched from the RAG retriever. "
        "Generate a response that satisfies the user's query.\n"
        "Give a natural response to the user without them knowing a RAG is being used.\n"
        "Don't say thing like 'in the passage' or related to that. Just give a "
        "natural response to the user using the info provided, "
        "but don't mention anything about any provided docs."
        "Documents fetched from retriever: {document}"
        "User's query: {query}\n",
    )
]

generate_response_prompt = ChatPromptTemplate.from_messages(generate_response_template)

GR_chain = generate_response_prompt | model
