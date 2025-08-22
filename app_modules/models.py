"""This module contains the Model(s) that could be invoked during the Multi-Step RAG's execution."""

from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(model="qwen/qwen3-32b", temperature=0.1, reasoning_effort="none")
