"""This script runs the Multi-Step RAG."""

from app_modules.graph import MultiStepRAGGraphBuilder

graph_builder = MultiStepRAGGraphBuilder()
graph = graph_builder.create_graph()
config = {"configurable": {"thread_id": 1}}

while True:
    query = input("User: ")
    if query.lower() == "exit":
        break
    response = graph.invoke({"query": query}, config=config)
    print("\nAI: ", response["response"], "\n")
