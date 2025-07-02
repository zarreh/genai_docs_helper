from typing import TypedDict

from langchain_core.runnables import RunnableLambda
from langgraph.graph import StateGraph


# Define the state schema
class MyState(TypedDict):
    name: str


# Define a simple function
def greet(state: MyState) -> MyState:
    return {"name": f"Hello, {state['name']}!"}


# Build the graph
graph_builder = StateGraph(MyState)
graph_builder.add_node("greet", RunnableLambda(greet))
graph_builder.set_entry_point("greet")
graph = graph_builder.compile()

# Run the graph
if __name__ == "__main__":
    result = graph.invoke({"name": "Alireza"})
    print(result)
