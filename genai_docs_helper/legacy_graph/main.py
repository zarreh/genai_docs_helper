from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnableLambda
from langgraph.graph import END, StateGraph

from genai_docs_helper.legacy_graph.chain import chain
from genai_docs_helper.config import EMBEDDING

# Load Chroma vector store
vectorstore = Chroma(persist_directory="./data/chroma_db_ollama", embedding_function=EMBEDDING)

# Define retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})


# Define LangGraph state
class GraphState(dict):
    question: str
    context: str
    answer: str = None
    docs: list[str] = []


# Node: Retrieve documents
def retrieve_documents(state: GraphState) -> GraphState:
    question = state["question"]
    docs = retriever.invoke(question)  # Updated method
    context = "\n\n".join([doc.page_content for doc in docs])
    return {"question": question, "context": context, "docs": docs}


# Node: Run the chain
def run_chain(state: GraphState) -> GraphState:
    try:
        response = chain.invoke({"context": state["context"], "question": state["question"]})
    except Exception as e:
        response = f"Error during chain execution: {e}"
    return {"question": state["question"], "context": state["context"], "answer": response}


# Build the LangGraph
graph_builder = StateGraph(GraphState)
graph_builder.add_node("retrieve_documents", RunnableLambda(retrieve_documents))
graph_builder.add_node("run_chain", RunnableLambda(run_chain))

# Define edges
graph_builder.set_entry_point("retrieve_documents")
graph_builder.add_edge("retrieve_documents", "run_chain")
graph_builder.add_edge("run_chain", END)

# Compile the graph
graph = graph_builder.compile()

# Example usage
if __name__ == "__main__":
    question = "What are the key challenges in demand forecasting?"
    result = graph.invoke({"question": question})
    print("Answer:\n", result.get("answer", "No answer returned."))
