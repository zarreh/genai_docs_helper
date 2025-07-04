from typing import Any, Dict

from genai_docs_helper.chains.generation import generation_chain
from genai_docs_helper.state import GraphState


# def generate(state: GraphState) -> Dict[str, Any]:
#     print("---GENERATE---")
#     question = state["question"]
#     documents = state["documents"]

#     generation = generation_chain.invoke({"context": documents, "question": question})
#     return {"documents": documents, "question": question, "generation": generation}

def generate(state: GraphState) -> Dict[str, Any]:
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    generation = generation_chain.invoke({"context": documents, "question": question})

    history = state.get("history", [])
    history.append(generation)

    retry_count = state.get("retry_count", 0) + 1

    return {
        "documents": documents,
        "question": question,
        "generation": generation,
        "history": history,
        "retry_count": retry_count,
    }
