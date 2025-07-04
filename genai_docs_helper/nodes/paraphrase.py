from typing import Any

from genai_docs_helper.chains.paraphraser import paraphraser_chain
from genai_docs_helper.state import GraphState


def paraphrase(state: GraphState) -> dict[str, Any]:
    print("---PARAPHRASE---")
    question = state["question"]
    paraphrased_question = paraphraser_chain.invoke({"question": question})

    retry_count = state.get("retry_count", 0) + 1

    generation = ""
    if retry_count >= 2:
        generation = "No groud truth were found for your question. Probably not related to the topic."

    return {
        "question": paraphrased_question,
        "documents": [],  # Fixed the erro on studio - Reset documents to trigger a new retrieval
        "history": state.get("history", []),
        "retry_count": retry_count,
        "generation": generation,
    }
