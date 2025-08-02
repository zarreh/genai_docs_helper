import time
from typing import Any

from genai_docs_helper.chains.paraphraser import paraphraser_chain
from genai_docs_helper.state import GraphState


def paraphrase(state: GraphState) -> dict[str, Any]:
    print("---ENHANCED PARAPHRASE---")
    start_time = time.time()

    question = state["question"]
    original_question = state.get("original_question", question)
    retry_count = state.get("retry_count", 0) + 1
    error_log = state.get("error_log", [])

    print(f"Retry count: {retry_count}")

    try:
        paraphrased_question = paraphraser_chain.invoke({"question": question})

        # Enhanced fallback logic
        generation = ""
        if retry_count >= 3:
            generation = "I apologize, but I couldn't find relevant information to answer your question. This might be because:\n1. The question is outside the scope of available documents\n2. The information might be phrased differently in the source material\n3. The topic might not be covered in the current knowledge base\n\nPlease try rephrasing your question or asking about a different topic."

        paraphrase_time = time.time() - start_time

        # Update performance metrics
        performance_metrics = state.get("performance_metrics", {})
        performance_metrics["paraphrase_time"] = paraphrase_time

        return {
            "question": paraphrased_question,
            "original_question": original_question,
            "documents": [],  # Reset documents to trigger new retrieval
            "history": state.get("history", []),
            "retry_count": retry_count,
            "generation": generation,
            "error_log": error_log,
            "performance_metrics": performance_metrics,
            "query_variations": [],  # Reset query variations
            "cache_key": None,  # Reset cache key
        }

    except Exception as e:
        print(f"Error in paraphrase: {e}")
        error_log.append(f"Paraphrase error: {str(e)}")

        # Fallback: slight modification of original question
        fallback_question = (
            f"Can you explain {question.lower()}?"
            if not question.lower().startswith(("what", "how", "why", "when", "where"))
            else question
        )

        return {
            "question": fallback_question,
            "original_question": original_question,
            "documents": [],
            "history": state.get("history", []),
            "retry_count": retry_count,
            "generation": "",
            "error_log": error_log,
            "performance_metrics": state.get("performance_metrics", {}),
            "query_variations": [],
            "cache_key": None,
        }
