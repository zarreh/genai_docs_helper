import time
from typing import Any, Dict

from genai_docs_helper.cache.query_cache import QueryCache
from genai_docs_helper.chains.generation import generation_chain
from genai_docs_helper.state import GraphState

# Initialize cache with Redis disabled by default
cache = QueryCache(redis_url="redis://localhost:6379", ttl=3600, enable_redis=False)


def generate(state: GraphState) -> Dict[str, Any]:
    print("---GENERATE---")
    start_time = time.time()

    question = state["question"]
    documents = state["documents"]

    # Check cache for similar question + documents combination
    cache_key = state.get("cache_key", "")
    if cache_key:
        cached_generation = cache.get(question, context=str(len(documents)))
        if cached_generation and cached_generation.get("generation"):
            print("Using cached generation")
            return {
                "documents": documents,
                "question": question,
                "generation": cached_generation["generation"],
                "history": state.get("history", []),
                "retry_count": state.get("retry_count", 0),
                "from_cache": True,
            }

    # Limit document context for faster generation
    max_docs = 10
    if len(documents) > max_docs:
        print(f"Limiting context to top {max_docs} documents for faster generation")
        documents = documents[:max_docs]

    generation = generation_chain.invoke({"context": documents, "question": question})

    generation_time = time.time() - start_time

    # Update performance metrics
    performance_metrics = state.get("performance_metrics", {})
    performance_metrics["generation_time"] = generation_time

    # Cache the generation
    if cache_key:
        cache.set(question, str(len(documents)), {"generation": generation})

    history = state.get("history", [])
    history.append(generation)

    retry_count = state.get("retry_count", 0)

    return {
        "documents": documents,
        "question": question,
        "generation": generation,
        "history": history,
        "retry_count": retry_count,
        "performance_metrics": performance_metrics,
    }
