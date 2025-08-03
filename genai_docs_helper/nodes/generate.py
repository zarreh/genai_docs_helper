import time
from typing import Any, Dict

from genai_docs_helper.cache.query_cache import QueryCache
from genai_docs_helper.chains.generation import generation_chain
from genai_docs_helper.state import GraphState
from genai_docs_helper.utils import get_logger, log_performance_metrics

logger = get_logger(__name__)

# Initialize cache with Redis disabled by default
cache = QueryCache(redis_url="redis://localhost:6379", ttl=3600, enable_redis=False)


def generate(state: GraphState) -> Dict[str, Any]:
    logger.info("=== STARTING ANSWER GENERATION ===")
    start_time = time.time()

    question = state["question"]
    documents = state["documents"]

    logger.debug(f"Generating answer for question: '{question[:100]}...'")
    logger.debug(f"Using {len(documents)} documents for context")

    # Check cache for similar question + documents combination
    cache_key = state.get("cache_key", "")
    if cache_key:
        cached_generation = cache.get(question, context=str(len(documents)))
        if cached_generation and cached_generation.get("generation"):
            logger.info("Cache hit for generation, returning cached answer")
            cache_metrics = {
                "generation_time": 0.0,
                "generation_source": "cache",
                "documents_used": len(documents),
            }
            log_performance_metrics(logger, cache_metrics, cache_key[:8])
            
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
        logger.info(f"Limiting context to top {max_docs} documents for faster generation")
        documents = documents[:max_docs]

    try:
        logger.debug("Invoking generation chain")
        generation = generation_chain.invoke({"context": documents, "question": question})
        logger.info("Successfully generated answer")
        logger.debug(f"Generated answer preview: '{generation[:100]}...'")
    except Exception as e:
        logger.error(f"Generation failed: {e}", exc_info=True)
        generation = "I apologize, but I encountered an error while generating the answer. Please try again."

    generation_time = time.time() - start_time

    # Update performance metrics
    performance_metrics = state.get("performance_metrics", {})
    generation_metrics = {
        "generation_time": generation_time,
        "documents_used": len(documents),
        "generation_length": len(generation),
    }
    performance_metrics.update(generation_metrics)
    
    log_performance_metrics(logger, generation_metrics, cache_key[:8])

    # Cache the generation
    if cache_key and generation != "I apologize, but I encountered an error while generating the answer. Please try again.":
        logger.debug(f"Caching generation for key: {cache_key[:8]}...")
        cache.set(question, str(len(documents)), {"generation": generation})

    history = state.get("history", [])
    history.append(generation)

    retry_count = state.get("retry_count", 0)

    logger.info(f"=== GENERATION COMPLETED in {generation_time:.2f}s ===")

    return {
        "documents": documents,
        "question": question,
        "generation": generation,
        "history": history,
        "retry_count": retry_count,
        "performance_metrics": performance_metrics,
    }
