import hashlib
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List

import numpy as np
from langchain.schema import Document
from langchain_community.vectorstores import Chroma

from genai_docs_helper.cache.query_cache import QueryCache
from genai_docs_helper.chains.query_expander import query_expander_chain
from genai_docs_helper.config import EMBEDDING, VECTOR_STORE_PATH
from genai_docs_helper.state import GraphState

# Initialize cache
cache = QueryCache(redis_url="redis://localhost:6379", ttl=3600)

# Load Chroma vector store
vectorstore = Chroma(persist_directory=VECTOR_STORE_PATH, embedding_function=EMBEDDING)

# Multiple retriever configurations for different strategies
fast_retriever = vectorstore.as_retriever(search_kwargs={"k": 20})
standard_retriever = vectorstore.as_retriever(search_kwargs={"k": 50})
comprehensive_retriever = vectorstore.as_retriever(search_kwargs={"k": 100})


def fast_semantic_rerank(question: str, documents: List[Document], top_k: int = 20) -> List[Document]:
    """Fast reranking using embeddings similarity (no LLM calls)"""
    if not documents or len(documents) <= top_k:
        return documents

    try:
        # Get question embedding
        question_embedding = EMBEDDING.embed_query(question)

        # Get document embeddings and compute similarities
        doc_scores = []
        for doc in documents:
            doc_embedding = EMBEDDING.embed_query(doc.page_content[:1000])  # Limit length for speed
            similarity = np.dot(question_embedding, doc_embedding)
            doc_scores.append((similarity, doc))

        # Sort by similarity and return top_k
        doc_scores.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in doc_scores[:top_k]]

    except Exception as e:
        print(f"Fast rerank error: {e}")
        return documents[:top_k]


def parallel_retrieve(queries: List[str], k: int = 50) -> List[Document]:
    """Retrieve documents in parallel for multiple queries"""
    all_docs = []
    seen_content = set()

    with ThreadPoolExecutor(max_workers=5) as executor:
        # Submit all retrieval tasks
        future_to_query = {executor.submit(standard_retriever.invoke, query): query for query in queries}

        # Collect results as they complete
        for future in as_completed(future_to_query):
            query = future_to_query[future]
            try:
                docs = future.result(timeout=5)  # 5 second timeout per query
                for doc in docs:
                    content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()
                    if content_hash not in seen_content:
                        seen_content.add(content_hash)
                        all_docs.append(doc)
            except Exception as e:
                print(f"Error retrieving for query '{query}': {e}")

    return all_docs


def retrieve(state: GraphState) -> Dict[str, Any]:
    print("---FAST ENHANCED RETRIEVE---")
    start_time = time.time()

    question = state["question"]
    original_question = state.get("original_question", question)

    # Check cache first
    cache_key = hashlib.md5(question.encode()).hexdigest()
    cached_result = cache.get(question)
    if cached_result and cached_result.get("documents"):
        print("Cache hit! Returning cached results")
        cached_result["from_cache"] = True
        return cached_result

    try:
        # Fast path: Start with smaller retrieval
        print("Fast retrieval phase...")
        initial_docs = fast_retriever.invoke(question)

        # If we have enough high-quality docs, use them
        if len(initial_docs) >= 10:
            # Quick rerank using embeddings
            reranked_docs = fast_semantic_rerank(question, initial_docs, top_k=15)

            result = {
                "documents": reranked_docs,
                "question": question,
                "original_question": original_question,
                "query_variations": [],
                "cache_key": cache_key,
                "performance_metrics": {
                    "retrieval_time": time.time() - start_time,
                    "retrieval_strategy": "fast",
                    "documents_retrieved": len(initial_docs),
                    "final_documents": len(reranked_docs),
                },
                "error_log": state.get("error_log", []),
                "retry_count": state.get("retry_count", 0),
            }

            # Cache the result
            cache.set(question, "", result)
            return result

        # Comprehensive path: Generate variations and retrieve more
        print("Comprehensive retrieval phase...")

        # Generate query variations in parallel with initial retrieval
        query_variations = query_expander_chain.invoke({"question": question})

        # Parallel retrieval for all queries
        all_queries = [question] + query_variations[:3]  # Limit to 4 queries total
        all_documents = parallel_retrieve(all_queries)

        print(f"Retrieved {len(all_documents)} unique documents")

        # Fast semantic reranking first
        reranked_documents = fast_semantic_rerank(original_question, all_documents, top_k=min(30, len(all_documents)))

        # Take top documents for final output
        final_documents = reranked_documents[:20]

        retrieval_time = time.time() - start_time

        result = {
            "documents": final_documents,
            "question": question,
            "original_question": original_question,
            "query_variations": query_variations,
            "cache_key": cache_key,
            "performance_metrics": {
                "retrieval_time": retrieval_time,
                "retrieval_strategy": "comprehensive",
                "raw_documents_count": len(all_documents),
                "final_documents_count": len(final_documents),
                "query_variations_count": len(query_variations),
            },
            "error_log": state.get("error_log", []),
            "retry_count": state.get("retry_count", 0),
        }

        # Cache the result
        cache.set(question, "", result)

        print(f"Retrieval completed in {retrieval_time:.2f}s")
        return result

    except Exception as e:
        print(f"Error in retrieve: {e}")
        error_log = state.get("error_log", [])
        error_log.append(f"Retrieve error: {str(e)}")

        # Fallback to simple retrieval
        try:
            print("Falling back to simple retrieval...")
            documents = fallback_retriever.invoke(question)
            return {
                "documents": documents,
                "question": question,
                "original_question": original_question,
                "query_variations": [],
                "error_log": error_log,
                "retry_count": state.get("retry_count", 0),
            }
        except Exception as fallback_error:
            error_log.append(f"Fallback retrieve error: {str(fallback_error)}")
            return {
                "documents": [],
                "question": question,
                "original_question": original_question,
                "error_log": error_log,
                "retry_count": state.get("retry_count", 0),
            }
