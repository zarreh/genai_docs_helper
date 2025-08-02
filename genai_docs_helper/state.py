from datetime import datetime
from typing import Any, Dict, List, Optional, TypedDict


class GraphState(TypedDict):
    """
    Represents the complete state of the document retrieval and generation graph.

    This state object tracks the entire lifecycle of a query through the system,
    from initial question processing through document retrieval, grading, and
    final answer generation. It includes performance metrics and error tracking
    for production monitoring.

    Attributes:
        question (str): Current processed question (may be paraphrased)
        original_question (str): The user's original unmodified question
        generation (str): Final generated answer from the LLM
        documents (List[str]): Current filtered and relevant documents
        history (List[str]): Historical record of all generated responses
        retry_count (int): Number of retry attempts for the current question

        # Enhanced retrieval and processing fields
        query_variations (Optional[List[str]]): Alternative phrasings for better retrieval
        retrieved_documents_raw (Optional[List[Any]]): All documents before filtering
        reranked_documents (Optional[List[Any]]): Documents after semantic reranking

        # Performance and monitoring fields
        cache_key (Optional[str]): Unique identifier for caching results
        performance_metrics (Optional[Dict[str, Any]]): Timing and efficiency data
        error_log (Optional[List[str]]): Accumulated error messages for debugging
        confidence_score (Optional[float]): System confidence in the answer (0.0-1.0)
        timestamp (Optional[datetime]): Processing timestamp for audit trails
    """

    question: str
    original_question: str
    generation: str
    documents: List[str]
    history: List[str]
    retry_count: int

    # Enhanced retrieval capabilities
    query_variations: Optional[List[str]]
    retrieved_documents_raw: Optional[List[Any]]
    reranked_documents: Optional[List[Any]]

    # Production monitoring and optimization
    cache_key: Optional[str]
    performance_metrics: Optional[Dict[str, Any]]
    error_log: Optional[List[str]]
    confidence_score: Optional[float]
    timestamp: Optional[datetime]
