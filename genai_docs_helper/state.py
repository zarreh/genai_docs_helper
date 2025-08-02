from datetime import datetime
from typing import Any, Dict, List, Optional, TypedDict


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        web_search: whether to add search
        documents: list of documents
    """

    question: str
    original_question: str  # Store original for comparison
    generation: str
    documents: List[str]
    history: List[str]
    retry_count: int

    # New fields for production robustness
    query_variations: Optional[List[str]]  # Multiple query variations
    retrieved_documents_raw: Optional[List[Any]]  # All retrieved docs before filtering
    reranked_documents: Optional[List[Any]]  # Documents after reranking
    cache_key: Optional[str]  # For caching
    performance_metrics: Optional[Dict[str, Any]]  # Timing and performance data
    error_log: Optional[List[str]]  # Track errors
    confidence_score: Optional[float]  # Confidence in the answer
    timestamp: Optional[datetime]  # For tracking and debugging
