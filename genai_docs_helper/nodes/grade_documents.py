import time
from typing import Any, Dict, List

from genai_docs_helper.chains.batch_grader import batch_document_grader
from genai_docs_helper.chains.retrieval_grader import retrieval_grader
from genai_docs_helper.state import GraphState


def grade_document_batch(documents: List[Any], question: str, batch_size: int = 5) -> List[tuple]:
    """Grade documents in batches for efficiency"""
    results = []

    for i in range(0, len(documents), batch_size):
        batch = documents[i : i + batch_size]
        try:
            # Use batch grader for multiple documents at once
            batch_results = batch_document_grader.invoke(
                {"question": question, "documents": [doc.page_content for doc in batch]}
            )

            for j, score in enumerate(batch_results.scores):
                results.append((batch[j], score.is_relevant, score.confidence))

        except Exception as e:
            print(f"Batch grading error: {e}")
            # Fallback to individual grading
            for doc in batch:
                try:
                    score = retrieval_grader.invoke({"question": question, "document": doc.page_content})
                    results.append((doc, score.binary_score.lower() == "yes", 0.5))
                except:
                    results.append((doc, True, 0.3))  # Include with low confidence

    return results


def grade_documents(state: GraphState) -> Dict[str, Any]:
    """
    Fast document grading with early stopping and batch processing
    """
    print("---FAST DOCUMENT GRADING---")
    start_time = time.time()

    question = state["question"]
    documents = state["documents"]
    error_log = state.get("error_log", [])

    if not documents:
        print("---NO DOCUMENTS TO GRADE---")
        return {
            "question": question,
            "documents": [],
            "history": state.get("history", []),
            "retry_count": state.get("retry_count", 0),
            "generation": state.get("generation", ""),
            "error_log": error_log,
            "confidence_score": 0.0,
        }

    # Early stopping configuration
    min_relevant_docs = 5
    confidence_threshold = 0.7

    filtered_docs = []
    confidence_scores = []

    print(f"Grading {len(documents)} documents in batches...")

    # Grade documents in batches
    graded_results = grade_document_batch(documents, question, batch_size=5)

    # Process results with early stopping
    high_confidence_count = 0
    for doc, is_relevant, confidence in graded_results:
        if is_relevant:
            filtered_docs.append(doc)
            confidence_scores.append(confidence)

            if confidence >= confidence_threshold:
                high_confidence_count += 1

            # Early stopping: if we have enough high-confidence documents
            if high_confidence_count >= min_relevant_docs and len(filtered_docs) >= 10:
                print(f"Early stopping: Found {high_confidence_count} high-confidence documents")
                break

    # Calculate overall confidence
    overall_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0

    grading_time = time.time() - start_time

    # Update performance metrics
    performance_metrics = state.get("performance_metrics", {})
    performance_metrics.update(
        {
            "grading_time": grading_time,
            "documents_graded": len(graded_results),
            "documents_filtered": len(filtered_docs),
            "overall_confidence": overall_confidence,
            "early_stopped": high_confidence_count >= min_relevant_docs,
        }
    )

    print(
        f"Grading completed in {grading_time:.2f}s: {len(filtered_docs)} relevant docs (confidence: {overall_confidence:.2f})"
    )

    return {
        "question": question,
        "documents": filtered_docs,
        "history": state.get("history", []),
        "retry_count": state.get("retry_count", 0),
        "generation": state.get("generation", ""),
        "error_log": error_log,
        "confidence_score": overall_confidence,
        "performance_metrics": performance_metrics,
        "original_question": state.get("original_question", question),
        "query_variations": state.get("query_variations", []),
        "cache_key": state.get("cache_key"),
    }
