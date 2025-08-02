from typing import Any, List

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from genai_docs_helper.config import llm


class DocumentScore(BaseModel):
    """Score for document relevance"""

    document_index: int = Field(description="Index of the document (0-based)")
    relevance_score: float = Field(description="Relevance score from 0.0 to 1.0")
    reasoning: str = Field(description="Brief explanation of the score")


class DocumentRanking(BaseModel):
    """Ranking of documents by relevance"""

    rankings: List[DocumentScore] = Field(description="List of document scores, ordered by relevance")


structured_llm_ranker = llm.with_structured_output(DocumentRanking)

system = """You are an expert document ranker. Given a question and a list of documents, rank them by relevance to the question.

For each document, provide:
1. document_index: The index of the document (starting from 0)
2. relevance_score: A score from 0.0 to 1.0 (1.0 being most relevant)
3. reasoning: Brief explanation of why this score was given

Rank documents based on:
- Direct relevance to the question
- Quality and depth of information
- Specificity to the topic
- Completeness of the answer

Return the rankings ordered from most relevant to least relevant."""

ranking_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Question: {question}\n\nDocuments to rank:\n{documents_text}"),
    ]
)


def format_documents_for_ranking(documents: List[Any]) -> str:
    """Format documents for the ranking prompt"""
    formatted = []
    for i, doc in enumerate(documents):
        content = doc.page_content if hasattr(doc, "page_content") else str(doc)
        # Truncate long documents for efficiency
        content = content[:500] + "..." if len(content) > 500 else content
        formatted.append(f"Document {i}:\n{content}\n")
    return "\n".join(formatted)


def rerank_documents(question: str, documents: List[Any]) -> List[Any]:
    """Rerank documents based on relevance to the question"""
    if not documents:
        return documents

    try:
        # Format documents for ranking
        documents_text = format_documents_for_ranking(documents)

        # Get rankings from LLM
        response = structured_llm_ranker.invoke({"question": question, "documents_text": documents_text})

        # Sort by relevance score (highest first) and reorder documents
        sorted_rankings = sorted(response.rankings, key=lambda x: x.relevance_score, reverse=True)

        # Return documents in the new order
        reranked = []
        for ranking in sorted_rankings:
            if 0 <= ranking.document_index < len(documents):
                reranked.append(documents[ranking.document_index])

        # Add any documents that weren't ranked
        ranked_indices = {r.document_index for r in sorted_rankings}
        for i, doc in enumerate(documents):
            if i not in ranked_indices:
                reranked.append(doc)

        return reranked

    except Exception as e:
        print(f"Error in document reranking: {e}")
        # Fallback: return original order
        return documents


document_reranker = rerank_documents

if __name__ == "__main__":
    # Test with sample documents
    class MockDoc:
        def __init__(self, content):
            self.page_content = content

    docs = [
        MockDoc("Machine learning models for demand forecasting include ARIMA and neural networks."),
        MockDoc("Weather affects retail demand significantly during seasonal periods."),
        MockDoc("Advanced forecasting techniques like ensemble methods improve accuracy."),
    ]

    question = "What machine learning models are used for demand forecasting?"
    reranked = document_reranker(question, docs)

    for i, doc in enumerate(reranked):
        print(f"{i+1}. {doc.page_content}")
