from typing import List

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from pydantic import BaseModel, Field

from genai_docs_helper.config import llm


class DocumentRelevance(BaseModel):
    """Relevance score for a single document"""

    document_index: int = Field(description="Index of the document (0-based)")
    is_relevant: bool = Field(description="Whether the document is relevant to the question")
    confidence: float = Field(description="Confidence score from 0.0 to 1.0")


class BatchGradeResult(BaseModel):
    """Batch grading results for multiple documents"""

    scores: List[DocumentRelevance] = Field(description="Relevance scores for each document")


structured_batch_grader = llm.with_structured_output(BatchGradeResult)

system = """You are a grader assessing relevance of multiple documents to a user question.
For each document, determine if it contains information relevant to answering the question.

Grade each document as:
- is_relevant: true if the document contains relevant information
- confidence: your confidence in this assessment (0.0 to 1.0)

Be efficient but accurate. Look for keywords, concepts, and semantic relevance."""

batch_grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Question: {question}\n\nDocuments to grade:\n{documents_text}"),
    ]
)


def format_documents_for_batch_grading(documents: List[str]) -> str:
    """Format documents for batch grading"""
    formatted = []
    for i, doc in enumerate(documents):
        # Truncate for efficiency
        content = doc[:500] + "..." if len(doc) > 500 else doc
        formatted.append(f"[Document {i}]:\n{content}\n")
    return "\n".join(formatted)


# Create a proper chain using RunnablePassthrough
batch_document_grader = (
    RunnablePassthrough.assign(documents_text=lambda x: format_documents_for_batch_grading(x["documents"]))
    | batch_grade_prompt
    | structured_batch_grader
)

