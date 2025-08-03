from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from genai_docs_helper.config import llm


class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")


structured_llm_grader = llm.with_structured_output(GradeDocuments)

system = """You are a grader assessing relevance of a retrieved document to a user question.

Be LENIENT in your grading. A document is relevant if it:
- Contains ANY keywords from the question
- Discusses related concepts or topics
- Provides context that could help answer the question
- Mentions similar terminology or synonyms

Only mark as 'no' if the document is completely unrelated to the question topic.

Give a binary score 'yes' or 'no' to indicate whether the document is relevant to the question."""
grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)

retrieval_grader = grade_prompt | structured_llm_grader
