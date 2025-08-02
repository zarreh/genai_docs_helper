from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from pydantic import BaseModel, Field

from genai_docs_helper.config import llm


class ConfidenceScore(BaseModel):
    """Confidence score for document relevance."""

    confidence_score: float = Field(
        description="Confidence score from 0.0 to 1.0 indicating how relevant the document is to the question"
    )
    reasoning: str = Field(description="Brief explanation of the confidence score")


structured_llm_scorer = llm.with_structured_output(ConfidenceScore)

system = """You are a confidence scorer assessing how well a document answers or relates to a user question.
Provide a confidence score from 0.0 to 1.0 where:
- 1.0: Document directly and completely answers the question
- 0.8-0.9: Document provides most of the answer with good detail
- 0.6-0.7: Document provides partial answer or relevant context
- 0.4-0.5: Document has some relevance but limited usefulness
- 0.2-0.3: Document has minimal relevance
- 0.0-0.1: Document is not relevant

Consider:
- Direct relevance to the question
- Completeness of information
- Quality and specificity of content
- How well it addresses the user's intent"""

confidence_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Question: {question}\n\nDocument: {document}"),
    ]
)

confidence_scorer: RunnableSequence = confidence_prompt | structured_llm_scorer
