from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from pydantic import BaseModel, Field

from genai_docs_helper.config import llm


class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""

    binary_score: bool = Field(description="Answer is grounded in the facts, True or False")


structured_llm_grader = llm.with_structured_output(GradeHallucinations)

system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts.

Give a binary score True or False. True means that the answer is grounded in / supported by the set of facts.

Be lenient in your assessment:
- If the answer correctly summarizes or paraphrases information from the documents, score it as True
- If the answer uses slightly different wording but conveys the same meaning, score it as True
- If the answer combines information from multiple documents logically, score it as True
- Only score as False if the answer contains information that directly contradicts the documents or makes claims not supported by any document

Consider semantic meaning, not just exact word matches."""

hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
    ]
)

hallucination_grader: RunnableSequence = hallucination_prompt | structured_llm_grader
