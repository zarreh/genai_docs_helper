from typing import List

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence

from genai_docs_helper.config import llm

system = """You are an expert at generating multiple variations of a search query to improve retrieval performance.
Generate 3-5 different variations of the given question that would help retrieve relevant documents.

Guidelines:
1. Maintain the core intent and meaning
2. Use different phrasings and terminology
3. Include both specific and general variations
4. Consider synonyms and related concepts
5. Keep variations concise and focused

Return ONLY the variations, one per line, without numbering or additional text."""

query_expansion_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Original question: {question}"),
    ]
)


def parse_query_variations(response: str) -> List[str]:
    """Parse the response into a list of query variations"""
    variations = [line.strip() for line in response.strip().split("\n") if line.strip()]
    return variations[:5]  # Limit to 5 variations


query_expander_chain: RunnableSequence = query_expansion_prompt | llm | StrOutputParser() | parse_query_variations

if __name__ == "__main__":
    question = "What are the key challenges in demand forecasting?"
    variations = query_expander_chain.invoke({"question": question})
    print(f"Original: {question}")
    print("Variations:")
    for i, var in enumerate(variations, 1):
        print(f"{i}. {var}")
