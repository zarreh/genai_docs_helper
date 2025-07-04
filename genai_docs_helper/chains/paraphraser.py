# from typing import Any, Dict
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence

from genai_docs_helper.config import llm

# System prompt for paraphrasing
system = """You are a helpful assistant that rephrases user questions to improve clarity and retrieval performance.
Rephrase the question while preserving its original intent.

Return only the rephrased question. Do not include any explanations, prefixes, or additional commentary."""


# Prompt template
paraphrase_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Original question: {question}"),
    ]
)

# Chain definition
paraphraser_chain: RunnableSequence = paraphrase_prompt | llm | StrOutputParser()

if __name__ == "__main__":
    question = "what models did we try in this project?"
    paraphrased_question = paraphraser_chain.invoke({"question": question})
    print(f"Original question: {question}")
    print(f"Paraphrased question: {paraphrased_question}")
