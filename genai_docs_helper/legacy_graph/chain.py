from dotenv import find_dotenv, load_dotenv
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser

from genai_docs_helper.config import llm

# from langchain_core.output_parsers import StrOutputParser  # type: ignore


# Load environment variables
load_dotenv(find_dotenv())

# Custom prompt template
custom_prompt_template = """You are a helpful AI assistant specialized in demand forecasting and related topics.
Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}

Question: {question}

Please provide your answer following these guidelines:
1. Use ONLY information from the provided context
2. ALWAYS cite your sources using [Source X] format where X is the source number
3. If multiple sources support a statement, cite all relevant sources: [Source 1,2]
4. If the context doesn't contain enough information, clearly state that
5. Structure your response in a clear, logical manner
6. Keep the answer focused and relevant to demand forecasting

Remember: Every significant statement should have a source citation.

Answer: Let me help you with that.
"""

# Create the prompt template
PROMPT = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])


chain = PROMPT | llm | StrOutputParser()


if __name__ == "__main__":
    # Example usage
    context = "Demand forecasting is the process of predicting future customer demand for a product or service."
    question = "What is demand forecasting?"

    # Run the chain
    response = chain.invoke({"context": context, "question": question})

    print(response)  # Output the response from the chain
