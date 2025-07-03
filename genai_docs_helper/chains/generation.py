from langchain import hub
from langchain_core.output_parsers import StrOutputParser

from genai_docs_helper.config import llm

prompt = hub.pull("rlm/rag-prompt")

generation_chain = prompt | llm | StrOutputParser()
