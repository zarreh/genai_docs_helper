from typing import Any, Dict

from langchain_community.vectorstores import Chroma

from genai_docs_helper.config import EMBEDDING, VECTOR_STORE_PATH
from genai_docs_helper.state import GraphState

# Load Chroma vector store
vectorstore = Chroma(persist_directory=VECTOR_STORE_PATH, embedding_function=EMBEDDING)

# Define retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 10})


def retrieve(state: GraphState) -> Dict[str, Any]:
    print("---RETRIEVE---")
    question = state["question"]
    documents = retriever.invoke(question)
    # context = "\n\n".join([doc.page_content for doc in documents])
    # return {"question": question, "context": context, "documents": documents}
    return {"documents": documents, "question": question}
