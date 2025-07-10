from typing import Any, Dict

from tqdm import tqdm

from genai_docs_helper.chains.retrieval_grader import retrieval_grader
from genai_docs_helper.state import GraphState


def grade_documents(state: GraphState) -> Dict[str, Any]:
    """
    Determines whether the retrieved documents are relevant to the question
    If any document is not relevant, we will set a flag to run web search

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Filtered out irrelevant documents and updated web_search state
    """

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]

    filtered_docs = []
    for d in tqdm(documents):
        score = retrieval_grader.invoke({"question": question, "document": d.page_content})
        grade = score.binary_score
        if grade.lower() == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            continue
    # if len(filtered_docs) == 0:
    #     print("---GRADE: NO RELEVENT DOCUMENT ARE FOUND---")
    #     return "end"
    # return {"documents": filtered_docs, "question": question}
    return {
        "question": question,
        "documents": filtered_docs,
        "history": state.get("history", []),
        "retry_count": state.get("retry_count", 0),
        "generation": state.get("generation", ""),
    }
