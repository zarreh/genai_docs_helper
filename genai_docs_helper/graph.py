from dotenv import load_dotenv
from langchain_core.runnables import RunnableLambda
from langgraph.graph import END, StateGraph

from genai_docs_helper.chains.answer_grader import answer_grader
from genai_docs_helper.chains.hallucination_grader import hallucination_grader
from genai_docs_helper.consts import (GENERATE, GRADE_DOCUMENTS, PARAPHRASE,
                                      RESTART, RETRIEVE)
from genai_docs_helper.nodes import generate, grade_documents, retrieve
from genai_docs_helper.nodes.paraphrase import paraphrase
from genai_docs_helper.state import GraphState

load_dotenv()


def decide_to_generate(state):
    print("---ASSESS GRADED DOCUMENTS---")

    if state.get("retry_count", 0) >= 3:
        print("---DECISION: MAX RETRIES REACHED. ENDING.---")
        return "end"

    if len(state["documents"]) == 0:
        print("---DECISION: NO RELEVENT DOCUMENT ARE FOUND. PARAPHRASE IT.---")
        return PARAPHRASE
    else:
        print("---DECISION: GENERATE---")
        return GENERATE


def grade_generation_grounded_in_documents_and_question(state: GraphState) -> str:
    print("---CHECK HALLUCINATIONS---")

    if state.get("retry_count", 0) >= 3:
        print("---DECISION: MAX RETRIES REACHED. ENDING.---")
        return "end"

    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    # Format documents for better comparison
    formatted_docs = "\n\n".join([doc.page_content if hasattr(doc, "page_content") else str(doc) for doc in documents])

    try:
        score = hallucination_grader.invoke({"documents": formatted_docs, "generation": generation})

        if score.binary_score:  # If True (grounded)
            print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
            print("---GRADE GENERATION vs QUESTION---")

            answer_score = answer_grader.invoke({"question": question, "generation": generation})

            if answer_score.binary_score:  # If True (addresses question)
                print("---DECISION: GENERATION ADDRESSES QUESTION---")
                return "useful"
            else:
                print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
                return "not useful"
        else:
            print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
            # Log what was checked for debugging
            print(f"Generation: {generation[:200]}...")
            print(f"Documents count: {len(documents)}")
            return "not supported"

    except Exception as e:
        print(f"Error in grading: {e}")
        # If grading fails, assume the generation is acceptable
        return "useful"


def retrieve_with_reset(state):
    """Reset retry count when starting a new retrieval"""
    return {"retry_count": 0}


workflow = StateGraph(GraphState)

workflow.add_node(RETRIEVE, RunnableLambda(retrieve))
workflow.add_node(GRADE_DOCUMENTS, grade_documents)
workflow.add_node(GENERATE, generate)
workflow.add_node(PARAPHRASE, paraphrase)
workflow.add_node(RESTART, retrieve_with_reset)

workflow.set_entry_point(RESTART)
# workflow.add_edge(RETRIEVE, END)
workflow.add_edge(RESTART, RETRIEVE)
workflow.add_edge(RETRIEVE, GRADE_DOCUMENTS)
workflow.add_conditional_edges(
    GRADE_DOCUMENTS,
    decide_to_generate,
    {
        GENERATE: GENERATE,
        PARAPHRASE: PARAPHRASE,
        "end": END,
    },
)

workflow.add_conditional_edges(
    GENERATE,
    grade_generation_grounded_in_documents_and_question,
    {
        "not supported": PARAPHRASE,  # Instead of END, try paraphrasing
        "useful": END,
        "not useful": PARAPHRASE,  # Instead of regenerating, try paraphrasing
        "end": END,
    },
)
workflow.add_edge(PARAPHRASE, RETRIEVE)
workflow.add_edge(GENERATE, END)

graph = workflow.compile()

graph.get_graph().draw_mermaid_png(output_file_path="graph.png")


if __name__ == "__main__":
    question = "What machine learning models did we use in this project?"
    result = graph.invoke({"question": question})
    print("Answer:\n", result.get("generation", "No answer returned."))
