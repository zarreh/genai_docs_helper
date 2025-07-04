from dotenv import load_dotenv
from langchain_core.runnables import RunnableLambda
from langgraph.graph import END, StateGraph

from genai_docs_helper.chains.answer_grader import answer_grader
from genai_docs_helper.chains.hallucination_grader import hallucination_grader
from genai_docs_helper.consts import (GENERATE, GRADE_DOCUMENTS, PARAPHRASE,
                                      RETRIEVE)
from genai_docs_helper.nodes import generate, grade_documents, retrieve
from genai_docs_helper.nodes.paraphrase import paraphrase
from genai_docs_helper.state import GraphState

load_dotenv()


def decide_to_generate(state):
    print("---ASSESS GRADED DOCUMENTS---")

    # if state.get("retry_count", 0) >= 3:
    #     print("---DECISION: MAX RETRIES REACHED. ENDING.---")
    #     return "end"

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

    # if len(documents) == 0:
    #     print("---DECISION: NO DOCUMENTS TO GRADE AGAINST. RE-TRY---")
    #     # state["generation"] = "There are no documents to grade against. Please try again with a different question."
    #     return "praphrase"

    score = hallucination_grader.invoke({"documents": documents, "generation": generation})

    if hallucination_grade := score.binary_score:
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        print("---GRADE GENERATION vs QUESTION---")
        score = answer_grader.invoke({"question": question, "generation": generation})
        if answer_grade := score.binary_score:
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
    else:
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"


workflow = StateGraph(GraphState)

workflow.add_node(RETRIEVE, RunnableLambda(retrieve))
workflow.add_node(GRADE_DOCUMENTS, grade_documents)
workflow.add_node(GENERATE, generate)
workflow.add_node(PARAPHRASE, paraphrase)

workflow.set_entry_point(RETRIEVE)
# workflow.add_edge(RETRIEVE, END)
workflow.add_edge(RETRIEVE, GRADE_DOCUMENTS)
workflow.add_conditional_edges(
    GRADE_DOCUMENTS,
    decide_to_generate,
    {
        GENERATE: GENERATE,
        PARAPHRASE: PARAPHRASE,
        # "end": END,
    },
)

workflow.add_conditional_edges(
    GENERATE,
    grade_generation_grounded_in_documents_and_question,
    {
        "not supported": PARAPHRASE,
        "useful": END,
        "not useful": GENERATE,
        # "praphrase": PARAPHRASE,
        "end": END,
    },
)
workflow.add_edge(PARAPHRASE, RETRIEVE)
workflow.add_edge(GENERATE, END)

graph = workflow.compile()

graph.get_graph().draw_mermaid_png(output_file_path="graph.png")


if __name__ == "__main__":
    question = "What are the key challenges in demand forecasting?"
    result = graph.invoke({"question": question})
    print("Answer:\n", result.get("generation", "No answer returned."))
