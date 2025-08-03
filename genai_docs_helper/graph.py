from dotenv import load_dotenv
from langchain_core.runnables import RunnableLambda
from langgraph.graph import END, StateGraph

from genai_docs_helper.chains.answer_grader import answer_grader
from genai_docs_helper.chains.hallucination_grader import hallucination_grader
from genai_docs_helper.consts import (GENERATE, GRADE_DOCUMENTS, PARAPHRASE,
                                      RESTART, RETRIEVE)
from genai_docs_helper.config import RETRY_COUNT
from genai_docs_helper.nodes import generate, grade_documents, retrieve
from genai_docs_helper.nodes.paraphrase import paraphrase
from genai_docs_helper.state import GraphState
from genai_docs_helper.utils import get_logger, setup_logging, log_performance_metrics

load_dotenv()

# Initialize logging for the main application
setup_logging(log_level="DEBUG")
logger = get_logger(__name__)


def decide_to_generate(state):
    logger.info("=== ASSESSING GRADED DOCUMENTS ===")

    if state.get("retry_count", 0) >= RETRY_COUNT:
        logger.warning("Maximum retry count reached, ending workflow")
        return "end"

    if len(state["documents"]) == 0:
        logger.info("No relevant documents found, proceeding to paraphrase")
        return PARAPHRASE
    else:
        logger.info(f"Found {len(state['documents'])} relevant documents, proceeding to generate")
        return GENERATE


def grade_generation_grounded_in_documents_and_question(state: GraphState) -> str:
    logger.info("=== CHECKING ANSWER QUALITY ===")

    if state.get("retry_count", 0) >= 3:
        logger.warning("Maximum retry count reached during grading, ending workflow")
        return "end"

    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    # Format documents for better comparison
    formatted_docs = "\n\n".join([doc.page_content if hasattr(doc, "page_content") else str(doc) for doc in documents])

    try:
        logger.debug("Checking for hallucinations in generated answer")
        score = hallucination_grader.invoke({"documents": formatted_docs, "generation": generation})

        if score.binary_score:  # If True (grounded)
            logger.info("Generation is grounded in documents, checking answer quality")

            answer_score = answer_grader.invoke({"question": question, "generation": generation})

            if answer_score.binary_score:  # If True (addresses question)
                logger.info("Generation successfully addresses the question")
                
                # Log final performance metrics
                if state.get("performance_metrics"):
                    logger.info("=== FINAL PERFORMANCE SUMMARY ===")
                    log_performance_metrics(logger, state["performance_metrics"])
                    
                return "useful"
            # !important: this is not properly handled. Need more attention
            else:
                logger.warning("Generation does not adequately address the question")
                return "not useful"
        else:
            logger.warning("Generation is not grounded in documents, needs retry")
            logger.debug(f"Generation preview: {generation[:300]}...")
            logger.debug(f"Documents count: {len(documents)}")
            return "not supported"

    except Exception as e:
        logger.error(f"Error in grading generation: {e}", exc_info=True)
        # If grading fails, assume the generation is acceptable
        return "useful"


def retrieve_with_reset(state):
    """Reset retry count when starting a new retrieval"""
    logger.info("=== STARTING NEW RETRIEVAL CYCLE ===")
    # Clear any stale cache keys to prevent contamination
    return {
        "retry_count": 0,
        "cache_key": None,
        "error_log": [],
        "performance_metrics": {},
    }


# Build the workflow
logger.info("Building document retrieval and generation workflow")

workflow = StateGraph(GraphState)

workflow.add_node(RETRIEVE, RunnableLambda(retrieve))
workflow.add_node(GRADE_DOCUMENTS, grade_documents)
workflow.add_node(GENERATE, generate)
workflow.add_node(PARAPHRASE, paraphrase)
workflow.add_node(RESTART, retrieve_with_reset)

workflow.set_entry_point(RESTART)
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
        "not useful": END,  # Instead of regenerating, try paraphrasing
        "end": END,
    },
)
workflow.add_edge(PARAPHRASE, RETRIEVE)
workflow.add_edge(GENERATE, END)

graph = workflow.compile()

logger.info("Workflow graph compiled successfully")
graph.get_graph().draw_mermaid_png(output_file_path="graph.png")


if __name__ == "__main__":
    logger.info("=== STARTING MAIN EXECUTION ===")
    question = "What machine learning models did we use in this project?"
    logger.info(f"Processing question: '{question}'")
    
    # Create a fresh state for each invocation
    initial_state = {
        "question": question,
        "original_question": question,
        "generation": "",
        "documents": [],
        "history": [],
        "retry_count": 0,
        "query_variations": None,
        "retrieved_documents_raw": None,
        "reranked_documents": None,
        "cache_key": None,
        "performance_metrics": {},
        "error_log": [],
        "confidence_score": None,
        "timestamp": None,
    }
    
    result = graph.invoke(initial_state)
    
    logger.info("=== EXECUTION COMPLETED ===")
    logger.info(f"Final answer: {result.get('generation', 'No answer returned.')}...")
    
    # Log final performance summary
    if result.get("performance_metrics"):
        logger.info("=== PERFORMANCE SUMMARY ===")
        log_performance_metrics(logger, result["performance_metrics"])
