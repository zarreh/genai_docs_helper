# Required imports
import glob
from typing import Dict, List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (DirectoryLoader,
                                                  NotebookLoader,
                                                  UnstructuredMarkdownLoader)

# Use the new langchain-chroma import
try:
    from langchain_chroma import Chroma
except ImportError:
    # Fallback to the community version if langchain-chroma is not available
    from langchain_community.vectorstores import Chroma

from tqdm import tqdm

from genai_docs_helper.config import (EMBEDDING, ORIGINAL_DOCS_PATH,
                                      VECTOR_STORE_PATH)
from genai_docs_helper.utils import get_logger, setup_logging

# Initialize logging for this script
setup_logging(log_level="INFO")
logger = get_logger(__name__)


def load_markdown_files(directory: str = "./data/docs/") -> List:
    """Load markdown files from directory"""
    logger.info(f"Loading markdown files from {directory}")
    loader = DirectoryLoader(directory, glob="**/*.md", loader_cls=UnstructuredMarkdownLoader)
    docs = loader.load()
    logger.info(f"Loaded {len(docs)} markdown documents")
    return docs


def load_jupyter_notebooks(directory: str = "./data/demand_forecast_notebooks/") -> List:
    """Load jupyter notebooks from directory"""
    logger.info(f"Loading Jupyter notebooks from {directory}")
    documents = []
    notebook_paths = list(glob.glob(f"{directory}/**/*.ipynb", recursive=True))
    
    for notebook_path in tqdm(notebook_paths, desc="Loading notebooks"):
        if ".ipynb_checkpoints" not in notebook_path:
            try:
                loader = NotebookLoader(notebook_path, include_outputs=True, max_output_length=50, remove_newline=True)
                docs = loader.load()
                documents.extend(docs)
                logger.debug(f"Successfully loaded notebook: {notebook_path}")
            except Exception as e:
                logger.error(f"Error loading notebook {notebook_path}: {e}")
                
    logger.info(f"Loaded {len(documents)} notebook documents")
    return documents


def process_documents(documents: List, chunk_size: int = 2000, chunk_overlap: int = 20):
    """Split documents into chunks"""
    logger.info(f"Processing {len(documents)} documents into chunks")
    logger.debug(f"Chunk size: {chunk_size}, Overlap: {chunk_overlap}")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n## ", "\n### ", "\n#### ", "\n", " ", ""],
    )
    splited_documents = text_splitter.split_documents(documents)
    logger.info(f"Total documents after splitting: {len(splited_documents)}")
    return splited_documents


def create_vector_store(documents: List, persist_directory: str = "./data/chroma_db"):
    """Create and persist vector store"""
    logger.info(f"Creating vector store with {len(documents)} documents")
    logger.info(f"Persist directory: {persist_directory}")
    
    try:
        vectorstore = Chroma.from_documents(
            documents=documents, embedding=EMBEDDING, persist_directory=persist_directory
        )
        logger.info("Vector store created and persisted successfully")
        return vectorstore
    except Exception as e:
        logger.error(f"Error creating vector store: {e}", exc_info=True)
        raise


def extract_relevant_context(doc) -> Dict:
    """Extract and format relevant context from a document."""
    return {
        "content": doc.page_content,
        "source": doc.metadata.get("source", "Unnamed Source"),
        "page": doc.metadata.get("page", None),
    }


def format_citation(source_info: Dict) -> str:
    """Format source information into a citation."""
    citation = source_info["source"]
    if source_info["page"]:
        citation += f", page {source_info['page']}"
    return citation


if __name__ == "__main__":
    logger.info("=== STARTING DOCUMENT LOADING AND VECTOR STORE CREATION ===")
    
    logger.info("Step 1: Loading markdown files")
    markdown_docs = load_markdown_files(directory=ORIGINAL_DOCS_PATH)

    logger.info("Step 2: Loading Jupyter notebooks")
    notebook_docs = load_jupyter_notebooks(directory="./data/demand_forecast_notebooks/")

    logger.info("Step 3: Combining all documents")
    all_docs = markdown_docs + notebook_docs
    logger.info(f"Total documents loaded: {len(all_docs)}")

    logger.info("Step 4: Processing documents into chunks")
    processed_docs = process_documents(all_docs)

    logger.info("Step 5: Creating and persisting vector store")
    vector_store = create_vector_store(processed_docs, persist_directory=VECTOR_STORE_PATH)

    logger.info(f"=== VECTOR STORE CREATION COMPLETED ===")
    logger.info(f"Vector store created with {len(processed_docs)} document chunks")
    logger.info(f"Stored at: {VECTOR_STORE_PATH}")
