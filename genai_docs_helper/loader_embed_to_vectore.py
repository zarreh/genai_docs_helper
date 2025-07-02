# Required imports
import glob
import logging
from typing import Dict, List, Literal

from dotenv import find_dotenv, load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (DirectoryLoader,
                                                  NotebookLoader,
                                                  UnstructuredMarkdownLoader)
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
from tqdm import tqdm

# from genai_docs_helper.config import LLM_TYPE

logging.basicConfig(level=logging.INFO)


LLM_TYPE = Literal['openai']  # use the openai embedding only for now
EMBEDDING = (
    OllamaEmbeddings(
        model="llama3.2",
        base_url="http://localhost:11434",
        # verbose=True
    )
    if LLM_TYPE == "ollama"
    else OpenAIEmbeddings()
)

# Load environment variables
load_dotenv(find_dotenv())


def load_markdown_files(directory: str = "./data/docs/") -> List:
    """Load markdown files from directory"""
    loader = DirectoryLoader(directory, glob="**/*.md", loader_cls=UnstructuredMarkdownLoader)
    return loader.load()


def load_jupyter_notebooks(directory: str = "./data/demand_forecast_notebooks/") -> List:
    """Load jupyter notebooks from directory"""
    documents = []
    for notebook_path in tqdm(glob.glob(f"{directory}/**/*.ipynb", recursive=True)):
        # print(f"Path: {notebook_path}")
        if ".ipynb_checkpoints" not in notebook_path:
            try:
                loader = NotebookLoader(notebook_path, include_outputs=True, max_output_length=50, remove_newline=True)
                documents.extend(loader.load())
            except Exception as e:
                print(f"Error loading notebook {notebook_path}: {e}")
    return documents


def process_documents(documents: List, chunk_size: int = 2000, chunk_overlap: int = 20):
    """Split documents into chunks"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n## ", "\n### ", "\n#### ", "\n", " ", ""],
    )
    splited_documents = text_splitter.split_documents(documents)
    print(f"Total documents after splitting: {len(splited_documents)}")
    return splited_documents


def create_vector_store(documents: List, persist_directory: str = "./data/chroma_db"):
    """Create and persist vector store"""
    print("Persisting vector store to disk...")
    # embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(documents=documents, embedding=EMBEDDING, persist_directory=persist_directory)
    print("Persisting vector store to disk completed.")
    return vectorstore


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
    print("Load markdown files")
    markdown_docs = load_markdown_files()

    print("Load Jupyter notebooks")
    notebook_docs = load_jupyter_notebooks()

    print("Combine all documents")
    all_docs = markdown_docs + notebook_docs

    print("Process documents into chunks")
    processed_docs = process_documents(all_docs)

    print("Create and persist vector store")
    vector_store = create_vector_store(processed_docs)

    print(f"Vector store created with {len(vector_store)} documents.")
