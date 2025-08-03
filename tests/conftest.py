"""Global test configuration and fixtures."""
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import Mock, MagicMock

import pytest
from langchain.schema import Document

from genai_docs_helper.state import GraphState


@pytest.fixture
def mock_document():
    """Create a mock document for testing."""
    doc = Mock(spec=Document)
    doc.page_content = "This is a test document about machine learning models."
    doc.metadata = {"source": "test.md", "page": 1}
    return doc


@pytest.fixture
def mock_documents(mock_document):
    """Create a list of mock documents."""
    docs = []
    for i in range(5):
        doc = Mock(spec=Document)
        doc.page_content = f"Test document {i} content about demand forecasting."
        doc.metadata = {"source": f"test{i}.md", "page": i}
        docs.append(doc)
    return docs


@pytest.fixture
def sample_state(mock_documents):
    """Create a sample GraphState for testing."""
    return {
        "question": "What machine learning models are used?",
        "original_question": "What machine learning models are used?",
        "generation": "",
        "documents": mock_documents,
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


@pytest.fixture
def mock_llm():
    """Create a mock LLM for testing."""
    llm = MagicMock()
    llm.invoke.return_value = "This is a generated response."
    return llm


@pytest.fixture
def mock_embedding():
    """Create a mock embedding function."""
    embedding = MagicMock()
    embedding.embed_query.return_value = [0.1] * 384  # Mock embedding vector
    return embedding


@pytest.fixture
def mock_vectorstore(mock_documents):
    """Create a mock vector store."""
    vectorstore = MagicMock()
    retriever = MagicMock()
    retriever.invoke.return_value = mock_documents
    vectorstore.as_retriever.return_value = retriever
    return vectorstore


@pytest.fixture
def temp_log_dir():
    """Create a temporary directory for logs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def mock_redis_client():
    """Create a mock Redis client."""
    client = MagicMock()
    client.get.return_value = None
    client.set.return_value = True
    client.setex.return_value = True
    client.ping.return_value = True
    return client


@pytest.fixture(autouse=True)
def setup_test_environment(monkeypatch):
    """Set up test environment variables."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379")
