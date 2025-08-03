"""Unit tests for the retrieve node."""
import time
from unittest.mock import Mock, patch, MagicMock

import pytest
import numpy as np

from genai_docs_helper.nodes.retrieve import (
    compute_semantic_similarity,
    fast_semantic_rerank,
    parallel_retrieve,
    generate_cache_key,
    execute_fast_retrieval_path,
    execute_comprehensive_retrieval_path,
    retrieve,
)


class TestSemanticSimilarity:
    """Test semantic similarity computation."""

    def test_compute_semantic_similarity_normal(self):
        """Test normal similarity computation."""
        query_embedding = [0.5, 0.5, 0.0]
        doc_embedding = [0.5, 0.5, 0.0]
        
        # Normalize vectors
        query_norm = np.array(query_embedding) / np.linalg.norm(query_embedding)
        doc_norm = np.array(doc_embedding) / np.linalg.norm(doc_embedding)
        
        similarity = compute_semantic_similarity(query_norm.tolist(), doc_norm.tolist())
        assert similarity == pytest.approx(1.0, rel=1e-5)

    def test_compute_semantic_similarity_orthogonal(self):
        """Test similarity for orthogonal vectors."""
        query_embedding = [1.0, 0.0]
        doc_embedding = [0.0, 1.0]
        
        similarity = compute_semantic_similarity(query_embedding, doc_embedding)
        assert similarity == pytest.approx(0.0, rel=1e-5)

    def test_compute_semantic_similarity_error(self):
        """Test similarity computation with error."""
        similarity = compute_semantic_similarity([1.0], [1.0, 2.0])  # Mismatched dimensions
        assert similarity == 0.0


class TestFastSemanticRerank:
    """Test fast semantic reranking."""

    @patch('genai_docs_helper.nodes.retrieve.EMBEDDING')
    def test_rerank_documents(self, mock_embedding, mock_documents):
        """Test document reranking."""
        # Mock embeddings
        mock_embedding.embed_query.side_effect = [
            [0.5, 0.5],  # Query embedding
            [0.4, 0.6],  # Doc 1
            [0.5, 0.5],  # Doc 2 (most similar)
            [0.1, 0.9],  # Doc 3
            [0.2, 0.8],  # Doc 4
            [0.3, 0.7],  # Doc 5
        ]
        
        reranked = fast_semantic_rerank("test question", mock_documents, top_k=3)
        
        assert len(reranked) == 3
        mock_embedding.embed_query.assert_called()

    def test_rerank_empty_documents(self):
        """Test reranking with empty document list."""
        reranked = fast_semantic_rerank("test question", [], top_k=5)
        assert reranked == []

    def test_rerank_fewer_documents_than_topk(self, mock_documents):
        """Test reranking when documents < top_k."""
        docs = mock_documents[:2]
        reranked = fast_semantic_rerank("test question", docs, top_k=5)
        assert reranked == docs


class TestParallelRetrieve:
    """Test parallel retrieval functionality."""

    @patch('genai_docs_helper.nodes.retrieve.standard_retriever')
    def test_parallel_retrieve_success(self, mock_retriever, mock_documents):
        """Test successful parallel retrieval."""
        mock_retriever.invoke.return_value = mock_documents[:2]
        
        queries = ["query1", "query2", "query3"]
        results = parallel_retrieve(queries, timeout_per_query=5)
        
        assert len(results) > 0
        assert mock_retriever.invoke.call_count == len(queries)

    def test_parallel_retrieve_empty_queries(self):
        """Test parallel retrieve with empty queries."""
        results = parallel_retrieve([])
        assert results == []

    @patch('genai_docs_helper.nodes.retrieve.standard_retriever')
    def test_parallel_retrieve_with_duplicates(self, mock_retriever, mock_documents):
        """Test deduplication in parallel retrieve."""
        # Return same documents for all queries
        mock_retriever.invoke.return_value = mock_documents[:2]
        
        queries = ["query1", "query2"]
        results = parallel_retrieve(queries)
        
        # Should deduplicate
        assert len(results) <= len(mock_documents[:2])


class TestCacheKey:
    """Test cache key generation."""

    def test_generate_cache_key_consistency(self):
        """Test that cache key is consistent for same input."""
        question = "What is machine learning?"
        key1 = generate_cache_key(question, "standard")
        key2 = generate_cache_key(question, "standard")
        assert key1 == key2

    def test_generate_cache_key_different_strategies(self):
        """Test different strategies produce different keys."""
        question = "What is machine learning?"
        key1 = generate_cache_key(question, "fast")
        key2 = generate_cache_key(question, "comprehensive")
        assert key1 != key2


class TestRetrievalPaths:
    """Test different retrieval paths."""

    @patch('genai_docs_helper.nodes.retrieve.fast_retriever')
    @patch('genai_docs_helper.nodes.retrieve.fast_semantic_rerank')
    def test_fast_retrieval_path_success(self, mock_rerank, mock_retriever, mock_documents):
        """Test successful fast retrieval path."""
        mock_retriever.invoke.return_value = mock_documents * 3  # 15 documents
        mock_rerank.return_value = mock_documents[:3]
        
        result = execute_fast_retrieval_path("test question", "original question")
        
        assert result is not None
        assert "documents" in result
        assert result["performance_metrics"]["retrieval_strategy"] == "fast"

    @patch('genai_docs_helper.nodes.retrieve.fast_retriever')
    def test_fast_retrieval_path_insufficient_docs(self, mock_retriever):
        """Test fast path with insufficient documents."""
        mock_retriever.invoke.return_value = []  # No documents
        
        result = execute_fast_retrieval_path("test question", "original question")
        assert result is None

    @patch('genai_docs_helper.nodes.retrieve.query_expander_chain')
    @patch('genai_docs_helper.nodes.retrieve.parallel_retrieve')
    @patch('genai_docs_helper.nodes.retrieve.fast_semantic_rerank')
    def test_comprehensive_retrieval_path(self, mock_rerank, mock_parallel, mock_expander, mock_documents):
        """Test comprehensive retrieval path."""
        mock_expander.invoke.return_value = ["variation1", "variation2"]
        mock_parallel.return_value = mock_documents * 2
        mock_rerank.return_value = mock_documents
        
        result = execute_comprehensive_retrieval_path("test question", "original question")
        
        assert "documents" in result
        assert "query_variations" in result
        assert result["performance_metrics"]["retrieval_strategy"] == "comprehensive"


class TestMainRetrieve:
    """Test main retrieve function."""

    @patch('genai_docs_helper.nodes.retrieve.cache')
    def test_retrieve_cache_hit(self, mock_cache, sample_state, mock_documents):
        """Test retrieve with cache hit."""
        cached_result = {
            "documents": mock_documents,
            "question": "test",
            "performance_metrics": {"cached": True}
        }
        mock_cache.get.return_value = cached_result
        
        result = retrieve(sample_state)
        
        assert result["from_cache"] is True
        assert "documents" in result

    @patch('genai_docs_helper.nodes.retrieve.cache')
    @patch('genai_docs_helper.nodes.retrieve.execute_fast_retrieval_path')
    def test_retrieve_fast_path_success(self, mock_fast_path, mock_cache, sample_state, mock_documents):
        """Test retrieve using fast path."""
        mock_cache.get.return_value = None
        mock_fast_path.return_value = {
            "documents": mock_documents,
            "performance_metrics": {"strategy": "fast"}
        }
        
        result = retrieve(sample_state)
        
        assert "documents" in result
        assert not result.get("from_cache", False)

    @patch('genai_docs_helper.nodes.retrieve.cache')
    @patch('genai_docs_helper.nodes.retrieve.execute_fast_retrieval_path')
    @patch('genai_docs_helper.nodes.retrieve.execute_comprehensive_retrieval_path')
    def test_retrieve_comprehensive_fallback(self, mock_comp_path, mock_fast_path, mock_cache, sample_state):
        """Test fallback to comprehensive path."""
        mock_cache.get.return_value = None
        mock_fast_path.return_value = None  # Fast path fails
        mock_comp_path.return_value = {
            "documents": [],
            "performance_metrics": {"strategy": "comprehensive"}
        }
        
        result = retrieve(sample_state)
        
        assert mock_comp_path.called
        assert result["performance_metrics"]["strategy"] == "comprehensive"

    @patch('genai_docs_helper.nodes.retrieve.cache')
    @patch('genai_docs_helper.nodes.retrieve.execute_fast_retrieval_path')
    @patch('genai_docs_helper.nodes.retrieve.execute_comprehensive_retrieval_path')
    @patch('genai_docs_helper.nodes.retrieve.fast_retriever')
    def test_retrieve_fallback_on_error(self, mock_retriever, mock_comp_path, mock_fast_path, mock_cache, sample_state):
        """Test fallback when all strategies fail."""
        mock_cache.get.return_value = None
        mock_fast_path.side_effect = Exception("Fast path error")
        mock_comp_path.side_effect = Exception("Comprehensive path error")
        mock_retriever.invoke.return_value = []
        
        result = retrieve(sample_state)
        
        assert result["documents"] == []
        assert result["performance_metrics"]["retrieval_strategy"] == "fallback"
        assert len(result["error_log"]) > 0
