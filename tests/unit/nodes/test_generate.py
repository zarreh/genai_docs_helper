"""Unit tests for generate node."""
from unittest.mock import patch, Mock

import pytest

from genai_docs_helper.nodes.generate import generate


class TestGenerate:
    """Test generate node functionality."""

    @patch('genai_docs_helper.nodes.generate.cache')
    @patch('genai_docs_helper.nodes.generate.generation_chain')
    def test_generate_success(self, mock_chain, mock_cache, sample_state):
        """Test successful generation."""
        mock_cache.get.return_value = None
        mock_chain.invoke.return_value = "Generated answer about machine learning models."
        
        result = generate(sample_state)
        
        assert result["generation"] == "Generated answer about machine learning models."
        assert len(result["history"]) == 1
        assert result["history"][0] == result["generation"]
        assert "generation_time" in result["performance_metrics"]

    @patch('genai_docs_helper.nodes.generate.cache')
    def test_generate_cache_hit(self, mock_cache, sample_state):
        """Test generation with cache hit."""
        sample_state["cache_key"] = "test_key"
        cached_result = {"generation": "Cached answer"}
        mock_cache.get.return_value = cached_result
        
        result = generate(sample_state)
        
        assert result["generation"] == "Cached answer"
        assert result["from_cache"] is True

    @patch('genai_docs_helper.nodes.generate.cache')
    @patch('genai_docs_helper.nodes.generate.generation_chain')
    def test_generate_with_document_limit(self, mock_chain, mock_cache, sample_state):
        """Test generation with document limiting."""
        # Add many documents
        sample_state["documents"] = sample_state["documents"] * 5  # 25 documents
        mock_cache.get.return_value = None
        mock_chain.invoke.return_value = "Generated answer"
        
        result = generate(sample_state)
        
        # Check that invoke was called with limited documents
        call_args = mock_chain.invoke.call_args[0][0]
        assert len(call_args["context"]) == 10  # Limited to max_docs

    @patch('genai_docs_helper.nodes.generate.cache')
    @patch('genai_docs_helper.nodes.generate.generation_chain')
    def test_generate_error_handling(self, mock_chain, mock_cache, sample_state):
        """Test generation error handling."""
        mock_cache.get.return_value = None
        mock_chain.invoke.side_effect = Exception("Generation failed")
        
        result = generate(sample_state)
        
        assert "I apologize" in result["generation"]
        assert "error" in result["generation"].lower()

    @patch('genai_docs_helper.nodes.generate.cache')
    @patch('genai_docs_helper.nodes.generate.generation_chain')
    def test_generate_caching(self, mock_chain, mock_cache, sample_state):
        """Test that successful generation is cached."""
        sample_state["cache_key"] = "test_key"
        mock_cache.get.return_value = None
        mock_chain.invoke.return_value = "Generated answer"
        
        generate(sample_state)
        
        # Verify cache.set was called
        mock_cache.set.assert_called_once()
        call_args = mock_cache.set.call_args[0]
        assert call_args[0] == sample_state["question"]
        assert "generation" in call_args[2]
