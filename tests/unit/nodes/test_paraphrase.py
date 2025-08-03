"""Unit tests for paraphrase node."""
from unittest.mock import patch

import pytest

from genai_docs_helper.nodes.paraphrase import paraphrase


class TestParaphrase:
    """Test paraphrase node functionality."""

    @patch('genai_docs_helper.nodes.paraphrase.paraphraser_chain')
    def test_paraphrase_success(self, mock_chain, sample_state):
        """Test successful paraphrasing."""
        mock_chain.invoke.return_value = "Can you explain the machine learning models utilized?"
        
        result = paraphrase(sample_state)
        
        assert result["question"] != sample_state["question"]
        assert result["retry_count"] == 1
        assert result["documents"] == []  # Reset for new retrieval
        assert result["cache_key"] is None
        assert "paraphrase_time" in result["performance_metrics"]

    @patch('genai_docs_helper.nodes.paraphrase.paraphraser_chain')
    def test_paraphrase_max_retries(self, mock_chain, sample_state):
        """Test paraphrase at max retry count."""
        sample_state["retry_count"] = 2  # Will become 3
        mock_chain.invoke.return_value = "Rephrased question"
        
        result = paraphrase(sample_state)
        
        assert result["retry_count"] == 3
        assert "I apologize" in result["generation"]
        assert "couldn't find relevant information" in result["generation"]

    @patch('genai_docs_helper.nodes.paraphrase.paraphraser_chain')
    def test_paraphrase_error_fallback(self, mock_chain, sample_state):
        """Test fallback when paraphrasing fails."""
        mock_chain.invoke.side_effect = Exception("Paraphrase failed")
        sample_state["question"] = "What is machine learning?"
        
        result = paraphrase(sample_state)
        
        # Should use fallback strategy
        assert result["question"] == "What is machine learning?"  # Starts with "what"
        assert len(result["error_log"]) > 0
        assert "Paraphrase error" in result["error_log"][0]

    @patch('genai_docs_helper.nodes.paraphrase.paraphraser_chain')
    def test_paraphrase_fallback_transformation(self, mock_chain, sample_state):
        """Test fallback transformation for non-question words."""
        mock_chain.invoke.side_effect = Exception("Paraphrase failed")
        sample_state["question"] = "Machine learning models"
        
        result = paraphrase(sample_state)
        
        assert result["question"] == "Can you explain machine learning models?"

    def test_paraphrase_preserves_state(self, sample_state):
        """Test that paraphrase preserves important state."""
        sample_state["original_question"] = "Original question"
        sample_state["history"] = ["previous answer"]
        
        with patch('genai_docs_helper.nodes.paraphrase.paraphraser_chain') as mock_chain:
            mock_chain.invoke.return_value = "Rephrased"
            result = paraphrase(sample_state)
        
        assert result["original_question"] == "Original question"
        assert result["history"] == ["previous answer"]
