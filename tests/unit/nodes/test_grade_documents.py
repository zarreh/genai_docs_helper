"""Unit tests for grade_documents node."""
from unittest.mock import Mock, patch

import pytest

from genai_docs_helper.nodes.grade_documents import (
    grade_document_batch,
    assess_grading_quality,
    should_stop_early,
    grade_documents,
    GRADING_CONFIG
)


class TestGradeDocumentBatch:
    """Test batch document grading."""

    @patch('genai_docs_helper.nodes.grade_documents.batch_document_grader')
    def test_batch_grading_success(self, mock_grader, mock_documents):
        """Test successful batch grading."""
        # Mock the batch grader response
        mock_scores = Mock()
        mock_scores.scores = [
            Mock(is_relevant=True, confidence=0.9),
            Mock(is_relevant=False, confidence=0.3),
            Mock(is_relevant=True, confidence=0.8),
        ]
        mock_grader.invoke.return_value = mock_scores
        
        results = grade_document_batch(mock_documents[:3], "test question", batch_size=5)
        
        assert len(results) == 3
        assert results[0][1] is True  # First doc is relevant
        assert results[1][1] is False  # Second doc is not relevant
        assert results[0][2] == 0.9  # Confidence score

    @patch('genai_docs_helper.nodes.grade_documents.batch_document_grader')
    @patch('genai_docs_helper.nodes.grade_documents.retrieval_grader')
    def test_batch_grading_fallback(self, mock_individual_grader, mock_batch_grader, mock_documents):
        """Test fallback to individual grading."""
        # Batch grader fails
        mock_batch_grader.invoke.side_effect = Exception("Batch grading failed")
        
        # Individual grader succeeds
        mock_score = Mock()
        mock_score.binary_score = "yes"
        mock_individual_grader.invoke.return_value = mock_score
        
        results = grade_document_batch(mock_documents[:2], "test question", batch_size=5)
        
        assert len(results) == 2
        assert all(r[1] for r in results)  # All marked as relevant in fallback
        assert mock_individual_grader.invoke.call_count == 2

    def test_empty_documents(self):
        """Test batch grading with empty documents."""
        results = grade_document_batch([], "test question")
        assert results == []


class TestGradingQuality:
    """Test grading quality assessment."""

    def test_assess_grading_quality_normal(self):
        """Test quality assessment with normal results."""
        graded_results = [
            (Mock(), True, 0.9),
            (Mock(), True, 0.8),
            (Mock(), False, 0.3),
            (Mock(), True, 0.7),
        ]
        
        quality = assess_grading_quality(graded_results)
        
        assert quality["total_docs"] == 4
        assert quality["relevant_docs"] == 3
        assert quality["relevance_rate"] == 0.75
        assert quality["avg_confidence"] == pytest.approx(0.675, rel=1e-3)
        assert quality["high_confidence_docs"] == 3  # Using default threshold of 0.7

    def test_assess_grading_quality_empty(self):
        """Test quality assessment with no results."""
        quality = assess_grading_quality([])
        
        assert quality["total_docs"] == 0
        assert quality["relevant_docs"] == 0
        assert quality["avg_confidence"] == 0.0


class TestEarlyStopping:
    """Test early stopping logic."""

    def test_should_stop_early_sufficient_docs(self):
        """Test early stopping with sufficient high-confidence docs."""
        relevant_docs = [Mock() for _ in range(12)]
        confidence_scores = [0.8] * 6 + [0.5] * 6
        
        with patch.dict(GRADING_CONFIG, {"early_stopping_enabled": True, "min_relevant_docs": 5}):
            assert should_stop_early(relevant_docs, confidence_scores) is True

    def test_should_stop_early_insufficient_confidence(self):
        """Test no early stopping with low confidence."""
        relevant_docs = [Mock() for _ in range(12)]
        confidence_scores = [0.5] * 12  # All below threshold
        
        with patch.dict(GRADING_CONFIG, {"early_stopping_enabled": True}):
            assert should_stop_early(relevant_docs, confidence_scores) is False

    def test_should_stop_early_disabled(self):
        """Test early stopping when disabled."""
        relevant_docs = [Mock() for _ in range(20)]
        confidence_scores = [0.9] * 20
        
        with patch.dict(GRADING_CONFIG, {"early_stopping_enabled": False}):
            assert should_stop_early(relevant_docs, confidence_scores) is False


class TestGradeDocuments:
    """Test main grade_documents function."""

    def test_grade_documents_empty_input(self, sample_state):
        """Test grading with no documents."""
        sample_state["documents"] = []
        
        result = grade_documents(sample_state)
        
        assert result["documents"] == []
        assert result["confidence_score"] == 0.0
        assert result["performance_metrics"]["documents_graded"] == 0

    @patch('genai_docs_helper.nodes.grade_documents.grade_document_batch')
    def test_grade_documents_success(self, mock_grade_batch, sample_state, mock_documents):
        """Test successful document grading."""
        # Mock grading results
        graded_results = [
            (mock_documents[0], True, 0.9),
            (mock_documents[1], True, 0.8),
            (mock_documents[2], False, 0.3),
            (mock_documents[3], True, 0.7),
            (mock_documents[4], False, 0.2),
        ]
        mock_grade_batch.return_value = graded_results
        
        result = grade_documents(sample_state)
        
        assert len(result["documents"]) == 3  # Only relevant docs
        assert result["confidence_score"] == pytest.approx(0.8, rel=1e-3)
        assert result["performance_metrics"]["documents_filtered"] == 3
        assert result["performance_metrics"]["relevance_rate"] == 0.6

    @patch('genai_docs_helper.nodes.grade_documents.grade_document_batch')
    def test_grade_documents_with_early_stopping(self, mock_grade_batch, sample_state, mock_documents):
        """Test grading with early stopping."""
        # Create many high-confidence results
        graded_results = [(mock_documents[i % 5], True, 0.9) for i in range(15)]
        mock_grade_batch.return_value = graded_results
        
        with patch.dict(GRADING_CONFIG, {"early_stopping_enabled": True, "min_relevant_docs": 5}):
            result = grade_documents(sample_state)
        
        assert result["performance_metrics"]["early_stopped"] is True

    @patch('genai_docs_helper.nodes.grade_documents.grade_document_batch')
    def test_grade_documents_error_fallback(self, mock_grade_batch, sample_state):
        """Test fallback when grading fails."""
        mock_grade_batch.side_effect = Exception("Grading failed")
        
        result = grade_documents(sample_state)
        
        assert len(result["documents"]) <= 15  # Fallback limit
        assert result["confidence_score"] == 0.3  # Low confidence
        assert result["performance_metrics"]["grading_strategy"] == "fallback"
        assert len(result["error_log"]) > 0
