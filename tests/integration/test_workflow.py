"""Integration tests for the complete workflow."""
from unittest.mock import patch, Mock

import pytest

from genai_docs_helper.graph import graph
from genai_docs_helper.state import GraphState


class TestWorkflowIntegration:
    """Test complete workflow integration."""

    @pytest.mark.integration
    @patch('genai_docs_helper.nodes.retrieve.vectorstore')
    @patch('genai_docs_helper.chains.generation.generation_chain')
    @patch('genai_docs_helper.chains.hallucination_grader.hallucination_grader')
    @patch('genai_docs_helper.chains.answer_grader.answer_grader')
    def test_successful_workflow(self, mock_answer_grader, mock_hall_grader, mock_gen_chain, mock_vectorstore, mock_documents):
        """Test successful end-to-end workflow."""
        # Setup mocks
        retriever = Mock()
        retriever.invoke.return_value = mock_documents
        mock_vectorstore.as_retriever.return_value = retriever
        
        mock_gen_chain.invoke.return_value = "Machine learning models include ARIMA and neural networks."
        
        # Grading mocks
        mock_hall_grader.invoke.return_value = Mock(binary_score=True)
        mock_answer_grader.invoke.return_value = Mock(binary_score=True)
        
        # Run workflow
        result = graph.invoke({
            "question": "What machine learning models are used?",
            "original_question": "What machine learning models are used?"
        })
        
        assert result["generation"] != ""
        assert len(result["documents"]) > 0
        assert result["retry_count"] >= 0

    @pytest.mark.integration
    @patch('genai_docs_helper.nodes.retrieve.vectorstore')
    def test_workflow_with_no_documents(self, mock_vectorstore):
        """Test workflow when no documents are found."""
        # Setup empty retrieval
        retriever = Mock()
        retriever.invoke.return_value = []
        mock_vectorstore.as_retriever.return_value = retriever
        
        # Run workflow - should trigger paraphrase
        result = graph.invoke({
            "question": "What is XYZ123?",
            "original_question": "What is XYZ123?"
        })
        
        # Should retry and eventually give up
        assert result["retry_count"] >= 1

    @pytest.mark.integration
    @pytest.mark.slow
    def test_workflow_performance(self, mock_documents):
        """Test workflow completes within reasonable time."""
        import time
        
        with patch('genai_docs_helper.nodes.retrieve.vectorstore') as mock_vs:
            retriever = Mock()
            retriever.invoke.return_value = mock_documents
            mock_vs.as_retriever.return_value = retriever
            
            start_time = time.time()
            
            result = graph.invoke({
                "question": "Test question",
                "original_question": "Test question"
            })
            
            elapsed = time.time() - start_time
            
            # Should complete within 10 seconds (adjust based on your requirements)
            assert elapsed < 10.0
            assert "performance_metrics" in result
