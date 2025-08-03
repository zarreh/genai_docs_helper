"""Unit tests for performance monitoring."""
import json
import time
from pathlib import Path

import pytest

from genai_docs_helper.monitoring.performance_monitor import PerformanceMonitor


class TestPerformanceMonitor:
    """Test PerformanceMonitor functionality."""

    def test_monitor_initialization(self, temp_log_dir):
        """Test monitor initialization."""
        monitor = PerformanceMonitor(log_dir=temp_log_dir)
        
        assert Path(temp_log_dir).exists()
        assert monitor.metrics == {}

    def test_request_timing(self, temp_log_dir):
        """Test request timing functionality."""
        monitor = PerformanceMonitor(log_dir=temp_log_dir)
        
        request_id = "test-request-123"
        monitor.start_request(request_id)
        
        time.sleep(0.1)  # Simulate some work
        
        monitor.log_stage(request_id, "retrieval", 0.05, {"docs_count": 10})
        monitor.log_stage(request_id, "generation", 0.03, {"model": "gpt-4"})
        
        summary = monitor.end_request(request_id)
        
        assert summary["total_time"] >= 0.1
        assert "retrieval" in summary["stages"]
        assert "generation" in summary["stages"]

    def test_bottleneck_identification(self, temp_log_dir):
        """Test bottleneck identification."""
        monitor = PerformanceMonitor(log_dir=temp_log_dir)
        
        request_id = "test-bottleneck"
        monitor.start_request(request_id)
        
        # Create a bottleneck scenario
        monitor.metrics[request_id]["total_time"] = 1.0
        monitor.log_stage(request_id, "slow_stage", 0.7)  # 70% of time
        monitor.log_stage(request_id, "fast_stage", 0.3)  # 30% of time
        
        summary = monitor.get_summary(request_id)
        
        assert len(summary["bottlenecks"]) == 1
        assert "slow_stage" in summary["bottlenecks"][0]

    def test_log_file_creation(self, temp_log_dir):
        """Test that log files are created correctly."""
        monitor = PerformanceMonitor(log_dir=temp_log_dir)
        
        request_id = "test-logging"
        monitor.start_request(request_id)
        monitor.log_stage(request_id, "test", 0.1)
        monitor.end_request(request_id)
        
        # Check log file exists
        log_files = list(Path(temp_log_dir).glob("*_performance.jsonl"))
        assert len(log_files) == 1
        
        # Verify content
        with open(log_files[0], "r") as f:
            log_data = json.loads(f.readline())
            assert log_data["stages"]["test"]["duration"] == 0.1

    def test_empty_request_handling(self, temp_log_dir):
        """Test handling of requests with no stages."""
        monitor = PerformanceMonitor(log_dir=temp_log_dir)
        
        summary = monitor.get_summary("non-existent-request")
        assert summary == {}

    def test_metadata_storage(self, temp_log_dir):
        """Test stage metadata storage."""
        monitor = PerformanceMonitor(log_dir=temp_log_dir)
        
        request_id = "test-metadata"
        monitor.start_request(request_id)
        
        metadata = {"model": "llama3.2", "temperature": 0.7}
        monitor.log_stage(request_id, "generation", 0.5, metadata)
        
        stage_data = monitor.metrics[request_id]["stages"]["generation"]
        assert stage_data["metadata"] == metadata

    def test_concurrent_requests(self, temp_log_dir):
        """Test handling multiple concurrent requests."""
        monitor = PerformanceMonitor(log_dir=temp_log_dir)
        
        # Start multiple requests
        requests = ["req1", "req2", "req3"]
        for req_id in requests:
            monitor.start_request(req_id)
            monitor.log_stage(req_id, "stage1", 0.1)
        
        # End them
        for req_id in requests:
            summary = monitor.end_request(req_id)
            assert summary["total_time"] > 0
            assert "stage1" in summary["stages"]
