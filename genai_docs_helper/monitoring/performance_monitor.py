import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


class PerformanceMonitor:
    """Monitor and log performance metrics"""

    def __init__(self, log_dir: str = "./logs/performance"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.metrics: Dict[str, Any] = {}
        self.start_time: Optional[float] = None

    def start_request(self, request_id: str):
        """Start timing a request"""
        self.metrics[request_id] = {"start_time": time.time(), "stages": {}, "total_time": 0}

    def log_stage(self, request_id: str, stage: str, duration: float, metadata: Dict[str, Any] = None):
        """Log a stage completion"""
        if request_id in self.metrics:
            self.metrics[request_id]["stages"][stage] = {"duration": duration, "metadata": metadata or {}}

    def end_request(self, request_id: str):
        """End timing and save metrics"""
        if request_id in self.metrics:
            self.metrics[request_id]["total_time"] = time.time() - self.metrics[request_id]["start_time"]
            self.metrics[request_id]["timestamp"] = datetime.now().isoformat()

            # Save to log file
            log_file = self.log_dir / f"{datetime.now().strftime('%Y%m%d')}_performance.jsonl"
            with open(log_file, "a") as f:
                f.write(json.dumps(self.metrics[request_id]) + "\n")

            # Return summary
            return self.get_summary(request_id)

    def get_summary(self, request_id: str) -> Dict[str, Any]:
        """Get performance summary"""
        if request_id not in self.metrics:
            return {}

        metrics = self.metrics[request_id]
        return {
            "total_time": metrics["total_time"],
            "stages": {stage: data["duration"] for stage, data in metrics["stages"].items()},
            "bottlenecks": self._identify_bottlenecks(metrics),
        }

    def _identify_bottlenecks(self, metrics: Dict[str, Any]) -> list:
        """Identify performance bottlenecks"""
        stages = metrics["stages"]
        if not stages:
            return []

        # Find stages taking more than 30% of total time
        total_time = metrics["total_time"]
        bottlenecks = [stage for stage, data in stages.items() if data["duration"] > total_time * 0.3]

        return bottlenecks
