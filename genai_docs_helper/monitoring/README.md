# Performance Monitoring Module 📊

A comprehensive performance monitoring system designed to track, analyze, and optimize the RAG pipeline performance in production environments.

## 🌟 Features

- **Request-Level Tracking**: Monitor individual request performance
- **Stage-Based Metrics**: Track each processing stage separately
- **Bottleneck Detection**: Automatically identify slow components
- **Persistent Logging**: JSON-based logs for analysis
- **Real-Time Analytics**: Get performance summaries on demand

## 📋 Overview

The monitoring module provides insights into:
- Response times for each processing stage
- Resource utilization patterns
- Performance bottlenecks
- Success/failure rates
- Historical performance trends

## 🏗️ Architecture

```
┌─────────────────┐
│ Request Start   │
└────────┬────────┘
         │
    ┌────▼────┐
    │ Monitor │───────► Stage Logging
    └────┬────┘
         │
    ┌────▼────┐
    │ Process │───────► Metrics Collection
    └────┬────┘
         │
    ┌────▼────┐
    │ Complete│───────► Performance Report
    └─────────┘
```

## 📦 Usage

### Basic Monitoring

```python
from genai_docs_helper.monitoring import PerformanceMonitor

# Initialize monitor
monitor = PerformanceMonitor(log_dir="./logs/performance")

# Track a request
request_id = "user-123-query-456"
monitor.start_request(request_id)

# Log stages
monitor.log_stage(request_id, "retrieval", 1.5, {
    "documents_retrieved": 50,
    "strategy": "comprehensive"
})

monitor.log_stage(request_id, "grading", 0.8, {
    "documents_graded": 50,
    "documents_kept": 20
})

monitor.log_stage(request_id, "generation", 2.1, {
    "model": "gpt-4",
    "tokens": 500
})

# Complete and get summary
summary = monitor.end_request(request_id)
print(f"Total time: {summary['total_time']:.2f}s")
print(f"Bottlenecks: {summary['bottlenecks']}")
```

### Integration with Nodes

```python
from genai_docs_helper.monitoring import PerformanceMonitor
from genai_docs_helper.utils import get_logger

logger = get_logger(__name__)
monitor = PerformanceMonitor()

def monitored_function(state, request_id):
    monitor.start_request(request_id)
    
    # Stage 1: Retrieval
    start = time.time()
    documents = retrieve_documents(state)
    monitor.log_stage(request_id, "retrieval", time.time() - start, {
        "doc_count": len(documents)
    })
    
    # Continue processing...
    return monitor.end_request(request_id)
```

## 📊 Metrics Collected

### Request Metrics
- Total request duration
- Individual stage timings
- Success/failure status
- Timestamp and request ID

### Stage Metrics
- Stage name and duration
- Custom metadata per stage
- Relative performance (% of total time)

### System Metrics
- Bottleneck identification
- Performance trends
- Resource utilization

## 📈 Log Format

Logs are stored in JSONL format for easy analysis:

```json
{
  "start_time": 1642598400.123,
  "total_time": 4.567,
  "timestamp": "2024-01-15T10:30:00",
  "stages": {
    "retrieval": {
      "duration": 1.5,
      "metadata": {
        "documents_retrieved": 50,
        "strategy": "comprehensive"
      }
    },
    "generation": {
      "duration": 2.1,
      "metadata": {
        "model": "gpt-4",
        "tokens": 500
      }
    }
  }
}
```

## 🔍 Analysis Tools

### Analyze Performance Logs

```python
import json
from pathlib import Path
from collections import defaultdict

def analyze_performance_logs(log_dir="./logs/performance"):
    stats = defaultdict(list)
    
    for log_file in Path(log_dir).glob("*_performance.jsonl"):
        with open(log_file) as f:
            for line in f:
                data = json.loads(line)
                stats['total_times'].append(data['total_time'])
                
                for stage, info in data['stages'].items():
                    stats[f'{stage}_times'].append(info['duration'])
    
    # Calculate averages
    for key, values in stats.items():
        avg = sum(values) / len(values) if values else 0
        print(f"Average {key}: {avg:.2f}s")
```

### Identify Trends

```python
# Plot performance over time
import matplotlib.pyplot as plt
from datetime import datetime

def plot_performance_trends(log_dir="./logs/performance"):
    times = []
    totals = []
    
    for log_file in Path(log_dir).glob("*_performance.jsonl"):
        with open(log_file) as f:
            for line in f:
                data = json.loads(line)
                times.append(datetime.fromisoformat(data['timestamp']))
                totals.append(data['total_time'])
    
    plt.figure(figsize=(10, 6))
    plt.plot(times, totals)
    plt.xlabel('Time')
    plt.ylabel('Response Time (s)')
    plt.title('Performance Over Time')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
```

## 🛠️ Configuration

### Log Retention

```python
# Configure in your settings
PERFORMANCE_LOG_DIR = "./logs/performance"
PERFORMANCE_LOG_RETENTION_DAYS = 30

# Cleanup old logs
from datetime import datetime, timedelta

def cleanup_old_logs(retention_days=30):
    cutoff = datetime.now() - timedelta(days=retention_days)
    log_dir = Path(PERFORMANCE_LOG_DIR)
    
    for log_file in log_dir.glob("*_performance.jsonl"):
        # Parse date from filename
        date_str = log_file.stem.split('_')[0]
        file_date = datetime.strptime(date_str, "%Y%m%d")
        
        if file_date < cutoff:
            log_file.unlink()
            print(f"Deleted old log: {log_file}")
```

## 🚨 Alerting

Set up alerts for performance degradation:

```python
def check_performance_thresholds(monitor, request_id):
    summary = monitor.get_summary(request_id)
    
    # Alert if request takes too long
    if summary['total_time'] > 10.0:
        logger.warning(f"Slow request detected: {request_id} took {summary['total_time']:.2f}s")
    
    # Alert on bottlenecks
    if summary['bottlenecks']:
        logger.warning(f"Bottlenecks detected in {request_id}: {summary['bottlenecks']}")
```

## 📊 Dashboard Integration

Export metrics for monitoring dashboards:

```python
def export_metrics_for_prometheus():
    """Export metrics in Prometheus format"""
    metrics = []
    
    # Parse recent logs
    for log_file in Path("./logs/performance").glob("*_performance.jsonl"):
        with open(log_file) as f:
            for line in f:
                data = json.loads(line)
                metrics.append(f'rag_request_duration_seconds{data["total_time"]}')
                
                for stage, info in data['stages'].items():
                    metrics.append(
                        f'rag_stage_duration_seconds{{stage="{stage}"}} {info["duration"]}'
                    )
    
    return "\n".join(metrics)
```

## 🧪 Testing

Test monitoring functionality:

```bash
# Run monitoring tests
pytest tests/unit/monitoring/test_performance_monitor.py -v

# Test with mock data
python -c "
from genai_docs_helper.monitoring import PerformanceMonitor
monitor = PerformanceMonitor('./test_logs')

# Simulate requests
for i in range(10):
    req_id = f'test-{i}'
    monitor.start_request(req_id)
    monitor.log_stage(req_id, 'retrieve', 1.0 + i*0.1)
    monitor.log_stage(req_id, 'generate', 2.0 + i*0.2)
    summary = monitor.end_request(req_id)
    print(f'Request {i}: {summary['total_time']:.2f}s')
"
```

## 🔧 Best Practices

1. **Unique Request IDs**: Use UUIDs or timestamp-based IDs
2. **Consistent Stage Names**: Use standard names across the application
3. **Meaningful Metadata**: Include relevant context in stage metadata
4. **Regular Analysis**: Review logs weekly to identify trends
5. **Set Thresholds**: Define acceptable performance limits
6. **Archive Old Logs**: Compress and archive logs older than 30 days
