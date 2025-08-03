# Caching Module 💾

A high-performance, dual-layer caching system designed for production RAG applications. This module provides intelligent caching with automatic failover between Redis and in-memory storage.

## 🌟 Features

- **Dual-Layer Architecture**: Redis (primary) + In-Memory (fallback)
- **Automatic Failover**: Seamless operation when Redis is unavailable
- **TTL Support**: Configurable time-to-live for cache entries
- **Memory Management**: Automatic cleanup and size limiting
- **Performance Monitoring**: Built-in statistics and metrics
- **Thread-Safe**: Safe for concurrent access

## 🏗️ Architecture

```
┌─────────────┐     ┌─────────────┐
│   Client    │────▶│ QueryCache  │
└─────────────┘     └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │ Redis Check │
                    └──────┬──────┘
                           │
                ┌──────────┴──────────┐
                ▼                     ▼
        ┌──────────────┐      ┌──────────────┐
        │    Redis     │      │   Memory     │
        │   Storage    │      │   Storage    │
        └──────────────┘      └──────────────┘
```

## 📦 Usage

### Basic Usage

```python
from genai_docs_helper.cache import QueryCache

# Initialize cache
cache = QueryCache(
    redis_url="redis://localhost:6379",
    ttl=3600,  # 1 hour
    enable_redis=True
)

# Store data
cache.set("What is ML?", "context1", {
    "answer": "Machine Learning is...",
    "confidence": 0.95
})

# Retrieve data
result = cache.get("What is ML?", "context1")
print(result["answer"])
```

### Development Mode (Memory-Only)

```python
# For development without Redis
cache = QueryCache(enable_redis=False)

# Works exactly the same, using memory storage
cache.set("question", "", {"data": "answer"})
```

### Advanced Features

```python
# Get cache statistics
stats = cache.get_stats()
print(f"Hit rate: {stats['hit_rate']:.2%}")
print(f"Total requests: {stats['total_requests']}")

# Clear cache
cache.clear()

# Invalidate by pattern
cache.invalidate_pattern("machine learning")
```

## 🔧 Configuration

### Environment Variables

```bash
# Redis configuration
REDIS_URL=redis://localhost:6379
ENABLE_REDIS=false  # Set to true for production

# Cache settings
CACHE_TTL=3600  # seconds
MAX_MEMORY_ENTRIES=1000
```

### Initialization Options

```python
cache = QueryCache(
    redis_url="redis://localhost:6379",  # Redis connection
    ttl=3600,                            # Time-to-live in seconds
    max_memory_entries=1000,             # Max in-memory entries
    enable_redis=True                    # Enable Redis layer
)
```

## 🏃 Standalone Testing

Run cache tests independently:

```python
# Test cache functionality
python -m genai_docs_helper.cache.query_cache

# Performance test
python -c "
from genai_docs_helper.cache import QueryCache
import time

cache = QueryCache(enable_redis=False)

# Benchmark
start = time.time()
for i in range(1000):
    cache.set(f'q{i}', '', {'data': i})
    cache.get(f'q{i}', '')
    
print(f'1000 operations in {time.time()-start:.2f}s')
print(cache.get_stats())
"
```

## 📊 Performance Characteristics

| Operation | Redis Enabled | Memory Only |
|-----------|--------------|-------------|
| Set | ~1ms | ~0.01ms |
| Get (hit) | ~1ms | ~0.01ms |
| Get (miss) | ~1ms | ~0.01ms |
| Clear | ~10ms | ~0.1ms |

## 🔍 Cache Key Strategy

Keys are generated using MD5 hashing:

```python
key = md5(f"{question}|{context}|v1.0").hexdigest()
```

This ensures:
- Consistent keys for same inputs
- No collision between different questions
- Version control for cache invalidation

## 🛡️ Error Handling

The cache gracefully handles failures:

1. **Redis Connection Failure**: Automatically falls back to memory
2. **Memory Limit Reached**: LRU eviction of old entries
3. **Serialization Errors**: Logs error and continues operation

## 📈 Monitoring

Built-in monitoring capabilities:

```python
# Real-time statistics
stats = cache.get_stats()
{
    "hits": 150,
    "misses": 50,
    "total_requests": 200,
    "hit_rate": 0.75,
    "redis_errors": 2,
    "memory_cleanups": 1,
    "memory_cache_size": 145,
    "redis_available": True
}
```

## 🧪 Testing

Run cache-specific tests:

```bash
# Unit tests
pytest tests/unit/cache/test_query_cache.py -v

# Integration test with Redis
docker run -d -p 6379:6379 redis:alpine
pytest tests/unit/cache/test_query_cache.py -v -m requires_redis
```

## 🔧 Troubleshooting

### Redis Connection Issues

```python
# Check Redis connectivity
cache = QueryCache(redis_url="redis://localhost:6379", enable_redis=True)
if not cache.redis_client:
    print("Redis not available, using memory cache")
```

### Memory Usage

```python
# Monitor memory usage
print(f"Entries in memory: {len(cache.memory_cache)}")
print(f"Max entries: {cache.max_memory_entries}")
```

### Cache Misses

```python
# Debug cache misses
import logging
logging.getLogger("genai_docs_helper.cache").setLevel(logging.DEBUG)
```

## 🚀 Best Practices

1. **Enable Redis in Production**: Better persistence and shared cache
2. **Set Appropriate TTL**: Balance freshness vs performance
3. **Monitor Hit Rate**: Aim for >70% hit rate
4. **Regular Cleanup**: Clear cache when updating document store
5. **Use Context Wisely**: Include relevant context for better key differentiation
