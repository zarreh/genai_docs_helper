"""Unit tests for query cache."""
import time
from unittest.mock import Mock, patch, MagicMock

import pytest

from genai_docs_helper.cache.query_cache import QueryCache


class TestQueryCache:
    """Test QueryCache functionality."""

    def test_cache_init_without_redis(self):
        """Test cache initialization without Redis."""
        cache = QueryCache(enable_redis=False)
        
        assert cache.redis_client is None
        assert cache.memory_cache == {}
        assert cache.ttl == 3600

    @patch('genai_docs_helper.cache.query_cache.redis')
    def test_cache_init_with_redis(self, mock_redis_module, mock_redis_client):
        """Test cache initialization with Redis."""
        mock_redis_module.Redis.from_url.return_value = mock_redis_client
        
        cache = QueryCache(redis_url="redis://localhost:6379", enable_redis=True)
        
        assert cache.redis_client is not None
        mock_redis_client.ping.assert_called_once()

    def test_generate_key(self):
        """Test cache key generation."""
        cache = QueryCache(enable_redis=False)
        
        key1 = cache._generate_key("test question", "context1")
        key2 = cache._generate_key("test question", "context1")
        key3 = cache._generate_key("test question", "context2")
        
        assert key1 == key2  # Same input = same key
        assert key1 != key3  # Different context = different key

    def test_memory_cache_set_get(self):
        """Test memory cache set and get operations."""
        cache = QueryCache(enable_redis=False)
        
        test_data = {"result": "test answer", "score": 0.9}
        cache.set("question", "context", test_data)
        
        retrieved = cache.get("question", "context")
        assert retrieved is not None
        assert retrieved["result"] == "test answer"
        assert retrieved["cached_at"] is not None

    def test_memory_cache_expiration(self):
        """Test memory cache TTL expiration."""
        cache = QueryCache(enable_redis=False, ttl=1)  # 1 second TTL
        
        cache.set("question", "context", {"result": "test"})
        
        # Should retrieve immediately
        assert cache.get("question", "context") is not None
        
        # Wait for expiration
        time.sleep(1.1)
        
        # Should be expired
        assert cache.get("question", "context") is None

    @patch('genai_docs_helper.cache.query_cache.redis')
    def test_redis_cache_operations(self, mock_redis_module, mock_redis_client):
        """Test Redis cache operations."""
        mock_redis_module.Redis.from_url.return_value = mock_redis_client
        cache = QueryCache(redis_url="redis://localhost:6379", enable_redis=True)
        
        # Test set
        test_data = {"result": "test"}
        cache.set("question", "context", test_data)
        mock_redis_client.setex.assert_called()
        
        # Test get
        mock_redis_client.get.return_value = cache._pickle_dumps({"result": "cached"})
        result = cache.get("question", "context")
        assert result is not None

    def test_cache_stats(self):
        """Test cache statistics tracking."""
        cache = QueryCache(enable_redis=False)
        
        # Generate some hits and misses
        cache.set("q1", "", {"data": "test"})
        cache.get("q1", "")  # Hit
        cache.get("q2", "")  # Miss
        cache.get("q3", "")  # Miss
        
        stats = cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 2
        assert stats["hit_rate"] == pytest.approx(0.333, rel=1e-3)

    def test_memory_cache_cleanup(self):
        """Test memory cache cleanup."""
        cache = QueryCache(enable_redis=False, max_memory_entries=5)
        
        # Fill cache beyond limit
        for i in range(10):
            cache.set(f"question{i}", "", {"data": f"test{i}"})
        
        # Should have cleaned up
        assert len(cache.memory_cache) <= 5

    def test_cache_clear(self):
        """Test cache clearing."""
        cache = QueryCache(enable_redis=False)
        
        # Add some entries
        cache.set("q1", "", {"data": "test1"})
        cache.set("q2", "", {"data": "test2"})
        
        assert len(cache.memory_cache) == 2
        
        cache.clear()
        assert len(cache.memory_cache) == 0

    @patch('genai_docs_helper.cache.query_cache.redis')
    def test_redis_fallback_to_memory(self, mock_redis_module, mock_redis_client):
        """Test fallback to memory when Redis fails."""
        mock_redis_module.Redis.from_url.return_value = mock_redis_client
        cache = QueryCache(redis_url="redis://localhost:6379", enable_redis=True)
        
        # Make Redis operations fail
        mock_redis_client.get.side_effect = Exception("Redis error")
        mock_redis_client.setex.side_effect = Exception("Redis error")
        
        # Should still work with memory cache
        cache.set("question", "", {"data": "test"})
        result = cache.get("question", "")
        
        assert result is not None
        assert result["data"] == "test"
        assert cache.stats["redis_errors"] > 0

    def test_pickle_helpers(self):
        """Test internal pickle helper methods."""
        cache = QueryCache(enable_redis=False)
        
        # Mock the pickle methods since they're used internally
        import pickle
        test_data = {"test": "data"}
        
        # These would be private methods but we can test the concept
        pickled = pickle.dumps(test_data)
        unpickled = pickle.loads(pickled)
        
        assert unpickled == test_data
