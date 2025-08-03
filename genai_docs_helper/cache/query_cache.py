import hashlib
import logging
import pickle
import time
from datetime import datetime
from typing import Any, Dict, Optional, Union

try:
    import redis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

# Configure logging
logger = logging.getLogger(__name__)


class QueryCache:
    """
    High-performance caching layer with Redis primary and in-memory fallback.

    This cache implementation provides transparent failover between Redis and
    in-memory storage, ensuring system reliability even when Redis is unavailable.
    It includes automatic cleanup, performance monitoring, and configurable TTL.

    Features:
        - Dual-layer caching (Redis + in-memory)
        - Automatic failover and recovery
        - TTL-based expiration
        - Memory usage monitoring and cleanup
        - Comprehensive error handling
        - Cache hit/miss statistics
    """

    def __init__(
        self,
        redis_url: Optional[str] = None,
        ttl: int = 3600,
        max_memory_entries: int = 1000,
        enable_redis: bool = False,
    ):
        """
        Initialize the caching system with Redis and memory backends.

        Args:
            redis_url: Redis connection string (optional)
            ttl: Time-to-live for cache entries in seconds
            max_memory_entries: Maximum entries in memory cache before cleanup
            enable_redis: Whether to attempt Redis connection (default: False for development)
        """
        self.ttl = ttl
        self.max_memory_entries = max_memory_entries
        self.memory_cache: Dict[str, Dict[str, Any]] = {}
        self.redis_client = None

        # Statistics tracking
        self.stats = {
            "hits": 0,
            "misses": 0,
            "redis_errors": 0,
            "memory_cleanups": 0,
        }

        # Only try Redis if explicitly enabled and URL provided
        if enable_redis and REDIS_AVAILABLE and redis_url:
            self._initialize_redis(redis_url)
        else:
            if not enable_redis:
                logger.info("Redis disabled by configuration, using memory-only caching")
            elif not REDIS_AVAILABLE:
                logger.info("Redis not available, using memory-only caching")
            else:
                logger.info("No Redis URL provided, using memory-only caching")

    def _initialize_redis(self, redis_url: str) -> None:
        """
        Initialize Redis connection with proper error handling.

        Args:
            redis_url: Redis connection string
        """
        try:
            self.redis_client = redis.Redis.from_url(
                redis_url, decode_responses=False, socket_connect_timeout=2, socket_timeout=2
            )
            # Test connection with timeout
            self.redis_client.ping()
            logger.info("Redis cache connected successfully")
        except Exception as e:
            logger.info(f"Redis not available ({e}), using memory cache only")
            self.redis_client = None

    def _generate_key(self, question: str, context: str = "") -> str:
        """
        Generate a deterministic cache key from question and context.

        Args:
            question: User's question
            context: Additional context for cache differentiation

        Returns:
            MD5 hash suitable for cache storage
        """
        # Include timestamp component for version control if needed
        combined = f"{question}|{context}|v1.0"
        return hashlib.md5(combined.encode()).hexdigest()

    def _get_from_redis(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Attempt to retrieve data from Redis cache.

        Args:
            key: Cache key to retrieve

        Returns:
            Cached data or None if not found/error
        """
        if not self.redis_client:
            return None

        try:
            cached = self.redis_client.get(f"genai:{key}")
            if cached:
                return pickle.loads(cached)
        except Exception as e:
            logger.warning(f"Redis get error for key {key}: {e}")
            self.stats["redis_errors"] += 1

        return None

    def _set_to_redis(self, key: str, data: Dict[str, Any]) -> bool:
        """
        Attempt to store data in Redis cache.

        Args:
            key: Cache key
            data: Data to cache

        Returns:
            True if successful, False otherwise
        """
        if not self.redis_client:
            return False

        try:
            self.redis_client.setex(f"genai:{key}", self.ttl, pickle.dumps(data))
            return True
        except Exception as e:
            logger.warning(f"Redis set error for key {key}: {e}")
            self.stats["redis_errors"] += 1
            return False

    def get(self, question: str, context: str = "") -> Optional[Dict[str, Any]]:
        """
        Retrieve cached result with fallback from Redis to memory.

        Args:
            question: User's question
            context: Additional context for cache lookup

        Returns:
            Cached result dictionary or None if not found
        """
        key = self._generate_key(question, context)

        # Try Redis first
        result = self._get_from_redis(key)
        if result:
            self.stats["hits"] += 1
            logger.debug(f"Cache hit (Redis) for key: {key[:8]}...")
            return result

        # Fallback to memory cache
        if key in self.memory_cache:
            cached = self.memory_cache[key]
            if cached["expires_at"] > time.time():
                self.stats["hits"] += 1
                logger.debug(f"Cache hit (memory) for key: {key[:8]}...")
                return cached["data"]
            else:
                # Remove expired entry
                del self.memory_cache[key]

        # Cache miss
        self.stats["misses"] += 1
        logger.debug(f"Cache miss for key: {key[:8]}...")
        return None

    def set(self, question: str, context: str, data: Dict[str, Any]) -> None:
        """
        Store data in cache with Redis primary and memory fallback.

        Args:
            question: User's question (for key generation)
            context: Additional context
            data: Data to cache
        """
        key = self._generate_key(question, context)

        # Enhance data with metadata
        cache_entry = {
            **data,
            "cached_at": datetime.now().isoformat(),
            "cache_key": key,
        }

        # Store in Redis (primary)
        redis_success = self._set_to_redis(key, cache_entry)

        # Always store in memory cache as fallback
        self.memory_cache[key] = {"data": cache_entry, "expires_at": time.time() + self.ttl}

        # Perform cleanup if memory cache is getting large
        if len(self.memory_cache) > self.max_memory_entries:
            self._cleanup_memory_cache()

        logger.debug(f"Cached result (Redis: {redis_success}, Memory: True) for key: {key[:8]}...")

    def _cleanup_memory_cache(self) -> None:
        """
        Remove expired entries and enforce size limits on memory cache.
        """
        current_time = time.time()
        initial_size = len(self.memory_cache)

        # Remove expired entries
        self.memory_cache = {k: v for k, v in self.memory_cache.items() if v["expires_at"] > current_time}

        # If still too large, remove oldest entries (LRU-style)
        if len(self.memory_cache) > self.max_memory_entries:
            items_by_expiry = sorted(self.memory_cache.items(), key=lambda x: x[1]["expires_at"])

            # Keep only the most recent entries
            keep_count = int(self.max_memory_entries * 0.8)  # Keep 80% after cleanup
            self.memory_cache = dict(items_by_expiry[-keep_count:])

        cleaned_count = initial_size - len(self.memory_cache)
        if cleaned_count > 0:
            self.stats["memory_cleanups"] += 1
            logger.info(f"Memory cache cleanup: removed {cleaned_count} entries")

    def get_stats(self) -> Dict[str, Union[int, float]]:
        """
        Get cache performance statistics.

        Returns:
            Dictionary with hit rates, error counts, and other metrics
        """
        total_requests = self.stats["hits"] + self.stats["misses"]
        hit_rate = (self.stats["hits"] / total_requests) if total_requests > 0 else 0.0

        return {
            **self.stats,
            "total_requests": total_requests,
            "hit_rate": hit_rate,
            "memory_cache_size": len(self.memory_cache),
            "redis_available": self.redis_client is not None,
        }

    def clear(self) -> None:
        """
        Clear all cached data from both Redis and memory.
        """
        # Clear memory cache
        self.memory_cache.clear()

        # Clear Redis cache (if available)
        if self.redis_client:
            try:
                # Get all keys with our prefix and delete them
                keys = self.redis_client.keys("genai:*")
                if keys:
                    self.redis_client.delete(*keys)
                logger.info("Redis cache cleared")
            except Exception as e:
                logger.warning(f"Error clearing Redis cache: {e}")

        logger.info("Cache cleared successfully")
