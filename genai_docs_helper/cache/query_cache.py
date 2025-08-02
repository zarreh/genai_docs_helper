import hashlib
import pickle
import time
from typing import Any, Dict, Optional

try:
    import redis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


class QueryCache:
    """High-performance caching layer with Redis and in-memory fallback"""

    def __init__(self, redis_url: Optional[str] = None, ttl: int = 3600):
        self.ttl = ttl  # Time to live in seconds
        self.memory_cache: Dict[str, Dict[str, Any]] = {}
        self.redis_client = None

        if REDIS_AVAILABLE and redis_url:
            try:
                self.redis_client = redis.Redis.from_url(redis_url, decode_responses=False)
                self.redis_client.ping()
                print("Redis cache connected")
            except Exception as e:
                print(f"Redis connection failed, using memory cache: {e}")
                self.redis_client = None

    def _generate_key(self, question: str, context: str = "") -> str:
        """Generate cache key from question and context"""
        combined = f"{question}|{context}"
        return hashlib.md5(combined.encode()).hexdigest()

    def get(self, question: str, context: str = "") -> Optional[Dict[str, Any]]:
        """Get cached result"""
        key = self._generate_key(question, context)

        # Try Redis first
        if self.redis_client:
            try:
                cached = self.redis_client.get(f"genai:{key}")
                if cached:
                    return pickle.loads(cached)
            except Exception as e:
                print(f"Redis get error: {e}")

        # Fallback to memory cache
        if key in self.memory_cache:
            cached = self.memory_cache[key]
            if cached["expires_at"] > time.time():
                return cached["data"]
            else:
                del self.memory_cache[key]

        return None

    def set(self, question: str, context: str, data: Dict[str, Any]) -> None:
        """Set cache entry"""
        key = self._generate_key(question, context)

        # Store in Redis
        if self.redis_client:
            try:
                self.redis_client.setex(f"genai:{key}", self.ttl, pickle.dumps(data))
            except Exception as e:
                print(f"Redis set error: {e}")

        # Store in memory cache
        self.memory_cache[key] = {"data": data, "expires_at": time.time() + self.ttl}

        # Cleanup old entries from memory cache
        if len(self.memory_cache) > 1000:
            self._cleanup_memory_cache()

    def _cleanup_memory_cache(self):
        """Remove expired entries from memory cache"""
        current_time = time.time()
        self.memory_cache = {k: v for k, v in self.memory_cache.items() if v["expires_at"] > current_time}
