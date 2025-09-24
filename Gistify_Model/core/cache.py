import redis
import json
from typing import Optional

# Connect to Redis
redis_conn = redis.Redis(decode_responses=True)

CACHE_EXPIRATION_SECONDS = 3600  # 1 hour

def get_from_cache(key: str) -> Optional[dict]:
    """
    Gets a value from the Redis cache.
    """
    cached_value = redis_conn.get(key)
    if cached_value:
        return json.loads(cached_value)
    return None

def set_to_cache(key: str, value: dict):
    """
    Sets a value in the Redis cache with an expiration time.
    """
    redis_conn.setex(
        key,
        CACHE_EXPIRATION_SECONDS,
        json.dumps(value)
    )
