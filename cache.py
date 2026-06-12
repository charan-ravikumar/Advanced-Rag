# ============================================
# IMPORTS
# ============================================

import os
import json
import hashlib
import logging

import redis

from dotenv import load_dotenv

from logger_config import get_logger


# ============================================
# LOGGER
# ============================================

logger = get_logger(__name__)


# ============================================
# LOAD ENV
# ============================================

load_dotenv()

REDIS_HOST = os.getenv(
    "REDIS_HOST", "localhost"
)

REDIS_PORT = int(
    os.getenv("REDIS_PORT", 6379)
)

REDIS_PASSWORD = os.getenv(
    "REDIS_PASSWORD", None
)

REDIS_DB = int(
    os.getenv("REDIS_DB", 0)
)

CACHE_TTL_SECONDS = int(
    os.getenv("CACHE_TTL_SECONDS", 3600)
)


# ============================================
# CONNECTION POOL
# ============================================

_pool = redis.ConnectionPool(

    host=REDIS_HOST,

    port=REDIS_PORT,

    password=REDIS_PASSWORD,

    db=REDIS_DB,

    decode_responses=True,

    max_connections=20,

    socket_connect_timeout=2,

    socket_timeout=2
)


def _get_client() -> redis.Redis:

    return redis.Redis(
        connection_pool=_pool
    )


# ============================================
# HEALTH CHECK
# ============================================

def ping_redis() -> bool:
    """
    Returns True if Redis is reachable,
    False otherwise. Never raises.
    """

    try:

        return _get_client().ping()

    except Exception:

        logger.warning(
            "Redis ping failed — "
            "cache unavailable."
        )

        return False


# ============================================
# QUERY NORMALISATION
# ============================================

def _normalise(query: str) -> str:
    """
    Lowercase + strip so that minor
    casing/whitespace differences still
    produce the same cache key.
    """

    return query.lower().strip()


# ============================================
# HASH KEY
# ============================================

def _hash_query(query: str) -> str:

    normalised = _normalise(query)

    return "rag:cache:" + hashlib.sha256(

        normalised.encode()

    ).hexdigest()


# ============================================
# GET CACHE
# ============================================

def get_cached_response(
    query: str
):
    """
    Returns the cached response string for
    *query*, or None on a miss or Redis error.
    """

    key = _hash_query(query)

    try:

        cached = _get_client().get(key)

        if cached:

            logger.info(
                f"Cache HIT for key {key[:16]}…"
            )

            return json.loads(cached)

        logger.info(
            f"Cache MISS for key {key[:16]}…"
        )

        return None

    except Exception:

        logger.warning(
            "Redis GET failed — treating as "
            "cache miss.",
            exc_info=True
        )

        return None


# ============================================
# SET CACHE
# ============================================

def set_cached_response(

    query: str,

    response: str,

    ttl: int = None
):
    """
    Stores *response* in Redis under the
    hashed *query* key. Silently swallows
    Redis errors so the caller is unaffected.
    """

    if ttl is None:
        ttl = CACHE_TTL_SECONDS

    key = _hash_query(query)

    try:

        _get_client().setex(

            key,

            ttl,

            json.dumps(response)
        )

        logger.info(
            f"Cached response stored "
            f"(ttl={ttl}s) for key {key[:16]}…"
        )

    except Exception:

        logger.warning(
            "Redis SET failed — response "
            "will not be cached.",
            exc_info=True
        )


# ============================================
# FLUSH CACHE
# ============================================

def flush_cache() -> int:
    """
    Deletes all keys matching the rag:cache:*
    pattern. Returns the number of keys deleted.
    """

    try:

        client = _get_client()

        keys = client.keys("rag:cache:*")

        if not keys:

            logger.info(
                "Flush called — no cache "
                "keys found."
            )

            return 0

        deleted = client.delete(*keys)

        logger.info(
            f"Cache flushed — "
            f"{deleted} key(s) deleted."
        )

        return deleted

    except Exception:

        logger.warning(
            "Redis FLUSH failed.",
            exc_info=True
        )

        return 0


# ============================================
# CACHE SIZE
# ============================================

def cache_size() -> int:
    """
    Returns the number of active rag:cache:*
    keys currently stored in Redis, or -1 on
    error.
    """

    try:

        return len(
            _get_client().keys("rag:cache:*")
        )

    except Exception:

        logger.warning(
            "Redis KEYS count failed.",
            exc_info=True
        )

        return -1