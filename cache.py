import redis
import json
import hashlib


redis_client = redis.Redis(

    host="localhost",

    port=6379,

    decode_responses=True
)


# ============================================
# HASH KEY
# ============================================

def hash_query(
    query
):

    return hashlib.md5(

        query.encode()

    ).hexdigest()


# ============================================
# GET CACHE
# ============================================

def get_cached_response(
    query
):

    key = hash_query(query)

    cached = redis_client.get(key)

    if cached:

        return json.loads(cached)

    return None


# ============================================
# SET CACHE
# ============================================

def set_cached_response(

    query,

    response,

    ttl=3600
):

    key = hash_query(query)

    redis_client.setex(

        key,

        ttl,

        json.dumps(response)
    )