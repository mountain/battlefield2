import os
import redis


REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))

print(f"Redis Host: {REDIS_HOST}")
print(f"Redis Port: {REDIS_PORT}")
print(f"Redis DB: {REDIS_DB}")

rc = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)
