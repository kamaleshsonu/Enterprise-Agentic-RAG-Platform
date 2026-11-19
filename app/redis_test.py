
import redis

# Connect to the local Redis server
client = redis.Redis(
    host="localhost",
    port=6379,
    db=0,
    decode_responses=True,
    socket_connect_timeout=5
)

try:
    # Test connection
    print("Redis connection:", client.ping())

    # Store a test value for 60 seconds
    client.set(
        "enterprise_rag:test",
        "Redis connection successful",
        ex=60
    )

    # Retrieve the stored value
    result = client.get("enterprise_rag:test")

    print("Retrieved value:", result)

    # Delete the test key
    client.delete("enterprise_rag:test")

    print("Test key deleted successfully")

except redis.RedisError as error:
    print("Redis error:", error)

finally:
    client.close()
