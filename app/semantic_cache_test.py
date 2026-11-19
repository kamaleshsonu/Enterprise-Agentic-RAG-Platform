
from app.semantic_cache import RedisSemanticCache


def main():

    print("=" * 60)
    print("REDIS SEMANTIC CACHE TEST")
    print("=" * 60)

    # Use a separate namespace for testing.
    # This prevents changes to your actual knowledge-base cache.
    cache = RedisSemanticCache(
        namespace="semantic-cache-test",
        similarity_threshold=0.96,
        ttl_seconds=300,
    )

    try:
        # 1. Test Redis connection
        print("\nTEST 1: Redis connection")

        print("Connected:", cache.ping())

        # 2. Create sample embeddings
        # These are dummy 3-dimensional vectors,
        # not actual Vertex AI embeddings.
        original_embedding = [
            1.0, 0.0, 0.0
        ]

        similar_embedding = [
            0.99, 0.01, 0.0
        ]

        unrelated_embedding = [
            0.0, 1.0, 0.0
        ]

        # 3. Store an answer
        print("\nTEST 2: Store cache entry")

        cache.store(
            question="What is HPA?",
            answer=(
                "HPA stands for Horizontal Pod Autoscaling. "
                "It automatically scales Kubernetes pods "
                "based on observed metrics."
            ),
            query_embedding=original_embedding,
        )

        print("Cache entry stored successfully.")

        # 4. Search using a similar question embedding
        print("\nTEST 3: Semantic cache hit")

        result = cache.find_similar(
            similar_embedding
        )

        if result:
            print("Cache HIT!")
            print("Matched question:", result["question"])
            print("Similarity:", result["similarity"])
            print("Cached answer:", result["answer"])
        else:
            print("Cache MISS!")

        # 5. Search using an unrelated embedding
        print("\nTEST 4: Semantic cache miss")

        result = cache.find_similar(
            unrelated_embedding
        )

        if result is None:
            print("Cache MISS! Expected result.")
        else:
            print("Unexpected cache HIT:", result)

        # 6. Clean up test records
        print("\nTEST 5: Clear test namespace")

        deleted = cache.clear_namespace()

        print("Deleted test entries:", deleted)

        print("\nAll semantic cache tests completed.")

    except Exception as error:
        print("Test failed:", error)
        raise

    finally:
        cache.close()


if __name__ == "__main__":
    main()
