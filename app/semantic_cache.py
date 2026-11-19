
import json
import math
import time
import uuid

import redis


class RedisSemanticCache:
    """
    Redis-backed semantic cache.

    Stores question embeddings and validated answers.
    Uses Python cosine similarity to find similar questions.
    """

    def __init__(
        self,
        redis_url="redis://127.0.0.1:6379/0",
        namespace="kubernetes-v1",
        similarity_threshold=0.96,
        ttl_seconds=3600,
    ):
        self.redis = redis.Redis.from_url(
            redis_url,
            decode_responses=True,
            socket_connect_timeout=3,
            socket_timeout=3,
        )

        self.namespace = namespace
        self.similarity_threshold = similarity_threshold
        self.ttl_seconds = ttl_seconds

        self.key_prefix = (
            f"enterprise_rag:semantic_cache:{namespace}:"
        )

    def ping(self):
        """Check whether Redis is available."""
        return self.redis.ping()

    @staticmethod
    def cosine_similarity(vector_a, vector_b):
        """Calculate cosine similarity between two vectors."""

        if len(vector_a) != len(vector_b):
            return -1.0

        dot_product = sum(
            a * b
            for a, b in zip(vector_a, vector_b)
        )

        norm_a = math.sqrt(
            sum(a * a for a in vector_a)
        )

        norm_b = math.sqrt(
            sum(b * b for b in vector_b)
        )

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)

    def find_similar(self, query_embedding):
        """
        Search Redis for the most similar cached question.

        Returns the cached record if similarity
        exceeds the configured threshold.
        """

        best_match = None
        best_score = -1.0

        pattern = f"{self.key_prefix}*"

        for key in self.redis.scan_iter(
            match=pattern,
            count=100,
        ):
            stored_data = self.redis.get(key)

            if not stored_data:
                continue

            try:
                record = json.loads(stored_data)

                stored_embedding = record.get(
                    "embedding"
                )

                if not stored_embedding:
                    continue

                score = self.cosine_similarity(
                    query_embedding,
                    stored_embedding,
                )

                if score > best_score:
                    best_score = score
                    best_match = record

            except (ValueError, TypeError, KeyError):
                continue

        if (
            best_match is not None
            and best_score >= self.similarity_threshold
        ):
            return {
                "question": best_match["question"],
                "answer": best_match["answer"],
                "similarity": best_score,
            }

        return None

    def store(
        self,
        question,
        answer,
        query_embedding,
    ):
        """Store a successfully validated answer."""

        record = {
            "question": question,
            "answer": answer,
            "embedding": query_embedding,
            "namespace": self.namespace,
            "created_at": time.time(),
        }

        key = (
            f"{self.key_prefix}{uuid.uuid4().hex}"
        )

        self.redis.set(
            key,
            json.dumps(record),
            ex=self.ttl_seconds,
        )

        return key

    def clear_namespace(self):
        """
        Delete cache entries belonging only
        to the current knowledge-base namespace.
        """

        pattern = f"{self.key_prefix}*"
        deleted_count = 0

        for key in self.redis.scan_iter(
            match=pattern,
            count=100,
        ):
            deleted_count += self.redis.delete(key)

        return deleted_count

    def close(self):
        """Close the Redis connection."""
        self.redis.close()
