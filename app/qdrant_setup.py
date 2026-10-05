from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams


QDRANT_HOST = "localhost"
QDRANT_PORT = 6333

COLLECTION_NAME = "enterprise_rag"

VECTOR_SIZE = 768


def create_collection():

    client = QdrantClient(
        host=QDRANT_HOST,
        port=QDRANT_PORT,
    )

    existing_collections = client.get_collections()

    collection_names = [
        collection.name
        for collection in existing_collections.collections
    ]

    if COLLECTION_NAME in collection_names:

        print(
            f"Collection '{COLLECTION_NAME}' "
            f"already exists."
        )

    else:

        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=VECTOR_SIZE,
                distance=Distance.COSINE,
            ),
        )

        print(
            f"Collection '{COLLECTION_NAME}' "
            f"created successfully."
        )

    client.close()


if __name__ == "__main__":
    create_collection()