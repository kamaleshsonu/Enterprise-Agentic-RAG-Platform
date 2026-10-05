import json
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct


QDRANT_HOST = "localhost"
QDRANT_PORT = 6333

COLLECTION_NAME = "enterprise_rag"
INPUT_FILE = "processed_data/embeddings.jsonl"

BATCH_SIZE = 100


def load_embeddings(input_file):
    records = []

    with Path(input_file).open(
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:
            if line.strip():
                records.append(
                    json.loads(line)
                )

    return records


def upload_to_qdrant(client, records):

    total = len(records)

    for start in range(
        0,
        total,
        BATCH_SIZE,
    ):

        batch = records[
            start:start + BATCH_SIZE
        ]

        points = []

        for index, record in enumerate(batch):

            point_id = start + index + 1

            point = PointStruct(
                id=point_id,
                vector=record["vector"],
                payload={
                    "chunk_id": record["chunk_id"],
                    "text": record["text"],
                    "document_id": record["document_id"],
                    "file_name": record["file_name"],
                    "file_type": record["file_type"],
                    "dataset": record["dataset"],
                    "file_path": record["file_path"],
                },
            )

            points.append(point)

        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
        )

        processed = min(
            start + BATCH_SIZE,
            total,
        )

        print(
            f"Uploaded {processed} / {total} vectors"
        )


if __name__ == "__main__":

    print("Connecting to Qdrant...")

    client = QdrantClient(
        host=QDRANT_HOST,
        port=QDRANT_PORT,
    )

    print("Loading embeddings...")

    records = load_embeddings(
        INPUT_FILE
    )

    print(
        f"Embeddings loaded: {len(records)}"
    )

    print("Starting Qdrant upload...")

    upload_to_qdrant(
        client,
        records,
    )

    print()
    print("Qdrant upload completed.")

    collection_info = client.get_collection(
        COLLECTION_NAME
    )

    print(
        "Vectors stored in Qdrant:",
        collection_info.points_count,
    )

    client.close()