import json
import time
from pathlib import Path

import vertexai
from vertexai.language_models import TextEmbeddingModel
from google.api_core.exceptions import ResourceExhausted


PROJECT_ID = "ai-security-bootcamp"
LOCATION = "us-central1"
MODEL_NAME = "text-embedding-005"

INPUT_FILE = "processed_data/chunks.jsonl"
OUTPUT_FILE = "processed_data/embeddings.jsonl"

BATCH_SIZE = 5
MAX_RETRIES = 5
INITIAL_RETRY_WAIT = 30


def create_embedding_model():
    vertexai.init(
        project=PROJECT_ID,
        location=LOCATION,
    )

    return TextEmbeddingModel.from_pretrained(
        MODEL_NAME
    )


def load_chunks(input_file):
    chunks = []

    with Path(input_file).open(
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:
            if line.strip():
                chunks.append(
                    json.loads(line)
                )

    return chunks


def generate_embeddings_with_retry(
    model,
    texts,
):
    for attempt in range(
        1,
        MAX_RETRIES + 1,
    ):

        try:

            embeddings = model.get_embeddings(
                texts
            )

            return [
                embedding.values
                for embedding in embeddings
            ]

        except ResourceExhausted:

            if attempt == MAX_RETRIES:
                print(
                    "Vertex AI quota remained "
                    "exhausted after all retries."
                )
                raise

            wait_time = (
                INITIAL_RETRY_WAIT * attempt
            )

            print()
            print(
                "Vertex AI returned 429 "
                "Resource Exhausted."
            )

            print(
                f"Waiting {wait_time} seconds "
                f"before retry..."
            )

            time.sleep(wait_time)


def generate_embeddings(
    model,
    chunks,
    output_file,
):

    output_path = Path(output_file)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    total = len(chunks)

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        for start in range(
            0,
            total,
            BATCH_SIZE,
        ):

            batch = chunks[
                start:start + BATCH_SIZE
            ]

            texts = [
                chunk["text"]
                for chunk in batch
            ]

            vectors = (
                generate_embeddings_with_retry(
                    model,
                    texts,
                )
            )

            for chunk, vector in zip(
                batch,
                vectors,
            ):

                record = {
                    "chunk_id": chunk["chunk_id"],
                    "text": chunk["text"],
                    "document_id": chunk["document_id"],
                    "file_name": chunk["file_name"],
                    "file_type": chunk["file_type"],
                    "dataset": chunk["dataset"],
                    "file_path": chunk["file_path"],
                    "vector": vector,
                }

                file.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            file.flush()

            processed = min(
                start + BATCH_SIZE,
                total,
            )

            print(
                f"Embedded and saved "
                f"{processed} / {total} chunks"
            )

            # Small pause between API requests
            time.sleep(1)


if __name__ == "__main__":

    print(
        "Loading chunks..."
    )

    chunks = load_chunks(
        INPUT_FILE
    )

    print(
        f"Chunks loaded: {len(chunks)}"
    )

    print(
        "Creating Vertex AI embedding model..."
    )

    model = create_embedding_model()

    print(
        "Starting embedding generation..."
    )

    generate_embeddings(
        model,
        chunks,
        OUTPUT_FILE,
    )

    print()
    print(
        "Embedding generation completed."
    )

    print(
        f"Embeddings saved to: {OUTPUT_FILE}"
    )