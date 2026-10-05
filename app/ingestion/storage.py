import json
from pathlib import Path

from app.ingestion.models import DocumentChunk


def save_chunks(
    chunks: list[DocumentChunk],
    output_file: str,
) -> None:
    """
    Save document chunks as JSON Lines.
    """

    output_path = Path(output_file)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        for chunk in chunks:

            record = {
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "document_id": chunk.document_id,
                "file_name": chunk.file_name,
                "file_type": chunk.file_type,
                "dataset": chunk.dataset,
                "file_path": chunk.file_path,
            }

            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )