from pathlib import Path

from app.ingestion.processor import extract_text
from app.ingestion.cleaner import clean_text, create_metadata
from app.ingestion.chunker import chunk_text
from app.ingestion.models import DocumentChunk


SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".pptx",
    ".html",
    ".txt",
}


def process_document(file_path: Path, dataset_name: str) -> list[DocumentChunk]:
    """
    Extract, clean, create metadata, and chunk one document.
    """

    # 1. Extract text
    raw_text = extract_text(file_path)

    if not raw_text.strip():
        return []

    # 2. Clean text
    cleaned_text = clean_text(raw_text)

    if not cleaned_text:
        return []

    # 3. Create document metadata
    metadata = create_metadata(
        file_path,
        dataset_name
    )

    # 4. Split into chunks
    chunks = chunk_text(cleaned_text)

    # 5. Create DocumentChunk objects
    document_chunks = []

    for index, chunk in enumerate(chunks):

        chunk_id = f"{metadata['document_id']}_{index + 1:04d}"

        document_chunk = DocumentChunk(
            chunk_id=chunk_id,
            text=chunk,
            document_id=metadata["document_id"],
            file_name=metadata["file_name"],
            file_type=metadata["file_type"],
            dataset=metadata["dataset"],
            file_path=metadata["file_path"],
        )

        document_chunks.append(document_chunk)

    return document_chunks

def process_dataset(data_path: str) -> list[DocumentChunk]:
    """
    Process all supported documents under a dataset directory.
    """

    data_dir = Path(data_path)

    all_chunks = []

    for file_path in data_dir.rglob("*"):

        if not file_path.is_file():
            continue

        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        dataset_name = data_dir.name

        chunks = process_document(
            file_path,
            dataset_name
        )

        all_chunks.extend(chunks)

    return all_chunks    