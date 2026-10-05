import re


def clean_text(text: str) -> str:
    """
    Basic text cleaning for extracted document text.
    """

    # Remove leading/trailing spaces from each line
    lines = [line.strip() for line in text.splitlines()]

    # Rebuild the text
    text = "\n".join(lines)

    # Replace multiple spaces/tabs with a single space
    text = re.sub(r"[ \t]+", " ", text)

    # Reduce excessive blank lines
    text = re.sub(r"\n\s*\n+", "\n\n", text)

    # Remove leading/trailing whitespace
    text = text.strip()

    return text


def create_metadata(file_path, dataset_name):
    """
    Create metadata for a document.
    """

    return {
        "document_id": file_path.stem,
        "file_name": file_path.name,
        "file_type": file_path.suffix.lower(),
        "dataset": dataset_name,
        "file_path": str(file_path),
    }