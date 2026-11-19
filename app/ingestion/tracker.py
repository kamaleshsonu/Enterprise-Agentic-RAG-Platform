
import hashlib
import json

from pathlib import Path


SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".pptx",
    ".html",
    ".txt",
}

DATA_DIR = Path("DATA")
MANIFEST_FILE = Path("processed_data/document_manifest.json")


def generate_document_id(file_path: Path) -> str:
    """
    Generate a stable document ID using its relative path.
    """

    relative_path = file_path.relative_to(DATA_DIR).as_posix()

    return hashlib.sha256(
        relative_path.encode("utf-8")
    ).hexdigest()


def calculate_file_hash(file_path: Path) -> str:
    """
    Generate SHA-256 hash of the file contents.
    """

    sha256 = hashlib.sha256()

    with file_path.open("rb") as file:

        for block in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            sha256.update(block)

    return sha256.hexdigest()


def load_manifest() -> dict:
    """
    Load previously tracked documents.
    """

    if not MANIFEST_FILE.exists():
        return {}

    with MANIFEST_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(file)


def save_manifest(manifest: dict) -> None:
    """
    Save document tracking information.
    """

    MANIFEST_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_file = MANIFEST_FILE.with_suffix(".tmp")

    with temp_file.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            manifest,
            file,
            indent=4,
        )

    temp_file.replace(MANIFEST_FILE)


def scan_documents() -> dict:
    """
    Scan DATA directory and collect document information.
    """

    current_documents = {}

    for file_path in DATA_DIR.rglob("*"):

        if not file_path.is_file():
            continue

        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        document_id = generate_document_id(file_path)

        current_documents[document_id] = {
            "document_id": document_id,
            "file_name": file_path.name,
            "file_path": file_path.as_posix(),
            "file_type": file_path.suffix.lower(),
            "content_hash": calculate_file_hash(file_path),
        }

    return current_documents


def detect_changes() -> dict:
    """
    Identify new, modified, unchanged and deleted documents.
    """

    previous_documents = load_manifest()
    current_documents = scan_documents()

    new_documents = []
    modified_documents = []
    unchanged_documents = []
    deleted_documents = []

    for document_id, current in current_documents.items():

        previous = previous_documents.get(document_id)

        if previous is None:

            new_documents.append(current)

        elif previous["content_hash"] != current["content_hash"]:

            modified_documents.append(current)

        else:

            unchanged_documents.append(current)

    for document_id, previous in previous_documents.items():

        if document_id not in current_documents:

            deleted_documents.append(previous)

    return {
        "new": new_documents,
        "modified": modified_documents,
        "unchanged": unchanged_documents,
        "deleted": deleted_documents,
    }


if __name__ == "__main__":

    changes = detect_changes()

    print("\nDOCUMENT TRACKING RESULTS")
    print("=" * 40)

    for status, documents in changes.items():

        print(f"\n{status.upper()}: {len(documents)}")

        for document in documents:

            print(
                f"  - {document['file_name']}"
            )

