from dataclasses import dataclass


@dataclass
class DocumentChunk:
    chunk_id: str
    text: str
    document_id: str
    file_name: str
    file_type: str
    dataset: str
    file_path: str