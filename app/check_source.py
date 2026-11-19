
from pathlib import Path

from app.ingestion.processor import extract_text

file_path = Path("DATA/true_data/cronjobs.docx")

text = extract_text(file_path)

print("Extracted characters:", len(text))

for keyword in ["Idempotent Job Design", "hashlib.md5", "metadata_file"]:
    position = text.lower().find(keyword.lower())

    print(f"\nKeyword: {keyword}")
    print("Found:", position != -1)

    if position != -1:
        start = max(0, position - 300)
        end = min(len(text), position + 1800)
        print(text[start:end])
