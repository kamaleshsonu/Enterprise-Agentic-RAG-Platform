from app.ingestion.pipeline import process_dataset
from app.ingestion.storage import save_chunks


true_chunks = process_dataset("DATA/true_data")
noisy_chunks = process_dataset("DATA/noisy_data")

all_chunks = true_chunks + noisy_chunks

save_chunks(
    all_chunks,
    "processed_data/chunks.jsonl",
)

print(f"True data chunks: {len(true_chunks)}")
print(f"Noisy data chunks: {len(noisy_chunks)}")
print(f"Total chunks saved: {len(all_chunks)}")
