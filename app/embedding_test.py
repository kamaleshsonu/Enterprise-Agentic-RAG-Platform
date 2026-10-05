import vertexai
from vertexai.language_models import TextEmbeddingModel

PROJECT_ID = "ai-security-bootcamp"
LOCATION = "us-central1"

vertexai.init(
    project=PROJECT_ID,
    location=LOCATION,
)

model = TextEmbeddingModel.from_pretrained(
    "text-embedding-005"
)

embeddings = model.get_embeddings(
    ["Kubernetes is a platform for managing containers."]
)

vector = embeddings[0].values

print("Embedding test successful")
print("Vector dimensions:", len(vector))
print("First 5 values:", vector[:5])