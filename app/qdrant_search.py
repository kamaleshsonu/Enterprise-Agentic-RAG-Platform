import vertexai
from vertexai.language_models import TextEmbeddingModel
from qdrant_client import QdrantClient
from flashrank import Ranker, RerankRequest


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ID = "ai-security-bootcamp"
LOCATION = "us-central1"
MODEL_NAME = "text-embedding-005"

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "enterprise_rag"

# Test question
QUESTION = "How does Kubernetes autoscaling work?"

# Retrieve 10 candidates from Qdrant
QDRANT_TOP_K = 10

# Keep the best 5 after reranking
FINAL_TOP_K = 5

# FlashRank model
FLASHRANK_MODEL = "ms-marco-MiniLM-L-12-v2"


# ============================================================
# VERTEX AI EMBEDDING MODEL
# ============================================================

def create_embedding_model():

    vertexai.init(
        project=PROJECT_ID,
        location=LOCATION,
    )

    model = TextEmbeddingModel.from_pretrained(
        MODEL_NAME
    )

    return model


# ============================================================
# CREATE QUERY EMBEDDING
# ============================================================

def create_query_embedding(
    model,
    question,
):

    embeddings = model.get_embeddings(
        [question]
    )

    query_vector = embeddings[0].values

    return query_vector


# ============================================================
# QDRANT SEARCH
# ============================================================

def search_qdrant(
    client,
    query_vector,
    limit=QDRANT_TOP_K,
):

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=limit,
        with_payload=True,
    )

    return results.points


# ============================================================
# CREATE FLASHRANK RERANKER
# ============================================================

def create_reranker():

    reranker = Ranker(
        model_name=FLASHRANK_MODEL
    )

    return reranker


# ============================================================
# RERANK QDRANT RESULTS
# ============================================================

def rerank_documents(
    reranker,
    question,
    qdrant_results,
):

    passages = []

    for result in qdrant_results:

        payload = result.payload

        passage = {
            "id": payload.get("chunk_id"),
            "text": payload.get("text", ""),
        }

        passages.append(passage)

    rerank_request = RerankRequest(
        query=question,
        passages=passages,
    )

    reranked_results = reranker.rerank(
        rerank_request
    )

    return reranked_results


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # STEP 1
    # Create Vertex AI embedding model
    # --------------------------------------------------------

    print()
    print("Creating Vertex AI embedding model...")

    model = create_embedding_model()


    # --------------------------------------------------------
    # STEP 2
    # Create embedding for user question
    # --------------------------------------------------------

    print("Creating question embedding...")

    query_vector = create_query_embedding(
        model,
        QUESTION,
    )

    print(
        "Query vector dimensions:",
        len(query_vector),
    )


    # --------------------------------------------------------
    # STEP 3
    # Connect to Qdrant
    # --------------------------------------------------------

    print("Connecting to Qdrant...")

    client = QdrantClient(
        host=QDRANT_HOST,
        port=QDRANT_PORT,
    )


    # --------------------------------------------------------
    # STEP 4
    # Retrieve candidates from Qdrant
    # --------------------------------------------------------

    print()
    print(
        f"Retrieving top {QDRANT_TOP_K} candidates from Qdrant..."
    )

    qdrant_results = search_qdrant(
        client=client,
        query_vector=query_vector,
        limit=QDRANT_TOP_K,
    )


    # --------------------------------------------------------
    # DISPLAY QDRANT RESULTS
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("QDRANT RESULTS")
    print("=" * 80)

    print()
    print("Question:")
    print(QUESTION)

    for index, result in enumerate(
        qdrant_results,
        start=1,
    ):

        payload = result.payload

        print()
        print(
            f"QDRANT RESULT #{index}"
        )

        print("-" * 80)

        print(
            "Qdrant Score:",
            result.score,
        )

        print(
            "Chunk ID:",
            payload.get("chunk_id"),
        )

        print(
            "File:",
            payload.get("file_name"),
        )

        print(
            "Dataset:",
            payload.get("dataset"),
        )

        print()
        print("Text:")

        print(
            payload.get("text", "")[:500]
        )


    # --------------------------------------------------------
    # STEP 5
    # Create FlashRank reranker
    # --------------------------------------------------------

    print()
    print(
        "Creating FlashRank reranker..."
    )

    reranker = create_reranker()


    # --------------------------------------------------------
    # STEP 6
    # Rerank Qdrant results
    # --------------------------------------------------------

    print()
    print(
        "Reranking Qdrant results with FlashRank..."
    )

    reranked_results = rerank_documents(
        reranker=reranker,
        question=QUESTION,
        qdrant_results=qdrant_results,
    )


    # --------------------------------------------------------
    # STEP 7
    # DISPLAY FINAL RERANKED RESULTS
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print(
        f"FINAL FLASHRANK RESULTS - TOP {FINAL_TOP_K}"
    )
    print("=" * 80)

    for index, result in enumerate(
        reranked_results[:FINAL_TOP_K],
        start=1,
    ):

        print()
        print(
            f"FINAL RESULT #{index}"
        )

        print("-" * 80)

        print(
            "FlashRank Score:",
            result["score"],
        )

        print(
            "Chunk ID:",
            result["id"],
        )

        print()
        print("Text:")

        print(
            result["text"][:1000]
        )


    # --------------------------------------------------------
    # STEP 8
    # Close Qdrant connection
    # --------------------------------------------------------

    client.close()

    print()
    print("=" * 80)
    print(
        "Qdrant + FlashRank retrieval completed."
    )
    print("=" * 80)