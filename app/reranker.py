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

QUESTION = "How does Kubernetes autoscaling work?"

QDRANT_TOP_K = 10
FINAL_TOP_K = 5

FLASHRANK_MODEL = "ms-marco-MiniLM-L-12-v2"


# ============================================================
# VERTEX AI EMBEDDING MODEL
# ============================================================

def create_embedding_model():

    vertexai.init(
        project=PROJECT_ID,
        location=LOCATION,
    )

    return TextEmbeddingModel.from_pretrained(
        MODEL_NAME
    )


def create_query_embedding(model, question):

    embeddings = model.get_embeddings(
        [question]
    )

    return embeddings[0].values


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
# FLASHRANK
# ============================================================

def create_reranker():

    return Ranker(
        model_name=FLASHRANK_MODEL
    )


def rerank_documents(
    reranker,
    question,
    qdrant_results,
):

    passages = []

    for result in qdrant_results:

        payload = result.payload

        passages.append(
            {
                "id": payload.get("chunk_id"),
                "text": payload.get("text", ""),
            }
        )

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
    # Step 1: Create Vertex AI embedding model
    # --------------------------------------------------------

    print(
        "Creating Vertex AI embedding model..."
    )

    model = create_embedding_model()


    # --------------------------------------------------------
    # Step 2: Convert question into embedding
    # --------------------------------------------------------

    print(
        "Creating question embedding..."
    )

    query_vector = create_query_embedding(
        model,
        QUESTION,
    )

    print(
        "Query vector dimensions:",
        len(query_vector),
    )


    # --------------------------------------------------------
    # Step 3: Connect to Qdrant
    # --------------------------------------------------------

    print(
        "Connecting to Qdrant..."
    )

    client = QdrantClient(
        host=QDRANT_HOST,
        port=QDRANT_PORT,
    )


    # --------------------------------------------------------
    # Step 4: Retrieve candidates from Qdrant
    # --------------------------------------------------------

    print()
    print(
        f"Retrieving top {QDRANT_TOP_K} candidates from Qdrant..."
    )

    qdrant_results = search_qdrant(
        client,
        query_vector,
        limit=QDRANT_TOP_K,
    )


    # --------------------------------------------------------
    # Display Qdrant results
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
    # Step 5: Create FlashRank reranker
    # --------------------------------------------------------

    print()
    print(
        "Creating FlashRank reranker..."
    )

    reranker = create_reranker()


    # --------------------------------------------------------
    # Step 6: Rerank Qdrant results
    # --------------------------------------------------------

    print()
    print(
        "Reranking Qdrant results with FlashRank..."
    )

    reranked_results = rerank_documents(
        reranker,
        QUESTION,
        qdrant_results,
    )


    # --------------------------------------------------------
    # Step 7: Display final results
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
    # Cleanup
    # --------------------------------------------------------

    client.close()

    print()
    print("=" * 80)
    print(
        "Qdrant + FlashRank retrieval completed."
    )
    print("=" * 80)