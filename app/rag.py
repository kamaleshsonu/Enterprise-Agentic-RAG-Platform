import vertexai
from vertexai.language_models import TextEmbeddingModel

from google import genai

from qdrant_client import QdrantClient

from flashrank import Ranker, RerankRequest


# ============================================================
# GOOGLE CLOUD CONFIGURATION
# ============================================================

PROJECT_ID = "ai-security-bootcamp"
LOCATION = "us-central1"

EMBEDDING_MODEL = "text-embedding-005"
LLM_MODEL = "gemini-2.5-flash"


# ============================================================
# QDRANT CONFIGURATION
# ============================================================

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "enterprise_rag"


# ============================================================
# RETRIEVAL CONFIGURATION
# ============================================================

QDRANT_TOP_K = 10
FINAL_TOP_K = 5

FLASHRANK_MODEL = "ms-marco-MiniLM-L-12-v2"


# ============================================================
# CREATE VERTEX AI EMBEDDING MODEL
# ============================================================

def create_embedding_model():

    vertexai.init(
        project=PROJECT_ID,
        location=LOCATION,
    )

    model = TextEmbeddingModel.from_pretrained(
        EMBEDDING_MODEL
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
# SEARCH QDRANT
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
# CREATE CONTEXT FOR THE LLM
# ============================================================

def create_context(
    reranked_results,
):

    context_parts = []

    for index, result in enumerate(
        reranked_results[:FINAL_TOP_K],
        start=1,
    ):

        context_parts.append(
            f"""
SOURCE {index}

Chunk ID:
{result["id"]}

Content:
{result["text"]}
"""
        )

    context = "\n".join(
        context_parts
    )

    return context


# ============================================================
# CREATE GEMINI CLIENT
# ============================================================

def create_llm_client():

    client = genai.Client(
        vertexai=True,
        project=PROJECT_ID,
        location=LOCATION,
    )

    return client


# ============================================================
# GENERATE FINAL RAG ANSWER
# ============================================================

def generate_answer(
    client,
    question,
    context,
):

    prompt = f"""
You are an enterprise RAG assistant.

Answer the user's question using ONLY the
information provided in the context.

Do not invent information.

If the answer cannot be found in the context,
say:

"I could not find the answer in the provided documents."

Keep the answer clear and concise.

Context:

{context}

User Question:

{question}

Answer:
"""

    response = client.models.generate_content(
        model=LLM_MODEL,
        contents=prompt,
    )

    return response.text


# ============================================================
# MAIN RAG PIPELINE
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 80)
    print("ENTERPRISE RAG")
    print("=" * 80)

    # --------------------------------------------------------
    # STEP 0: GET USER QUESTION
    # --------------------------------------------------------

    print()

    question = input(
        "Ask your question: "
    ).strip()

    if not question:

        print()
        print("Question cannot be empty.")
        exit()


    # --------------------------------------------------------
    # STEP 1: CREATE EMBEDDING MODEL
    # --------------------------------------------------------

    print()
    print(
        "1. Creating Vertex AI embedding model..."
    )

    embedding_model = create_embedding_model()


    # --------------------------------------------------------
    # STEP 2: CREATE QUESTION EMBEDDING
    # --------------------------------------------------------

    print()
    print(
        "2. Creating question embedding..."
    )

    query_vector = create_query_embedding(
        embedding_model,
        question,
    )

    print(
        "Query vector dimensions:",
        len(query_vector),
    )


    # --------------------------------------------------------
    # STEP 3: CONNECT TO QDRANT
    # --------------------------------------------------------

    print()
    print(
        "3. Connecting to Qdrant..."
    )

    qdrant_client = QdrantClient(
        host=QDRANT_HOST,
        port=QDRANT_PORT,
    )


    # --------------------------------------------------------
    # STEP 4: VECTOR SEARCH
    # --------------------------------------------------------

    print()
    print(
        f"4. Retrieving top {QDRANT_TOP_K} "
        "results from Qdrant..."
    )

    qdrant_results = search_qdrant(
        client=qdrant_client,
        query_vector=query_vector,
        limit=QDRANT_TOP_K,
    )

    print(
        f"Retrieved {len(qdrant_results)} "
        "results from Qdrant."
    )


    # --------------------------------------------------------
    # STEP 5: CREATE FLASHRANK
    # --------------------------------------------------------

    print()
    print(
        "5. Creating FlashRank reranker..."
    )

    reranker = create_reranker()


    # --------------------------------------------------------
    # STEP 6: RERANK
    # --------------------------------------------------------

    print()
    print(
        f"6. Reranking results and selecting "
        f"top {FINAL_TOP_K}..."
    )

    reranked_results = rerank_documents(
        reranker=reranker,
        question=question,
        qdrant_results=qdrant_results,
    )


    # --------------------------------------------------------
    # STEP 7: CREATE RAG CONTEXT
    # --------------------------------------------------------

    print()
    print(
        "7. Creating RAG context..."
    )

    context = create_context(
        reranked_results
    )


    # --------------------------------------------------------
    # STEP 8: CONNECT TO GEMINI
    # --------------------------------------------------------

    print()
    print(
        "8. Connecting to Gemini..."
    )

    llm_client = create_llm_client()


    # --------------------------------------------------------
    # STEP 9: GENERATE ANSWER
    # --------------------------------------------------------

    print()
    print(
        "9. Generating final RAG answer..."
    )

    answer = generate_answer(
        client=llm_client,
        question=question,
        context=context,
    )


    # --------------------------------------------------------
    # FINAL OUTPUT
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("QUESTION")
    print("=" * 80)

    print()
    print(question)


    print()
    print("=" * 80)
    print("FINAL RAG ANSWER")
    print("=" * 80)

    print()
    print(answer)


    print()
    print("=" * 80)
    print("RAG PIPELINE COMPLETED")
    print("=" * 80)


    # --------------------------------------------------------
    # CLOSE QDRANT CONNECTION
    # --------------------------------------------------------

    qdrant_client.close()