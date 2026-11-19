import os
from typing import TypedDict

# ============================================================
# LANGGRAPH OPENTELEMETRY CONFIGURATION
# Set these before importing LangGraph.
# ============================================================
os.environ["LANGSMITH_OTEL_ENABLED"] = "true"
os.environ["LANGSMITH_OTEL_ONLY"] = "true"
os.environ["LANGSMITH_TRACING"] = "true"

import vertexai
import logfire
from vertexai.language_models import TextEmbeddingModel
from google import genai
from qdrant_client import QdrantClient
from flashrank import Ranker, RerankRequest
from langgraph.graph import StateGraph, START, END
from redis.exceptions import RedisError

from app.guardrail import check_input_guardrail
from app.semantic_cache import RedisSemanticCache
from app.config_loader import load_config


# ============================================================
# APPLICATION CONFIGURATION
# ============================================================

CONFIG = load_config()



# ============================================================
# PYDANTIC LOGFIRE CONFIGURATION
# ============================================================
logfire.configure(
    service_name="enterprise-agentic-rag"
)

# Instrument calls made through the Google Gen AI SDK.
logfire.instrument_google_genai()

print("Pydantic Logfire initialized successfully.")


# ============================================================
# 1. GOOGLE CLOUD CONFIGURATION
# ============================================================
PROJECT_ID = CONFIG["google_cloud"]["project_id"]
LOCATION = CONFIG["google_cloud"]["location"]

EMBEDDING_MODEL = CONFIG["models"]["embedding"]
LLM_MODEL = CONFIG["models"]["llm"]


# ============================================================
# 2. QDRANT CONFIGURATION
# ============================================================
QDRANT_HOST = CONFIG["qdrant"]["host"]
QDRANT_PORT = CONFIG["qdrant"]["port"]
COLLECTION_NAME = CONFIG["qdrant"]["collection_name"]


# ============================================================
# 3. RETRIEVAL CONFIGURATION
# ============================================================
QDRANT_TOP_K = CONFIG["retrieval"]["top_k"]
FINAL_TOP_K = CONFIG["retrieval"]["final_top_k"]

FLASHRANK_MODEL = CONFIG["reranker"]["model"]


# ============================================================
# 4. AGENT CONFIGURATION
# ============================================================
MAX_RETRIES = CONFIG["agent"]["max_retries"]


# ============================================================
# 5. REDIS SEMANTIC CACHE CONFIGURATION
# ============================================================
CACHE_ENABLED = CONFIG["redis_cache"]["enabled"]
CACHE_NAMESPACE = CONFIG["redis_cache"]["namespace"]
CACHE_SIMILARITY_THRESHOLD = CONFIG["redis_cache"]["similarity_threshold"]
CACHE_TTL_SECONDS = CONFIG["redis_cache"]["ttl_seconds"]


# ============================================================
# 6. LANGGRAPH STATE
# ============================================================
class AgentState(TypedDict):
    question: str
    input_safe: str
    query_vector: list[float]
    query_vector_text: str
    cache_query_vector: list[float]
    qdrant_results: list
    reranked_results: list
    context: str
    context_relevant: str
    rewritten_question: str
    retry_count: int
    answer: str
    answer_valid: str
    cache_hit: bool
    cache_similarity: float
    cache_stored: bool


# ============================================================
# 7. CREATE EMBEDDING MODEL
# ============================================================
def create_embedding_model():
    vertexai.init(
        project=PROJECT_ID,
        location=LOCATION,
    )

    return TextEmbeddingModel.from_pretrained(
        EMBEDDING_MODEL
    )


# ============================================================
# 8. CREATE QUERY EMBEDDING
# ============================================================
def create_query_embedding(model, question):
    embeddings = model.get_embeddings([question])
    return embeddings[0].values


# ============================================================
# 9. CREATE QDRANT CLIENT
# ============================================================
def create_qdrant_client():
    return QdrantClient(
        host=QDRANT_HOST,
        port=QDRANT_PORT,
    )


# ============================================================
# 10. CREATE FLASHRANK RERANKER
# ============================================================
def create_reranker():
    return Ranker(
        model_name=FLASHRANK_MODEL
    )


# ============================================================
# 11. CREATE GEMINI CLIENT
# ============================================================
def create_llm_client():
    return genai.Client(
        vertexai=True,
        project=PROJECT_ID,
        location=LOCATION,
    )


# ============================================================
# NODE 0 — INPUT GUARDRAIL
# ============================================================
def input_guardrail_node(state: AgentState):
    print()
    print("NODE 0: INPUT GUARDRAIL")
    print("-" * 60)

    question = state["question"]
    decision = check_input_guardrail(question)

    print("Input guardrail decision:", decision)

    return {
        "input_safe": decision
    }


# ============================================================
# INPUT GUARDRAIL ROUTER
# ============================================================
def route_after_input_guardrail(state: AgentState):
    decision = state["input_safe"]

    if decision == "SAFE":
        print()
        print("Input is safe.")
        print("ROUTER → SEMANTIC CACHE")
        return "cache_lookup"

    print()
    print("Input blocked by security guardrail.")
    print("ROUTER → BLOCKED INPUT")
    return "blocked"


# ============================================================
# BLOCKED INPUT NODE
# ============================================================
def blocked_input_node(state: AgentState):
    print()
    print("NODE: BLOCKED INPUT")
    print("-" * 60)

    answer = "I'm sorry, but I can't process that request."

    print("Security guardrail blocked the request.")

    return {
        "answer": answer,
        "answer_valid": "BLOCKED",
    }


# ============================================================
# NODE 1 — REDIS SEMANTIC CACHE LOOKUP
# ============================================================
def semantic_cache_lookup_node(state: AgentState):
    print()
    print("NODE 1: SEMANTIC CACHE LOOKUP")
    print("-" * 60)

    question = state["question"]
    print("User question:", question)

    # Generate one embedding for the original question.
    # Reuse it for Qdrant on a cache miss.
    embedding_model = create_embedding_model()
    query_vector = create_query_embedding(
        embedding_model,
        question,
    )

    print("Query embedding dimensions:", len(query_vector))

    cache_match = None
    cache_client = None

    if CACHE_ENABLED:
        try:
            cache_client = RedisSemanticCache(
                namespace=CACHE_NAMESPACE,
                similarity_threshold=CACHE_SIMILARITY_THRESHOLD,
                ttl_seconds=CACHE_TTL_SECONDS,
            )

            # Custom Logfire span for Redis semantic lookup.
            with logfire.span(
                "redis.semantic_cache.lookup",
                cache_namespace=CACHE_NAMESPACE,
                embedding_dimensions=len(query_vector),
            ) as cache_span:
                cache_match = cache_client.find_similar(query_vector)

                cache_span.set_attribute(
                    "cache.hit",
                    cache_match is not None,
                )

                if cache_match is not None:
                    cache_span.set_attribute(
                        "cache.similarity",
                        float(cache_match["similarity"]),
                    )

        except RedisError as error:
            # Cache failure should not stop the normal RAG pipeline.
            print()
            print("Redis unavailable. Continuing with normal RAG.")
            print("Redis error:", error)

            logfire.warning(
                "Redis semantic cache lookup failed; bypassing cache",
                error_type=type(error).__name__,
            )

        finally:
            if cache_client is not None:
                try:
                    cache_client.close()
                except RedisError:
                    pass

    # --------------------------------------------------------
    # CACHE HIT
    # --------------------------------------------------------
    if cache_match is not None:
        similarity = float(cache_match["similarity"])

        print()
        print("SEMANTIC CACHE HIT!")
        print("Matched question:", cache_match["question"])
        print("Similarity:", similarity)

        return {
            "query_vector": query_vector,
            "query_vector_text": question,
            "cache_query_vector": query_vector,
            "cache_hit": True,
            "cache_similarity": similarity,
            "answer": cache_match["answer"],
        }

    # --------------------------------------------------------
    # CACHE MISS
    # --------------------------------------------------------
    print()
    print("SEMANTIC CACHE MISS")
    print("Continuing with Qdrant retrieval.")

    return {
        "query_vector": query_vector,
        "query_vector_text": question,
        "cache_query_vector": query_vector,
        "cache_hit": False,
        "cache_similarity": 0.0,
    }


# ============================================================
# CACHE ROUTER
# ============================================================
def route_after_cache_lookup(state: AgentState):
    if state.get("cache_hit", False):
        print()
        print("CACHE ROUTER → CACHED ANSWER")
        return "cache_hit"

    print()
    print("CACHE ROUTER → RETRIEVE")
    return "cache_miss"


# ============================================================
# NODE 1A — RETURN CACHED ANSWER
# ============================================================
def cached_answer_node(state: AgentState):
    print()
    print("NODE: RETURN CACHED ANSWER")
    print("-" * 60)

    print("Returning previously validated answer.")
    print("Similarity score:", state.get("cache_similarity", 0.0))

    print()
    print("Cached answer:")
    print(state["answer"])

    return {
        "answer_valid": "CACHE_HIT",
    }


# ============================================================
# NODE 2 — QDRANT RETRIEVAL
# ============================================================
def retrieve_node(state: AgentState):
    print()
    print("NODE 2: RETRIEVE")
    print("-" * 60)

    question = state["question"]
    rewritten_question = state.get("rewritten_question", "")

    if rewritten_question:
        question_for_search = rewritten_question
        print("Using rewritten question:", question_for_search)
    else:
        question_for_search = question
        print("Using original question:", question_for_search)

    # Reuse the original query embedding when the query is unchanged.
    query_vector = state.get("query_vector", [])
    query_vector_text = state.get("query_vector_text", "")

    if query_vector and query_vector_text == question_for_search:
        print("Reusing existing query embedding.")
    else:
        print("Generating embedding for rewritten query.")
        embedding_model = create_embedding_model()
        query_vector = create_query_embedding(
            embedding_model,
            question_for_search,
        )

    print("Query vector dimensions:", len(query_vector))

    qdrant_client = create_qdrant_client()
    try:
        # Custom Logfire span for Qdrant retrieval.
        with logfire.span(
            "qdrant.retrieve",
            collection_name=COLLECTION_NAME,
            top_k=QDRANT_TOP_K,
        ) as qdrant_span:
            results = qdrant_client.query_points(
                collection_name=COLLECTION_NAME,
                query=query_vector,
                limit=QDRANT_TOP_K,
                with_payload=True,
            )
            qdrant_results = results.points
            qdrant_span.set_attribute(
                "qdrant.result_count",
                len(qdrant_results),
            )
    finally:
        qdrant_client.close()

    print(
        f"Retrieved {len(qdrant_results)} results from Qdrant."
    )

    return {
        "query_vector": query_vector,
        "query_vector_text": question_for_search,
        "qdrant_results": qdrant_results,
    }


# ============================================================
# NODE 3 — RERANK
# ============================================================
def rerank_node(state: AgentState):
    print()
    print("NODE 3: RERANK")
    print("-" * 60)

    question = state["question"]
    rewritten_question = state.get("rewritten_question", "")
    question_for_reranking = rewritten_question or question

    qdrant_results = state["qdrant_results"]

    if not qdrant_results:
        print("No documents retrieved from Qdrant.")
        return {
            "reranked_results": [],
        }

    reranker = create_reranker()
    passages = []

    for result in qdrant_results:
        payload = result.payload or {}
        passages.append(
            {
                "id": payload.get("chunk_id"),
                "text": payload.get("text", ""),
            }
        )

    rerank_request = RerankRequest(
        query=question_for_reranking,
        passages=passages,
    )

    # Custom Logfire span for FlashRank.
    with logfire.span(
        "flashrank.rerank",
        candidate_count=len(passages),
        final_top_k=FINAL_TOP_K,
    ) as rerank_span:
        reranked_results = reranker.rerank(rerank_request)
        rerank_span.set_attribute(
            "flashrank.result_count",
            len(reranked_results[:FINAL_TOP_K]),
        )

    print(
        f"FlashRank selected top {FINAL_TOP_K} results."
    )

    for index, result in enumerate(
        reranked_results[:FINAL_TOP_K],
        start=1,
    ):
        print(
            f"{index}. {result['id']} "
            f"(score={result['score']})"
        )

    return {
        "reranked_results": reranked_results,
    }


# ============================================================
# NODE 4 — CREATE CONTEXT
# ============================================================
def context_node(state: AgentState):
    print()
    print("NODE 4: CREATE CONTEXT")
    print("-" * 60)

    reranked_results = state["reranked_results"]
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

    context = "\n".join(context_parts)

    print(
        f"Created context from {len(context_parts)} sources."
    )

    return {
        "context": context,
    }


# ============================================================
# NODE 5 — CHECK CONTEXT
# ============================================================
def check_context_node(state: AgentState):
    print()
    print("NODE 5: CHECK CONTEXT")
    print("-" * 60)

    question = state["question"]
    context = state["context"]

    if not context.strip():
        print("No context available.")
        return {
            "context_relevant": "NO",
        }

    llm_client = create_llm_client()

    prompt = f"""
You are evaluating retrieved documents for an enterprise RAG system.

Determine whether the retrieved context contains enough relevant information
 to answer the user's question.

Return ONLY one word: YES or NO.

User Question:
{question}

Retrieved Context:
{context}
"""

    response = llm_client.models.generate_content(
        model=LLM_MODEL,
        contents=prompt,
    )

    decision_text = (response.text or "").strip().upper()
    decision = "YES" if decision_text == "YES" else "NO"

    print("Context sufficient:", decision)

    return {
        "context_relevant": decision,
    }


# ============================================================
# NODE 6 — REWRITE QUERY
# ============================================================
def rewrite_query_node(state: AgentState):
    print()
    print("NODE 6: REWRITE QUERY")
    print("-" * 60)

    question = state["question"]
    context = state["context"]
    current_retry_count = state.get("retry_count", 0)

    llm_client = create_llm_client()

    prompt = f"""
You are a query rewriting component inside an enterprise RAG system.

The original question did not produce sufficiently relevant retrieval results.
Rewrite the user's question so that it is more precise and useful for semantic search.
Keep the same intent. Return ONLY the rewritten question.

Original Question:
{question}

Retrieved Context:
{context}

Rewritten Question:
"""

    response = llm_client.models.generate_content(
        model=LLM_MODEL,
        contents=prompt,
    )

    rewritten_question = (response.text or "").strip()
    if not rewritten_question:
        rewritten_question = question

    new_retry_count = current_retry_count + 1

    print("Rewritten question:", rewritten_question)
    print(f"Retry count: {new_retry_count}/{MAX_RETRIES}")

    return {
        "rewritten_question": rewritten_question,
        "retry_count": new_retry_count,
    }


# ============================================================
# NODE 7 — GENERATE ANSWER
# ============================================================
def generate_node(state: AgentState):
    print()
    print("NODE 7: GENERATE")
    print("-" * 60)

    question = state["question"]
    context = state["context"]
    llm_client = create_llm_client()

    prompt = f"""
You are an enterprise RAG assistant.

Answer the user's question using ONLY the information provided in the context.
Do not invent information.

If the answer cannot be found in the context, say:
"I could not find the answer in the provided documents."

Keep the answer clear and concise.

Context:
{context}

User Question:
{question}

Answer:
"""

    response = llm_client.models.generate_content(
        model=LLM_MODEL,
        contents=prompt,
    )

    answer = (response.text or "").strip()

    print()
    print("Generated answer:")
    print(answer)

    return {
        "answer": answer,
    }


# ============================================================
# NODE 8 — VALIDATE ANSWER
# ============================================================
def validate_answer_node(state: AgentState):
    print()
    print("NODE 8: VALIDATE ANSWER")
    print("-" * 60)

    question = state["question"]
    context = state["context"]
    answer = state["answer"]
    llm_client = create_llm_client()

    prompt = f"""
You are an answer validation component inside an enterprise RAG system.

Determine whether the generated answer is supported by the retrieved context.
Check:
1. Is the answer relevant to the user's question?
2. Is the answer supported by the retrieved context?

If the answer is relevant AND supported, return YES.
Otherwise return NO.
Return ONLY YES or NO.

USER QUESTION:
{question}

RETRIEVED CONTEXT:
{context}

GENERATED ANSWER:
{answer}
"""

    response = llm_client.models.generate_content(
        model=LLM_MODEL,
        contents=prompt,
    )

    decision_text = (response.text or "").strip().upper()
    decision = "YES" if decision_text == "YES" else "NO"

    print("Answer valid:", decision)

    return {
        "answer_valid": decision,
    }


# ============================================================
# NODE 9 — STORE VALIDATED ANSWER IN REDIS
# ============================================================
def cache_store_node(state: AgentState):
    print()
    print("NODE 9: STORE SEMANTIC CACHE")
    print("-" * 60)

    if state.get("answer_valid") != "YES":
        print("Answer was not validated. Skipping cache storage.")
        return {
            "cache_stored": False,
        }

    question = state["question"]
    answer = state["answer"]
    query_vector = state.get("cache_query_vector", [])

    if not query_vector:
        print("Original query embedding unavailable. Skipping cache storage.")
        return {
            "cache_stored": False,
        }

    cache_client = None

    try:
        cache_client = RedisSemanticCache(
            namespace=CACHE_NAMESPACE,
            similarity_threshold=CACHE_SIMILARITY_THRESHOLD,
            ttl_seconds=CACHE_TTL_SECONDS,
        )

        # Custom Logfire span for Redis cache storage.
        with logfire.span(
            "redis.semantic_cache.store",
            cache_namespace=CACHE_NAMESPACE,
        ) as cache_span:
            cache_client.store(
                question=question,
                answer=answer,
                query_embedding=query_vector,
            )
            cache_span.set_attribute("cache.stored", True)

        print("Validated answer stored in Redis.")

        return {
            "cache_stored": True,
        }

    except RedisError as error:
        print("Redis storage failed:", error)
        print("Returning answer without caching.")

        logfire.warning(
            "Redis semantic cache storage failed",
            error_type=type(error).__name__,
        )

        return {
            "cache_stored": False,
        }

    finally:
        if cache_client is not None:
            try:
                cache_client.close()
            except RedisError:
                pass


# ============================================================
# NODE 10 — SAFE FALLBACK
# ============================================================
def fallback_node(state: AgentState):
    print()
    print("NODE 10: SAFE FALLBACK")
    print("-" * 60)

    answer = (
        "I could not provide a sufficiently grounded answer "
        "from the provided documents."
    )

    print("Safe fallback activated.")

    return {
        "answer": answer,
    }


# ============================================================
# CONTEXT ROUTER
# ============================================================
def route_after_check(state: AgentState):
    decision = state["context_relevant"]
    retry_count = state.get("retry_count", 0)

    if decision == "YES":
        print()
        print("Context is sufficient.")
        print("ROUTER → GENERATE")
        return "generate"

    print()
    print("Context is insufficient.")
    print(f"Current retry count: {retry_count}/{MAX_RETRIES}")

    if retry_count < MAX_RETRIES:
        print("ROUTER → REWRITE QUERY")
        return "rewrite_query"

    print("ROUTER → FALLBACK")
    return "fallback"


# ============================================================
# ANSWER VALIDATION ROUTER
# ============================================================
def route_after_validation(state: AgentState):
    decision = state["answer_valid"]

    if decision == "YES":
        print()
        print("Answer is grounded.")
        print("ROUTER → STORE CACHE")
        return "store_cache"

    print()
    print("Answer failed validation.")
    print("ROUTER → FALLBACK")
    return "fallback"


# ============================================================
# BUILD LANGGRAPH
# ============================================================
def build_graph():
    graph_builder = StateGraph(AgentState)

    # Register guardrail nodes.
    graph_builder.add_node("input_guardrail", input_guardrail_node)
    graph_builder.add_node("blocked_input", blocked_input_node)

    # Register semantic cache nodes.
    graph_builder.add_node(
        "semantic_cache_lookup",
        semantic_cache_lookup_node,
    )
    graph_builder.add_node("cached_answer", cached_answer_node)
    graph_builder.add_node("cache_store", cache_store_node)

    # Register RAG nodes.
    graph_builder.add_node("retrieve", retrieve_node)
    graph_builder.add_node("rerank", rerank_node)
    graph_builder.add_node("create_context", context_node)
    graph_builder.add_node("check_context", check_context_node)
    graph_builder.add_node("rewrite_query", rewrite_query_node)
    graph_builder.add_node("generate", generate_node)
    graph_builder.add_node("validate_answer", validate_answer_node)
    graph_builder.add_node("fallback", fallback_node)

    # START → INPUT GUARDRAIL
    graph_builder.add_edge(START, "input_guardrail")

    # INPUT GUARDRAIL ROUTING
    graph_builder.add_conditional_edges(
        "input_guardrail",
        route_after_input_guardrail,
        {
            "cache_lookup": "semantic_cache_lookup",
            "blocked": "blocked_input",
        },
    )

    # BLOCKED INPUT → END
    graph_builder.add_edge("blocked_input", END)

    # SEMANTIC CACHE ROUTING
    graph_builder.add_conditional_edges(
        "semantic_cache_lookup",
        route_after_cache_lookup,
        {
            "cache_hit": "cached_answer",
            "cache_miss": "retrieve",
        },
    )

    # CACHED ANSWER → END
    graph_builder.add_edge("cached_answer", END)

    # NORMAL RETRIEVAL FLOW
    graph_builder.add_edge("retrieve", "rerank")
    graph_builder.add_edge("rerank", "create_context")
    graph_builder.add_edge("create_context", "check_context")

    # CONTEXT ROUTING
    graph_builder.add_conditional_edges(
        "check_context",
        route_after_check,
        {
            "generate": "generate",
            "rewrite_query": "rewrite_query",
            "fallback": "fallback",
        },
    )

    # QUERY REWRITE LOOP
    graph_builder.add_edge("rewrite_query", "retrieve")

    # GENERATE → VALIDATE
    graph_builder.add_edge("generate", "validate_answer")

    # ANSWER VALIDATION ROUTING
    graph_builder.add_conditional_edges(
        "validate_answer",
        route_after_validation,
        {
            "store_cache": "cache_store",
            "fallback": "fallback",
        },
    )

    # CACHE STORE → END
    graph_builder.add_edge("cache_store", END)

    # FALLBACK → END
    graph_builder.add_edge("fallback", END)

    return graph_builder.compile()


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print()
    print("=" * 80)
    print("ENTERPRISE AGENTIC RAG WITH REDIS SEMANTIC CACHE")
    print("=" * 80)

    question = input("Ask your question: ").strip()

    if not question:
        print("Question cannot be empty.")
        raise SystemExit(1)

    graph = build_graph()

    initial_state: AgentState = {
        "question": question,
        "input_safe": "",
        "query_vector": [],
        "query_vector_text": "",
        "cache_query_vector": [],
        "qdrant_results": [],
        "reranked_results": [],
        "context": "",
        "context_relevant": "",
        "rewritten_question": "",
        "retry_count": 0,
        "answer": "",
        "answer_valid": "",
        "cache_hit": False,
        "cache_similarity": 0.0,
        "cache_stored": False,
    }

    result = graph.invoke(initial_state)

    print()
    print("=" * 80)
    print("FINAL RESULT")
    print("=" * 80)

    print("INPUT GUARDRAIL:", result.get("input_safe", ""))
    print("CACHE HIT:", result.get("cache_hit", False))
    print("CACHE SIMILARITY:", result.get("cache_similarity", 0.0))
    print("CACHE STORED:", result.get("cache_stored", False))
    print("FINAL CONTEXT DECISION:", result.get("context_relevant", ""))
    print("TOTAL QUERY REWRITES:", result.get("retry_count", 0))
    print("ANSWER VALIDATION:", result.get("answer_valid", ""))

    if result.get("rewritten_question"):
        print()
        print("LAST REWRITTEN QUESTION:")
        print(result["rewritten_question"])

    print()
    print("-" * 80)
    print("FINAL ANSWER")
    print("-" * 80)
    print()
    print(result["answer"])
    print()
    print("=" * 80)
    print("AGENTIC RAG COMPLETED")
    print("=" * 80)