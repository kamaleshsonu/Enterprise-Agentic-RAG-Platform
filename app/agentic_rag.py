from typing import TypedDict

import vertexai
from vertexai.language_models import TextEmbeddingModel

from google import genai

from qdrant_client import QdrantClient

from flashrank import Ranker, RerankRequest

from langgraph.graph import StateGraph, START, END


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
# AGENT CONFIGURATION
# ============================================================

MAX_RETRIES = 2


# ============================================================
# LANGGRAPH STATE
# ============================================================

class AgentState(TypedDict):

    question: str

    query_vector: list[float]

    qdrant_results: list

    reranked_results: list

    context: str

    context_relevant: str

    rewritten_question: str

    retry_count: int

    answer: str

    answer_valid: str


# ============================================================
# CREATE EMBEDDING MODEL
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
# CREATE QUERY EMBEDDING
# ============================================================

def create_query_embedding(
    model,
    question,
):

    embeddings = model.get_embeddings(
        [question]
    )

    return embeddings[0].values


# ============================================================
# CREATE QDRANT CLIENT
# ============================================================

def create_qdrant_client():

    return QdrantClient(
        host=QDRANT_HOST,
        port=QDRANT_PORT,
    )


# ============================================================
# CREATE FLASHRANK
# ============================================================

def create_reranker():

    return Ranker(
        model_name=FLASHRANK_MODEL
    )


# ============================================================
# CREATE GEMINI CLIENT
# ============================================================

def create_llm_client():

    return genai.Client(
        vertexai=True,
        project=PROJECT_ID,
        location=LOCATION,
    )


# ============================================================
# NODE 1 — RETRIEVE
# ============================================================

def retrieve_node(state: AgentState):

    print()
    print("NODE 1: RETRIEVE")
    print("-" * 60)

    question = state["question"]

    rewritten_question = state.get(
        "rewritten_question",
        "",
    )

    if rewritten_question:

        question_for_search = rewritten_question

        print(
            "Using rewritten question:",
            question_for_search,
        )

    else:

        question_for_search = question

        print(
            "Using original question:",
            question_for_search,
        )

    embedding_model = create_embedding_model()

    query_vector = create_query_embedding(
        embedding_model,
        question_for_search,
    )

    print(
        "Query vector dimensions:",
        len(query_vector),
    )

    qdrant_client = create_qdrant_client()

    results = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=QDRANT_TOP_K,
        with_payload=True,
    )

    qdrant_results = results.points

    print(
        f"Retrieved {len(qdrant_results)} "
        "results from Qdrant."
    )

    qdrant_client.close()

    return {
        "query_vector": query_vector,
        "qdrant_results": qdrant_results,
    }


# ============================================================
# NODE 2 — RERANK
# ============================================================

def rerank_node(state: AgentState):

    print()
    print("NODE 2: RERANK")
    print("-" * 60)

    question = state["question"]

    rewritten_question = state.get(
        "rewritten_question",
        "",
    )

    if rewritten_question:

        question_for_reranking = rewritten_question

    else:

        question_for_reranking = question

    qdrant_results = state[
        "qdrant_results"
    ]

    reranker = create_reranker()

    passages = []

    for result in qdrant_results:

        payload = result.payload

        passages.append(
            {
                "id": payload.get("chunk_id"),
                "text": payload.get(
                    "text",
                    "",
                ),
            }
        )

    rerank_request = RerankRequest(
        query=question_for_reranking,
        passages=passages,
    )

    reranked_results = reranker.rerank(
        rerank_request
    )

    print(
        f"FlashRank selected top "
        f"{FINAL_TOP_K} results."
    )

    for index, result in enumerate(
        reranked_results[:FINAL_TOP_K],
        start=1,
    ):

        print(
            f"{index}. "
            f"{result['id']} "
            f"(score={result['score']})"
        )

    return {
        "reranked_results": reranked_results,
    }


# ============================================================
# NODE 3 — CREATE CONTEXT
# ============================================================

def context_node(state: AgentState):

    print()
    print("NODE 3: CREATE CONTEXT")
    print("-" * 60)

    reranked_results = state[
        "reranked_results"
    ]

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

    print(
        f"Created context from "
        f"{FINAL_TOP_K} sources."
    )

    return {
        "context": context,
    }


# ============================================================
# NODE 4 — CHECK CONTEXT
# ============================================================

def check_context_node(state: AgentState):

    print()
    print("NODE 4: CHECK CONTEXT")
    print("-" * 60)

    question = state["question"]

    context = state["context"]

    llm_client = create_llm_client()

    prompt = f"""
You are evaluating retrieved documents
for an enterprise RAG system.

Determine whether the retrieved context
contains enough relevant information to
answer the user's question.

Return ONLY one word:

YES

or

NO


User Question:

{question}


Retrieved Context:

{context}
"""

    response = llm_client.models.generate_content(
        model=LLM_MODEL,
        contents=prompt,
    )

    decision = response.text.strip().upper()

    if "YES" in decision:

        decision = "YES"

    else:

        decision = "NO"

    print(
        "Context sufficient:",
        decision,
    )

    return {
        "context_relevant": decision,
    }


# ============================================================
# NODE 5 — REWRITE QUERY
# ============================================================

def rewrite_query_node(state: AgentState):

    print()
    print("NODE 5: REWRITE_QUERY")
    print("-" * 60)

    question = state["question"]

    context = state["context"]

    current_retry_count = state.get(
        "retry_count",
        0,
    )

    llm_client = create_llm_client()

    prompt = f"""
You are a query rewriting component
inside an enterprise RAG system.

The original question did not produce
sufficiently relevant retrieval results.

Rewrite the user's question so that it is
more precise and useful for semantic search.

Keep the same intent.

Return ONLY the rewritten question.


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

    rewritten_question = (
        response.text.strip()
    )

    new_retry_count = (
        current_retry_count + 1
    )

    print(
        "Rewritten question:",
        rewritten_question,
    )

    print(
        f"Retry count: "
        f"{new_retry_count}/{MAX_RETRIES}"
    )

    return {
        "rewritten_question":
            rewritten_question,

        "retry_count":
            new_retry_count,
    }


# ============================================================
# NODE 6 — GENERATE
# ============================================================

def generate_node(state: AgentState):

    print()
    print("NODE 6: GENERATE")
    print("-" * 60)

    question = state["question"]

    context = state["context"]

    llm_client = create_llm_client()

    prompt = f"""
You are an enterprise RAG assistant.

Answer the user's question using ONLY
the information provided in the context.

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

    response = llm_client.models.generate_content(
        model=LLM_MODEL,
        contents=prompt,
    )

    answer = response.text.strip()

    print()
    print("Generated answer:")
    print(answer)

    return {
        "answer": answer,
    }


# ============================================================
# NODE 7 — VALIDATE ANSWER
# ============================================================

def validate_answer_node(state: AgentState):

    print()
    print("NODE 7: VALIDATE ANSWER")
    print("-" * 60)

    question = state["question"]

    context = state["context"]

    answer = state["answer"]

    llm_client = create_llm_client()

    prompt = f"""
You are an answer validation component
inside an enterprise RAG system.

Your job is to determine whether the generated
answer is supported by the retrieved context.

Check TWO things:

1. Is the answer relevant to the user's question?

2. Is the answer supported by the retrieved context?

If the answer is relevant AND supported,
return:

YES

Otherwise return:

NO

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

    decision = response.text.strip().upper()

    if "YES" in decision:

        decision = "YES"

    else:

        decision = "NO"

    print(
        "Answer valid:",
        decision,
    )

    return {
        "answer_valid": decision,
    }


# ============================================================
# NODE 8 — SAFE FALLBACK
# ============================================================

def fallback_node(state: AgentState):

    print()
    print("NODE 8: SAFE FALLBACK")
    print("-" * 60)

    answer = (
        "I could not provide a sufficiently "
        "grounded answer from the provided "
        "documents."
    )

    print(
        "Safe fallback activated."
    )

    return {
        "answer": answer,
    }


# ============================================================
# CONTEXT ROUTER
# ============================================================

def route_after_check(state: AgentState):

    decision = state[
        "context_relevant"
    ]

    retry_count = state.get(
        "retry_count",
        0,
    )

    if decision == "YES":

        print()
        print(
            "ROUTER: Context is sufficient."
        )

        print(
            "ROUTER → GENERATE"
        )

        return "generate"

    else:

        print()
        print(
            "ROUTER: Context is insufficient."
        )

        print(
            f"Current retry count: "
            f"{retry_count}/{MAX_RETRIES}"
        )

        if retry_count < MAX_RETRIES:

            print(
                "ROUTER → REWRITE_QUERY"
            )

            return "rewrite_query"

        else:

            print(
                "ROUTER → FALLBACK"
            )

            return "fallback"


# ============================================================
# ANSWER VALIDATION ROUTER
# ============================================================

def route_after_validation(
    state: AgentState
):

    decision = state[
        "answer_valid"
    ]

    if decision == "YES":

        print()
        print(
            "ROUTER: Answer is grounded."
        )

        print(
            "ROUTER → END"
        )

        return "end"

    else:

        print()
        print(
            "ROUTER: Answer failed validation."
        )

        print(
            "ROUTER → FALLBACK"
        )

        return "fallback"


# ============================================================
# BUILD LANGGRAPH
# ============================================================

def build_graph():

    graph_builder = StateGraph(
        AgentState
    )

    # --------------------------------------------------------
    # Register nodes
    # --------------------------------------------------------

    graph_builder.add_node(
        "retrieve",
        retrieve_node,
    )

    graph_builder.add_node(
        "rerank",
        rerank_node,
    )

    graph_builder.add_node(
        "create_context",
        context_node,
    )

    graph_builder.add_node(
        "check_context",
        check_context_node,
    )

    graph_builder.add_node(
        "rewrite_query",
        rewrite_query_node,
    )

    graph_builder.add_node(
        "generate",
        generate_node,
    )

    graph_builder.add_node(
        "validate_answer",
        validate_answer_node,
    )

    graph_builder.add_node(
        "fallback",
        fallback_node,
    )

    # --------------------------------------------------------
    # Initial flow
    # --------------------------------------------------------

    graph_builder.add_edge(
        START,
        "retrieve",
    )

    graph_builder.add_edge(
        "retrieve",
        "rerank",
    )

    graph_builder.add_edge(
        "rerank",
        "create_context",
    )

    graph_builder.add_edge(
        "create_context",
        "check_context",
    )

    # --------------------------------------------------------
    # Context routing
    # --------------------------------------------------------

    graph_builder.add_conditional_edges(
        "check_context",
        route_after_check,
        {
            "generate": "generate",

            "rewrite_query":
                "rewrite_query",

            "fallback":
                "fallback",
        },
    )

    # --------------------------------------------------------
    # Query rewrite loop
    # --------------------------------------------------------

    graph_builder.add_edge(
        "rewrite_query",
        "retrieve",
    )

    # --------------------------------------------------------
    # Generate → Validate
    # --------------------------------------------------------

    graph_builder.add_edge(
        "generate",
        "validate_answer",
    )

    # --------------------------------------------------------
    # Answer validation routing
    # --------------------------------------------------------

    graph_builder.add_conditional_edges(
        "validate_answer",
        route_after_validation,
        {
            "end": END,

            "fallback":
                "fallback",
        },
    )

    # --------------------------------------------------------
    # Fallback → END
    # --------------------------------------------------------

    graph_builder.add_edge(
        "fallback",
        END,
    )

    return graph_builder.compile()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 80)
    print("AGENTIC RAG WITH LANGGRAPH")
    print("=" * 80)

    question = input(
        "Ask your question: "
    ).strip()

    if not question:

        print(
            "Question cannot be empty."
        )

        exit()

    # --------------------------------------------------------
    # Build graph
    # --------------------------------------------------------

    graph = build_graph()

    # --------------------------------------------------------
    # Initial state
    # --------------------------------------------------------

    initial_state = {

        "question": question,

        "query_vector": [],

        "qdrant_results": [],

        "reranked_results": [],

        "context": "",

        "context_relevant": "",

        "rewritten_question": "",

        "retry_count": 0,

        "answer": "",

        "answer_valid": "",
    }

    # --------------------------------------------------------
    # Execute graph
    # --------------------------------------------------------

    result = graph.invoke(
        initial_state
    )

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("FINAL RESULT")
    print("=" * 80)

    print()

    print(
        "FINAL CONTEXT DECISION:",
        result["context_relevant"],
    )

    print(
        "TOTAL QUERY REWRITES:",
        result.get(
            "retry_count",
            0,
        ),
    )

    print(
        "ANSWER VALIDATION:",
        result.get(
            "answer_valid",
            "",
        ),
    )

    if result.get(
        "rewritten_question"
    ):

        print()

        print(
            "LAST REWRITTEN QUESTION:"
        )

        print(
            result[
                "rewritten_question"
            ]
        )

    print()
    print("-" * 80)
    print("FINAL ANSWER")
    print("-" * 80)

    print()

    print(
        result["answer"]
    )

    print()
    print("=" * 80)
    print("AGENTIC RAG COMPLETED")
    print("=" * 80)