from typing import TypedDict

from langgraph.graph import StateGraph, START, END


# ============================================================
# STATE
# ============================================================

class AgentState(TypedDict):
    question: str
    message: str


# ============================================================
# NODE
# ============================================================

def process_question(state: AgentState):

    question = state["question"]

    message = f"LangGraph received the question: {question}"

    return {
        "message": message
    }


# ============================================================
# BUILD GRAPH
# ============================================================

graph_builder = StateGraph(AgentState)

graph_builder.add_node(
    "process_question",
    process_question,
)

graph_builder.add_edge(
    START,
    "process_question",
)

graph_builder.add_edge(
    "process_question",
    END,
)


# ============================================================
# COMPILE
# ============================================================

graph = graph_builder.compile()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    question = input(
        "Ask your question: "
    ).strip()

    result = graph.invoke(
        {
            "question": question,
            "message": "",
        }
    )

    print()
    print("=" * 80)
    print("LANGGRAPH RESULT")
    print("=" * 80)
    print()

    print(result["message"])