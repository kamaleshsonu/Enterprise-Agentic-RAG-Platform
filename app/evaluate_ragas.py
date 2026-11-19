
import json
import re
from pathlib import Path

from app.agentic_rag import build_graph


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "evaluation" / "dataset.json"
OUTPUT_PATH = PROJECT_ROOT / "evaluation" / "ragas_inputs.json"


def create_initial_state(question: str) -> dict:
    """Create the initial state expected by the existing LangGraph."""

    return {
        "question": question,
        "input_safe": "",
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


def split_context(context: str) -> list[str]:
    """Convert the combined context into individual passages."""

    if not context.strip():
        return []

    passages = re.split(r"\n\s*SOURCE \d+\s*\n", context)

    return [
        passage.strip()
        for passage in passages
        if passage.strip()
    ]


def main():
    with DATASET_PATH.open("r", encoding="utf-8") as file:
        questions = json.load(file)

    graph = build_graph()
    evaluation_records = []

    for index, item in enumerate(questions, start=1):
        question = item["question"]

        print(f"\nEvaluating question {index}/{len(questions)}")
        print(f"Question: {question}")

        result = graph.invoke(create_initial_state(question))

        context = result.get("context", "")
        answer = result.get("answer", "")

        record = {
            "question": question,
            "answer": answer,
            "contexts": split_context(context),
            "reference": item["reference_answer"],
            "source_document": item.get("source_document", ""),
            "verification_status": item.get(
                "verification_status", "pending_sme"
            ),
        }

        evaluation_records.append(record)

        print(f"Answer: {answer}")
        print(f"Retrieved passages: {len(record['contexts'])}")

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(
            evaluation_records,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("\nEvaluation inputs saved successfully.")
    print(f"Questions processed: {len(evaluation_records)}")
    print(f"Output file: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
