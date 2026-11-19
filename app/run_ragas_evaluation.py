
import json
from pathlib import Path

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = PROJECT_ROOT / "evaluation" / "ragas_inputs.json"
OUTPUT_PATH = PROJECT_ROOT / "evaluation" / "ragas_scores.csv"

# Vertex AI configuration
PROJECT_ID = "ai-security-bootcamp"
LOCATION = "us-central1"


def main():
    print("Loading evaluation inputs...")

    with INPUT_PATH.open("r", encoding="utf-8") as file:
        records = json.load(file)

    usable_records = [
        record
        for record in records
        if record.get("question", "").strip()
        and record.get("answer", "").strip()
        and record.get("contexts")
        and record.get("reference", "").strip()
    ]

    if not usable_records:
        raise ValueError("No usable evaluation records found.")

    print(f"Questions ready for evaluation: {len(usable_records)}")

    # Convert records into the format expected by RAGAS
    dataset = Dataset.from_list([
        {
            "question": record["question"],
            "answer": record["answer"],
            "contexts": record["contexts"],
            "reference": record["reference"],
        }
        for record in usable_records
    ])

    # Configure Gemini through Vertex AI
    vertex_llm = ChatVertexAI(
        model_name="gemini-2.5-flash",
        project=PROJECT_ID,
        location=LOCATION,
        temperature=0,
    )

    vertex_embeddings = VertexAIEmbeddings(
        model_name="text-embedding-005",
        project=PROJECT_ID,
        location=LOCATION,
    )

    evaluator_llm = LangchainLLMWrapper(vertex_llm)
    evaluator_embeddings = LangchainEmbeddingsWrapper(vertex_embeddings)

    print("Starting RAGAS evaluation...")
    print("Metrics: Faithfulness, Answer Relevancy, Context Precision, Context Recall")

    result = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
        raise_exceptions=True,
    )

    scores = result.to_pandas()
    scores.to_csv(OUTPUT_PATH, index=False)

    print("\nRAGAS evaluation completed.")
    print(f"Questions evaluated: {len(usable_records)}")
    print(f"Scores saved to: {OUTPUT_PATH}")
    print("\nIndividual question scores:")
    print(scores.to_string(index=False))

    metric_columns = [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
    ]
    available_metrics = [
        column for column in metric_columns if column in scores.columns
    ]

    print("\nAverage scores (0 to 1):")
    print(scores[available_metrics].mean().round(3).to_string())


if __name__ == "__main__":
    main()
