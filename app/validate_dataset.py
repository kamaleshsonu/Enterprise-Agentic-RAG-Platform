
import json
from pathlib import Path


def validate_dataset():
    dataset_path = (
        Path(__file__).resolve().parent.parent
        / "evaluation"
        / "dataset.json"
    )

    with dataset_path.open("r", encoding="utf-8") as file:
        dataset = json.load(file)

    if not isinstance(dataset, list) or not dataset:
        raise ValueError("Dataset must be a non-empty JSON list.")

    for index, item in enumerate(dataset, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Item {index} must be an object.")

        if not item.get("question", "").strip():
            raise ValueError(f"Item {index} has no question.")

        if not item.get("reference_answer", "").strip():
            raise ValueError(f"Item {index} has no reference answer.")

    print("SUCCESS: Evaluation dataset is valid.")
    print(f"Number of evaluation questions: {len(dataset)}")


if __name__ == "__main__":
    validate_dataset()
