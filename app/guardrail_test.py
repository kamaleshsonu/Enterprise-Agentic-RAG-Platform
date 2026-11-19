from app.guardrail import check_input_guardrail


if __name__ == "__main__":

    print("=" * 60)
    print("INPUT GUARDRAIL TEST")
    print("=" * 60)

    question = input(
        "Enter your question: "
    ).strip()

    if not question:

        print("Question cannot be empty.")

        exit()

    decision = check_input_guardrail(
        question
    )

    print()
    print("GUARDRAIL DECISION:", decision)