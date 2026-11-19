import os

from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.gateway import gateway_provider


def main():
    api_key = os.getenv("PYDANTIC_AI_GATEWAY_API_KEY")

    if not api_key:
        raise RuntimeError("Gateway API key is missing.")

    provider = gateway_provider(
        "google-cloud",
        api_key=api_key,
        route="enterprise-gemini",
    )

    model = GoogleModel(
        "gemini-2.5-flash",
        provider=provider,
    )

    agent = Agent(model)

    result = agent.run_sync(
        "Reply with exactly: Gateway connection successful."
    )

    print("\nGateway response:")
    print(result.output)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nGateway test failed: {type(exc).__name__}: {exc}")