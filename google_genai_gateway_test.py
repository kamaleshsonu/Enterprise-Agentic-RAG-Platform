import os
from google import genai


def main():
    gateway_key = os.getenv("PYDANTIC_AI_GATEWAY_API_KEY")

    if not gateway_key:
        raise RuntimeError("Gateway API key is missing.")

    gateway_base_url = os.getenv(
        "PYDANTIC_AI_GATEWAY_BASE_URL"
    )

    if not gateway_base_url:
        raise RuntimeError("Gateway base URL is missing.")

    route = "enterprise-gemini"

    client = genai.Client(
        enterprise=True,
        http_options={
            "base_url": f"{gateway_base_url.rstrip('/')}/{route}",
            "headers": {
                "Authorization": f"Bearer {gateway_key}"
            },
        },
    )

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents="Reply with exactly: Google Gen AI SDK Gateway successful.",
        )

        print("\nGateway response:")
        print(response.text)

    finally:
        client.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(
            f"\nGateway test failed: "
            f"{type(exc).__name__}: {exc}"
        )