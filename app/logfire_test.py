
import logfire

# Connect Python application to Logfire
logfire.configure()

print("Logfire configuration initialized.")

# Create a test trace
with logfire.span(
    "Enterprise Agentic RAG - Smoke Test",
    component="observability",
    project="enterprise-rag",
):
    logfire.info(
        "First telemetry event successfully sent."
    )

    logfire.info(
        "Testing Pydantic Logfire integration."
    )

# Send buffered telemetry
flushed = logfire.force_flush()

print("Telemetry flushed:", flushed)
print("Logfire test completed.")
