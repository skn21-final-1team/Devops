from typing import Any

import runpod

from src.embedding_service import VllmEmbeddingService
from src.utils import create_error_response

try:
    embedding_service = VllmEmbeddingService()
except Exception as e:
    import sys

    sys.stderr.write(f"\nstartup failed: {e}\n")
    sys.exit(1)


async def async_generator_handler(job: dict[str, Any]):
    job_input = job["input"]
    try:
        if job_input.get("openai_route"):
            openai_route, openai_input = (
                job_input.get("openai_route"),
                job_input.get("openai_input"),
            )

            if openai_route and openai_route == "/v1/embeddings":
                model_name = openai_input.get("model")
                if not openai_input:
                    return create_error_response("Missing input")
                if not model_name:
                    return create_error_response(
                        "Did not specify model in openai_input"
                    )
                return [r.to_dict() for r in embedding_service.embed(openai_input.get("input"))]
            else:
                return create_error_response(
                    f"Invalid OpenAI Route: {openai_route}"
                )
        else:
            if job_input.get("input"):
                return [r.to_dict() for r in embedding_service.embed(job_input.get("input"))]
            else:
                return create_error_response(f"Invalid input: {job}")

    except Exception as e:
        return create_error_response(str(e))


if __name__ == "__main__":
    runpod.serverless.start({"handler": async_generator_handler})
