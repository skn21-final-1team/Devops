import logging

import requests
import runpod

from engine import SGlangEngine
from utils import process_response

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


def log_request_creation(endpoint: str, payload: dict):
    stream_enabled = payload.get("stream", False)
    payload_keys = sorted(payload.keys())
    logger.info(
        "Creating request | endpoint=%s stream=%s payload_keys=%s",
        endpoint,
        stream_enabled,
        payload_keys,
    )


engine = SGlangEngine()
engine.start_server()
engine.wait_for_server()


async def async_handler(job):
    """Handle the requests asynchronously."""
    job_input = job["input"]

    print("핸들러 시작")
    if job_input.get("openai_route"):
        openai_route, openai_input = (
            job_input.get("openai_route"),
            job_input.get("openai_input"),
        )

        openai_url = f"{engine.base_url}" + openai_route
        headers = {"Content-Type": "application/json"}

        log_request_creation(openai_url, openai_input)
        response = requests.post(openai_url, headers=headers, json=openai_input)
        logger.info(
            "Response received | endpoint=%s status_code=%s",
            openai_url,
            response.status_code,
        )
        # Process the streamed response
        if openai_input.get("stream", False):
            for formated_chunk in process_response(response):
                yield formated_chunk
        else:
            for chunk in response.iter_lines():
                if chunk:
                    decoded_chunk = chunk.decode("utf-8")
                    yield decoded_chunk

    elif "messages" in job_input:
        openai_url = f"{engine.base_url}/v1/chat/completions"
        headers = {"Content-Type": "application/json"}

        if "model" not in job_input:
            job_input["model"] = engine.model or "default"

        log_request_creation(openai_url, job_input)
        response = requests.post(openai_url, headers=headers, json=job_input)
        logger.info(
            "Response received | endpoint=%s status_code=%s",
            openai_url,
            response.status_code,
        )

        if job_input.get("stream", False):
            for formated_chunk in process_response(response):
                yield formated_chunk
        else:
            for chunk in response.iter_lines():
                if chunk:
                    yield chunk.decode("utf-8")

    else:
        generate_url = f"{engine.base_url}/generate"
        headers = {"Content-Type": "application/json"}
        log_request_creation(generate_url, job_input)
        response = requests.post(generate_url, json=job_input, headers=headers)
        logger.info(
            "Response received | endpoint=%s status_code=%s",
            generate_url,
            response.status_code,
        )

        if response.status_code == 200:
            yield response.json()
        else:
            yield {
                "error": f"Generate request failed with status code {response.status_code}",
                "details": response.text,
            }


runpod.serverless.start(
    {
        "handler": async_handler,
        "concurrency_modifier": 300,
        "return_aggregate_stream": True,
    }
)
