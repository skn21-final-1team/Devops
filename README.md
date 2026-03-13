# Devops

Runpod serverless worker that exposes a vLLM-backed OpenAI embeddings-compatible interface.

## Runtime

This project is serverless-only.
The container starts [`handler.py`](/Users/tera/Desktop/workspace/final/devops/handler.py), which boots the Runpod serverless handler.

## Request shape

The worker accepts the standard OpenAI embeddings payload in `event.input`.

```json
{
  "input": {
    "model": "Qwen/Qwen3-Embedding-8B",
    "input": "hello world"
  }
}
```

## Response shape

Successful responses are serialized from vLLM's OpenAI protocol embedding models and follow the default embeddings schema.
