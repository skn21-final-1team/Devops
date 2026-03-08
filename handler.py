import runpod
import torch
from sentence_transformers import CrossEncoder

device = "cuda" if torch.cuda.is_available() else "cpu"
model = CrossEncoder("BAAI/bge-reranker-v2-m3", device=device)


def handler(job):
    """
    job["input"] 구조 예시:
    {
        "query": "What is AI?",
        "documents": ["AI is...", "Python is..."]
    }
    """
    job_input = job["input"]
    query = job_input.get("query")
    docs = job_input.get("documents")
    top_k = job_input.get(
        "top_k", len(docs)
    )  # top_k가 명시되지 않으면 모든 문서를 반환

    if not query or not docs:
        return {"error": "Query and documents are required."}

    pairs = [[query, doc] for doc in docs]
    scores = model.predict(pairs).tolist()

    results = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)[:top_k]

    return [{"document": doc, "score": float(score)} for doc, score in results]


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
