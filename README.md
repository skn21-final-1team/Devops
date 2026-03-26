# RunPod Serverless Workers

RAG 파이프라인을 위한 RunPod Serverless Worker 모음. 각 브랜치가 하나의 독립된 워커입니다.

## 워커 목록

| 브랜치 | 모델 | 추론 엔진 | 배포 방식 | 역할 |
|--------|------|----------|----------|------|
| `embedding` | BAAI/bge-m3 | vLLM | GitHub Repository | 텍스트 임베딩 생성 |
| `exaone` | EXAONE-4.0-32B | SGLang | GitHub Repository | LLM 채팅/생성 |
| `post-processing` | Llama-3.1-8B-Instruct | vLLM | GitHub Repository | 후처리 추론 |
| `rerank` | bge-reranker-v2-m3 | Sentence Transformers | GitHub Repository | 문서 리랭킹 |
| `vllm-exaone` | EXAONE-4.0-32B-FP8 | vLLM (RunPod 공식) | Docker Hub | LLM 채팅/생성 |

---

## Embedding

> BAAI/bge-m3 기반 텍스트 임베딩 생성 워커

**API 요청**

```json
{
  "input": {
    "model": "BAAI/bge-m3",
    "input": "임베딩할 텍스트"
  }
}
```

**API 응답**

```json
[
  {
    "vector": [0.123, -0.456, ...],
    "prompt_token_count": 5
  }
]
```

**환경변수**

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `HF_HOME` | `/runpod-volume` | 모델 캐시 경로 |

---

## Exaone

> EXAONE-4.0-32B 기반 LLM 채팅/생성 워커 (스트리밍 지원)

**API 요청**

```json
{
  "input": {
    "openai_route": "/v1/chat/completions",
    "openai_input": {
      "model": "LGAI-EXAONE/EXAONE-4.0-32B",
      "messages": [{"role": "user", "content": "안녕하세요"}],
      "stream": false,
      "max_tokens": 100
    }
  }
}
```

**API 응답**: OpenAI Chat Completion 형식

**환경변수**

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `MODEL_NAME` | `LGAI-EXAONE/EXAONE-4.0-32B` | 모델명 |
| `PORT` | `30000` | SGLang 서버 포트 |
| `TENSOR_PARALLEL_SIZE` | - | 텐서 병렬 크기 |
| `HF_TOKEN` | - | HuggingFace 액세스 토큰 |

---

## Post-Processing

> Llama-3.1-8B-Instruct 기반 후처리 추론 워커

**API 요청**

```json
{
  "input": {
    "openai_route": "/v1/chat/completions",
    "openai_input": {
      "model": "meta-llama/Llama-3.1-8B-Instruct",
      "messages": [{"role": "user", "content": "후처리할 텍스트"}],
      "max_tokens": 100
    }
  }
}
```

**API 응답**: OpenAI Chat Completion 형식

**환경변수**

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `MODEL_NAME` | `meta-llama/Llama-3.1-8B-Instruct` | 모델명 |
| `GPU_MEMORY_UTILIZATION` | `0.92` | GPU 메모리 사용률 |
| `MAX_MODEL_LEN` | `8192` | 최대 시퀀스 길이 |
| `MAX_CONCURRENCY` | `50` | 최대 동시 요청 수 |

---

## Rerank

> bge-reranker-v2-m3 기반 문서 리랭킹 워커

**API 요청**

```json
{
  "input": {
    "query": "검색 쿼리",
    "documents": ["문서1", "문서2", "문서3"],
    "top_k": 2
  }
}
```

**API 응답**

```json
[
  {"document": "문서1", "score": 0.95},
  {"document": "문서3", "score": 0.72}
]
```

**환경변수**: 별도 설정 불필요 (CUDA 자동 감지)

---

## vLLM-Exaone

> EXAONE-4.0-32B-FP8 기반 LLM 채팅/생성 워커 (FP8 양자화)

**API 요청**: OpenAI SDK 호환

```python
from openai import OpenAI

client = OpenAI(
    api_key="RUNPOD_API_KEY",
    base_url="https://api.runpod.ai/v2/<ENDPOINT_ID>/openai/v1",
)

response = client.chat.completions.create(
    model="LGAI-EXAONE/EXAONE-4.0-32B-FP8",
    messages=[{"role": "user", "content": "안녕하세요"}],
)
```

**환경변수**

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `MODEL_NAME` | `LGAI-EXAONE/EXAONE-4.0-32B-FP8` | 모델명 |
| `MAX_MODEL_LEN` | `8192` | 최대 시퀀스 길이 |
| `GPU_MEMORY_UTILIZATION` | `0.90` | GPU 메모리 사용률 |
| `TENSOR_PARALLEL_SIZE` | `1` | 텐서 병렬 크기 |
