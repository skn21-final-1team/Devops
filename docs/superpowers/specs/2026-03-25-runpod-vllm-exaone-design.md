# RunPod Serverless vLLM EXAONE Serving Design

## Overview

RunPod Serverless에서 `LGAI-EXAONE/EXAONE-4.0-32B-FP8` 모델을 vLLM으로 서빙하는 시스템.
공식 `runpod/worker-v1-vllm` 이미지를 베이스로 모델을 빌드 시점에 베이킹한다.

## Architecture

```
[Client (OpenAI SDK)]
        |
        v
[RunPod Serverless Endpoint]
        |
        v
[runpod/worker-v1-vllm Container]
  - vLLM 0.16.0 (OpenAI 호환 서버)
  - EXAONE-4.0-32B-FP8 (빌드 시 베이킹)
  - 1x H100 80GB, TENSOR_PARALLEL_SIZE=1
```

## Key Decisions

| 항목 | 결정 | 이유 |
|------|------|------|
| 서빙 방식 | RunPod 공식 worker-v1-vllm | 코드 작성 불필요, OpenAI streaming 내장 |
| 모델 로딩 | 빌드 시 베이킹 | cold start 최소화 |
| GPU | 1x H100 80GB | FP8 32B 모델에 충분한 VRAM |
| MAX_MODEL_LEN | 16384 | 긴 문서 처리 가능하면서 VRAM 절약 |
| Base Image | `runpod/worker-v1-vllm:v2.14.0` | vLLM 0.16.0, CUDA 12.9.1 |

## Components

### Dockerfile

`runpod/worker-v1-vllm:v2.14.0` 이미지를 확장하여 모델 가중치를 포함시킨다.

- Base: `runpod/worker-v1-vllm:v2.14.0` (vLLM 0.16.0, CUDA 12.9.1)
- HF_TOKEN을 Docker build secret으로 전달 (보안)
- 이미지 내장 `/src/download_model.py` 스크립트로 모델 다운로드
- 환경변수로 vLLM 설정 주입
- 예상 이미지 크기: ~40GB (베이스 8.2GB + 모델 ~32GB)

### 내장 라이브러리 (공식 이미지 포함)

| 라이브러리 | 버전 | 비고 |
|-----------|------|------|
| vLLM | 0.16.0 | EXAONE 4.0 지원 (>= 0.10.0) |
| PyTorch | 2.x + CUDA 12.9.1 | H100 FP8 네이티브 지원 |
| transformers | >= 4.54.0 | EXAONE 4.0 지원 요구사항 |
| runpod SDK | 내장 | serverless handler |
| OpenAI 호환 API | 내장 | /openai/v1/chat/completions streaming |

### 불필요 파일 (삭제 대상)

- `handler.py` — 공식 이미지에 핸들러 내장 (`/src/handler.py`)
- `requirements.txt` — 공식 이미지에 모든 의존성 포함

### .dockerignore

빌드 컨텍스트에서 불필요한 파일 제외.

## Environment Variables

```
MODEL_NAME=LGAI-EXAONE/EXAONE-4.0-32B-FP8
MAX_MODEL_LEN=16384
TENSOR_PARALLEL_SIZE=1
GPU_MEMORY_UTILIZATION=0.90
```

## Build & Deploy

```bash
# Build (Docker BuildKit 필수)
DOCKER_BUILDKIT=1 docker build \
  --secret id=HF_TOKEN,env=HF_TOKEN \
  -t <registry>/vllm-exaone:latest .

# Push
docker push <registry>/vllm-exaone:latest

# RunPod 콘솔에서 Serverless Endpoint 생성
# - Docker Image: <registry>/vllm-exaone:latest
# - GPU: 1x H100 80GB
```

## Client Usage

```python
from openai import OpenAI

client = OpenAI(
    api_key="RUNPOD_API_KEY",
    base_url="https://api.runpod.ai/v2/<ENDPOINT_ID>/openai/v1",
)

stream = client.chat.completions.create(
    model="LGAI-EXAONE/EXAONE-4.0-32B-FP8",
    messages=[{"role": "user", "content": "Hello"}],
    stream=True,
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

## Sources

- [runpod-workers/worker-vllm GitHub](https://github.com/runpod-workers/worker-vllm)
- [runpod/worker-v1-vllm Docker Hub](https://hub.docker.com/r/runpod/worker-v1-vllm)
- [LGAI-EXAONE/EXAONE-4.0-32B-FP8 HuggingFace](https://huggingface.co/LGAI-EXAONE/EXAONE-4.0-32B-FP8)
- [vLLM Documentation](https://docs.vllm.ai/en/latest/)
