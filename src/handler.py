import os
import sys
import multiprocessing
import subprocess
import traceback
from typing import Mapping

import runpod
from runpod import RunPodLogger

log = RunPodLogger()

vllm_engine = None
openai_engine = None
LOCAL_MODEL_ARGS_PATH = "/local_model_args.json"


class ModelBootstrapper:
    """런타임 시작 시 모델 다운로드와 메타데이터 준비를 담당한다."""

    def __init__(self, model_name: str | None) -> None:
        self._model_name = model_name

    def bootstrap(self) -> None:
        """환경 변수 기준으로 모델을 준비한다."""
        if not self._model_name:
            log.info("MODEL_NAME is not set. Skipping runtime model download.")
            return

        log.info(f"Starting runtime model download for MODEL_NAME={self._model_name}")
        subprocess.run(["python3", "/src/download_model.py"], check=True)

        if not os.path.exists(LOCAL_MODEL_ARGS_PATH):
            raise FileNotFoundError(
                f"Model metadata was not generated: {LOCAL_MODEL_ARGS_PATH}"
            )
        log.info("Runtime model download completed successfully")


async def handler(job: Mapping[str, object]):
    """RunPod 추론 요청을 처리하고 결과 배치를 스트리밍한다."""
    try:
        from utils import JobInput
        job_input = JobInput(job["input"])
        engine = openai_engine if job_input.openai_route else vllm_engine
        results_generator = engine.generate(job_input)
        async for batch in results_generator:
            yield batch
    except Exception as e:
        error_str = str(e)
        full_traceback = traceback.format_exc()

        log.error(f"Error during inference: {error_str}")
        log.error(f"Full traceback:\n{full_traceback}")

        # CUDA errors = worker is broken, exit to let RunPod spin up a healthy one
        if "CUDA" in error_str or "cuda" in error_str:
            log.error("Terminating worker due to CUDA/GPU error")
            sys.exit(1)

        yield {"error": error_str}


# Only run in main process to prevent re-initialization when vLLM spawns worker subprocesses
if __name__ == "__main__" or multiprocessing.current_process().name == "MainProcess":

    try:
        ModelBootstrapper(os.getenv("MODEL_NAME")).bootstrap()

        from engine import vLLMEngine, OpenAIvLLMEngine

        vllm_engine = vLLMEngine()
        openai_engine = OpenAIvLLMEngine(vllm_engine)
        log.info("vLLM engines initialized successfully")
    except Exception as e:
        log.error(f"Worker startup failed: {e}\n{traceback.format_exc()}")
        sys.exit(1)

    runpod.serverless.start(
        {
            "handler": handler,
            "concurrency_modifier": lambda x: vllm_engine.max_concurrency if vllm_engine else 1,
            "return_aggregate_stream": True,
        }
    )
