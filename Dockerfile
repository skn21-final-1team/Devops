FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS base

RUN pip install runpod sentence-transformers

RUN python -c "from sentence_transformers import CrossEncoder; CrossEncoder('BAAI/bge-reranker-v2-m3')"

COPY handler.py /handler.py
CMD [ "python", "-u", "/handler.py" ]