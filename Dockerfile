FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS base

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install runpod sentence-transformers

RUN python3 -c "from sentence_transformers import CrossEncoder; CrossEncoder('BAAI/bge-reranker-v2-m3')"

COPY handler.py /handler.py
CMD [ "python3", "-u", "/handler.py" ]