FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS base

ENV HF_HOME=/runpod-volume
ENV PYTHONPATH=/

# install python and other packages
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3-pip \
    git \
    wget \
    libgl1 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && python3.11 -m ensurepip --upgrade \
    && python3.11 -m pip install --no-cache-dir --upgrade pip

RUN pip install uv

# install python dependencies
COPY requirements.txt /requirements.txt
RUN uv pip install -r /requirements.txt --system

# Add application files
COPY handler.py .
COPY src ./src

# Add test input
COPY test_input.json .

# start the Runpod serverless worker
CMD python -u /handler.py
