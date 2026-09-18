# Polygon — RunPod Serverless Workers

## Inference Endpoint
Serves the chain-trained Qwen3.8-27B model via OpenAI-compatible API.
- Handler: `inference/handler.py`
- Dockerfile: `inference/Dockerfile`
- GPU: A6000 48GB (or A100 80GB)

## Conversion Endpoint
A2D diffusion conversion with GaLore optimizer.
- Handler: `conversion/handler.py`
- Dockerfile: `conversion/Dockerfile`
- GPU: A100 80GB required

## Usage
Deploy via RunPod serverless endpoints. Each directory is a self-contained worker.
