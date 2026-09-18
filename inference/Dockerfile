FROM runpod/pytorch:2.5.1-cuda12.4-cudnn9-devel

ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        wget curl && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY handler.py /app/

CMD ["python3", "-u", "handler.py"]
