FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=100 --retries=5 -r requirements.txt
RUN pip install prometheus-client

COPY . .

ENV PYTHONPATH=/app

COPY start.sh .
RUN chmod +x start.sh

CMD ./start.sh