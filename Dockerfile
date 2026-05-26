FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=100 --retries=5 -r requirements.txt
RUN pip install prometheus-client

COPY . .

# Streamlit will listen on Render's assigned PORT
CMD streamlit run streamlit/app.py --server.port ${PORT:-8501} --server.address 0.0.0.0