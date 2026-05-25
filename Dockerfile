FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=100 --retries=5 -r requirements.txt

COPY . .

EXPOSE 8000
EXPOSE $PORT   # This is a placeholder; Render will set it

CMD sh -c "uvicorn src.api:app --host 0.0.0.0 --port 8000 & streamlit run streamlit/app.py --server.port $PORT --server.address 0.0.0.0"