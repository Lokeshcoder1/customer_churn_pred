FROM python:3.10-slim

# Install system dependencies for LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=100 --retries=5 -r requirements.txt

# Install prometheus-client separately (add to requirements.txt later)
RUN pip install prometheus-client

COPY . .

# Tell Docker the container listens on port 8501 (Streamlit).
# NOTE: The EXPOSE instruction does NOT support variable substitution.
EXPOSE 8501

# The HEALTHCHECK verifies if the app is responding.
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run both FastAPI (internal, port 8000) and Streamlit (external, port 8501).
CMD sh -c "uvicorn src.api:app --host 0.0.0.0 --port 8000 & exec streamlit run streamlit/app.py --server.port ${PORT:-8501} --server.address 0.0.0.0"