FROM python:3.10-slim

# Install system dependencies for LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=100 --retries=5 -r requirements.txt
RUN pip install prometheus-client

# Copy the entire application code
COPY . .

# Debug: list the models directory to confirm model is present
RUN ls -la models/ || echo "No models directory found"

# Create the start script
RUN echo '#!/bin/bash\n\
echo "Starting FastAPI on port 8000..."\n\
uvicorn src.api:app --host 0.0.0.0 --port 8000 &> /tmp/uvicorn.log &\n\
API_PID=$!\n\
\n\
sleep 5\n\
\n\
echo "--- FastAPI startup log ---"\n\
cat /tmp/uvicorn.log\n\
echo "----------------------------"\n\
\n\
# Now test if the API is actually responding\n\
echo "Testing API health endpoint..."\n\
for i in 1 2 3 4 5; do\n\
    if curl -s --fail http://localhost:8000/health > /dev/null 2>&1; then\n\
        echo "✅ API health check passed"\n\
        break\n\
    fi\n\
    echo "Attempt $i: API not ready yet, waiting..."\n\
    sleep 2\n\
done\n\
\n\
if ! curl -s --fail http://localhost:8000/health > /dev/null 2>&1; then\n\
    echo "❌ API health check FAILED. Streamlit will likely show 'API not running'."\n\
    echo "Last 20 lines of uvicorn log:"\n\
    tail -20 /tmp/uvicorn.log\n\
    exit 1\n\
fi\n\
\n\
echo "Starting Streamlit on port ${PORT:-8501}..."\n\
streamlit run streamlit/app.py --server.port ${PORT:-8501} --server.address 0.0.0.0\n\
' > start.sh && chmod +x start.sh