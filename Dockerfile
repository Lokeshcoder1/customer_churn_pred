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
sleep 3\n\
\n\
echo "--- FastAPI startup log ---"\n\
cat /tmp/uvicorn.log\n\
echo "----------------------------"\n\
\n\
if kill -0 $API_PID 2>/dev/null; then\n\
    echo "✅ FastAPI is running (PID: $API_PID)"\n\
else\n\
    echo "❌ FastAPI failed to start. Check the log above."\n\
    exit 1\n\
fi\n\
\n\
echo "Starting Streamlit on port ${PORT:-8501}..."\n\
streamlit run streamlit/app.py --server.port ${PORT:-8501} --server.address 0.0.0.0\n\
' > start.sh && chmod +x start.sh

# Set PYTHONPATH to ensure src module is found
ENV PYTHONPATH=/app

# Run the start script
CMD ./start.sh