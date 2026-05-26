FROM python:3.10-slim

# Install system dependencies for LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
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
COPY start.sh .
RUN chmod +x start.sh

# Set PYTHONPATH to ensure src module is found
ENV PYTHONPATH=/app

CMD ./start.sh