#!/bin/bash
set -e

echo "=== Starting API ==="
python --version

echo "Testing imports..."
python -c "from src.api import app; print('✓ Import OK')"

echo "Starting on port ${PORT:-8000}..."

# Production start
exec gunicorn src.api:app \
  --workers 1 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:${PORT:-8000} \
  --timeout 120 \
  --access-logfile - \
  --error-logfile -