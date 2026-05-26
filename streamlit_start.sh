#!/bin/bash

echo "=== Starting Streamlit Dashboard ==="
echo "Python version:"
python --version

echo "API URL: ${API_URL}"

# Start Streamlit
streamlit run streamlit_app.py \
  --server.port=${PORT:-8501} \
  --server.address=0.0.0.0 \
  --server.headless=true \
  --logger.level=info