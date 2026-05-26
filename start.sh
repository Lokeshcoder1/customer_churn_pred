#!/bin/bash
echo "Starting FastAPI on port 8000..."
uvicorn src.api:app --host 0.0.0.0 --port 8000 &> /tmp/uvicorn.log &
API_PID=$!

sleep 5

echo "--- FastAPI startup log ---"
cat /tmp/uvicorn.log
echo "----------------------------"

echo "Testing API health endpoint..."
for i in 1 2 3 4 5; do
    if curl -s --fail http://localhost:8000/health > /dev/null 2>&1; then
        echo "✅ API health check passed"
        break
    fi
    echo "Attempt $i: API not ready yet, waiting..."
    sleep 2
done

if ! curl -s --fail http://localhost:8000/health > /dev/null 2>&1; then
    echo "❌ API health check FAILED. Streamlit will likely show 'API not running'."
    echo "Last 20 lines of uvicorn log:"
    tail -20 /tmp/uvicorn.log
    exit 1
fi

echo "Starting Streamlit on port ${PORT:-8501}..."
streamlit run streamlit/app.py --server.port ${PORT:-8501} --server.address 0.0.0.0