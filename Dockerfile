FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Ensure both scripts are executable (just in case)
RUN chmod +x start.sh streamlit_start.sh

# Expose the port Streamlit will use
EXPOSE 8501

# Run the Streamlit start script
CMD ["./streamlit_start.sh"]