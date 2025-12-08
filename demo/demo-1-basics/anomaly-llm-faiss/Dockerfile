# Use a lightweight Python image
FROM python:3.11-slim


# Install build dependencies
RUN apt-get update && apt-get install -y curl \
    gcc \
    python3-dev \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user (OpenShift runs containers as random UID by default)
RUN mkdir -p /app /tmp/huggingface /mnt/hf_cache && chmod -R 777 /app /tmp/huggingface /mnt/hf_cache

# Set the working directory
WORKDIR /app

# Set environment variables for Hugging Face
ENV HF_HOME=/tmp/huggingface
ENV TRANSFORMERS_CACHE=/tmp/huggingface/transformers


# Copy requirement files and install dependencies
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application
COPY . .

# Set PYTHONPATH to ensure 'app' is recognized
ENV PYTHONPATH=/app

# Command to run the app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8002" , "--log-level", "debug", "--access-log" ]
