# FROM python:3.11-slim

# WORKDIR /app
# COPY . .

# RUN pip install fastapi uvicorn psutil

# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# COPY app.py /app/app.py
COPY . .

RUN pip install --no-cache-dir fastapi uvicorn psutil

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
