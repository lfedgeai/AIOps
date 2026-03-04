FROM python:3.10-slim

# Install drivers
RUN pip install --no-cache-dir mlflow boto3 psycopg2-binary cryptography

# Set the base executable
ENTRYPOINT ["mlflow"]

# Expose port
EXPOSE 5000
