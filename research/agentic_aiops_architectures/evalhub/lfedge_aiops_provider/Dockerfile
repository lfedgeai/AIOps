FROM registry.access.redhat.com/ubi9/python-311:latest

USER root
WORKDIR /app
COPY main.py /app/main.py
RUN chmod 755 /app/main.py && \
    pip install --no-cache-dir -U pip setuptools wheel

USER 1001
ENTRYPOINT ["python3", "-u", "/app/main.py"]
