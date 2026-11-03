FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY run.py report.py ./
COPY scenarios scenarios
COPY generator generator
COPY jev jev
COPY evaluation evaluation
COPY llm llm
COPY prompts prompts
# Credentials are passed at run time (-e JEV_API_KEY=...). Results go to /app/results.
ENTRYPOINT ["python", "run.py"]
