FROM python:3.13-slim

WORKDIR /app

# Dependencies first so a code edit doesn't invalidate the install layer.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY anchor/ ./anchor/
COPY data/corpus.jsonl ./data/corpus.jsonl

# Bake the embedding model into the image rather than downloading it on first
# request — otherwise the first query pays a ~130MB cold start.
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5')"

EXPOSE 8000
CMD ["uvicorn", "anchor.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
