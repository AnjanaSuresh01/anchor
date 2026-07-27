FROM python:3.13-slim

WORKDIR /app

# Dependencies first so a code edit doesn't invalidate the install layer.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY anchor/ ./anchor/

# The corpus is not baked in — it's generated, gitignored, and would go stale.
# docker-compose mounts ./data instead; run the ingest on the host first:
#   python -m anchor.ingest.arxiv_fetch --limit 200

# Bake the embedding model into the image rather than downloading it on first
# request — otherwise the first query pays a ~130MB cold start.
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5')"

EXPOSE 8000
CMD ["uvicorn", "anchor.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
