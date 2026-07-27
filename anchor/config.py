"""Central configuration, loaded from environment / .env."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="ANCHOR_", extra="ignore"
    )

    # LLM
    llm_provider: str = "anthropic"
    llm_model: str = ""
    llm_temperature: float = 0.0

    # Qdrant. Empty url => embedded local-file mode (no Docker needed).
    qdrant_url: str = ""
    qdrant_path: str = "data/qdrant"
    collection: str = "anchor_chunks"

    # Retrieval
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    top_k: int = 6
    max_grader_retries: int = 2

    # Ingest
    chunk_chars: int = 1400
    chunk_overlap: int = 200
    corpus_path: str = "data/corpus.jsonl"

    @property
    def qdrant_location(self) -> Path:
        return REPO_ROOT / self.qdrant_path

    @property
    def corpus_file(self) -> Path:
        return REPO_ROOT / self.corpus_path


settings = Settings()
