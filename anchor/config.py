"""Central configuration, loaded from environment / .env."""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parent.parent

# Settings below only picks up ANCHOR_*-prefixed keys, and it populates this
# object rather than the process environment. Provider SDKs read their keys
# straight from os.environ (ANTHROPIC_API_KEY, OPENROUTER_API_KEY, ...), so
# .env has to be loaded into the environment too or those lookups miss.
load_dotenv(REPO_ROOT / ".env")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="ANCHOR_", extra="ignore"
    )

    # LLM
    llm_provider: str = "anthropic"
    llm_model: str = ""
    llm_temperature: float = 0.0
    llm_base_url: str = ""
    """Override the API endpoint. Only used by OpenAI-compatible gateways
    (OpenRouter, Together, a local vLLM); ignored otherwise."""

    # Qdrant. Empty url => embedded local-file mode (no Docker needed).
    qdrant_url: str = ""
    qdrant_path: str = "data/qdrant"
    collection: str = "anchor_chunks"

    # Retrieval
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    top_k: int = 6
    max_grader_retries: int = 2
    enable_graph_route: bool = True
    """Whether the router may pick the resolved-entity graph. Turning it off is
    how the eval isolates what entity resolution actually contributes: the same
    agent runs with and without it, and the difference is the answer."""

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
