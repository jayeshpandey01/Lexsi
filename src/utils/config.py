"""Configuration loader for production-grade agent runtime.

Loads and validates environment variables from .env and system environment,
supporting cloud LLM gateways (OpenRouter, NVIDIA NIM), databases (Turso, SQLite, Postgres),
and observability platforms (Langfuse).
"""

from functools import lru_cache
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field


def _load_env_file() -> None:
    """Find and load .env file from project root if present and not skipped."""
    if os.environ.get("SKIP_DOTENV") == "1":
        return
    root_dir = Path(__file__).resolve().parent.parent.parent
    env_path = root_dir / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=False)
    else:
        load_dotenv(override=False)


class Settings(BaseModel):
    """Production settings schema with fallback defaults and environment variable resolution."""

    model_config = ConfigDict(extra="ignore")

    # LLM Inference Providers
    openrouter_api_key: Optional[str] = Field(default=None)
    nvidia_api_key: Optional[str] = Field(default=None)
    groq_api_key: Optional[str] = Field(default=None)
    hf_token: Optional[str] = Field(default=None)
    cohere_api_key: Optional[str] = Field(default=None)

    # Active Model Routing
    default_model_provider: str = Field(default="openrouter")
    default_model_name: str = Field(default="nvidia/nemotron-3.5-lightning:free")
    fallback_model_provider: str = Field(default="openrouter")
    fallback_model_name: str = Field(default="google/gemma-4-31b-it:free")

    model_temperature: float = Field(default=0.0)
    model_max_tokens: int = Field(default=2048)
    model_timeout_seconds: float = Field(default=30.0)
    model_max_retries: int = Field(default=3)
    model_retry_backoff_factor: float = Field(default=1.5)

    # Database Configuration (Turso / SQLite / PostgreSQL)
    database_url: str = Field(default="sqlite+aiosqlite:///./data/lexsi_production.db")
    turso_auth_token: Optional[str] = Field(default=None)
    allow_local_db_fallback: bool = Field(default=False)
    db_pool_size: int = Field(default=20)
    db_max_overflow: int = Field(default=10)
    db_pool_timeout: float = Field(default=30.0)
    db_echo_sql: bool = Field(default=False)
    real_dataset_path: str = Field(default="./data/enterprise_supply_chain.db")

    # Observability & Tracing (Langfuse)
    enable_langfuse_tracing: bool = Field(default=True)
    langfuse_public_key: Optional[str] = Field(default=None)
    langfuse_secret_key: Optional[str] = Field(default=None)
    langfuse_host: str = Field(default="https://cloud.langfuse.com")

    # Multi-Tenant Runtime Limits
    max_steps_per_episode: int = Field(default=10)
    per_call_tool_timeout: float = Field(default=5.0)
    rate_limit_rpm: int = Field(default=60)
    max_concurrent_sessions: int = Field(default=50)
    enable_strict_redaction: bool = Field(default=True)

    # Training & Artifacts
    base_model_name_or_path: str = Field(default="CohereLabs/aya-expanse-8b")
    hf_repo_id: Optional[str] = Field(default=None)
    training_output_dir: str = Field(default="./results/checkpoints")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Retrieve cached production settings with alias resolution."""
    _load_env_file()

    # Resolve aliases gracefully
    openrouter_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENROUTER_API")
    nvidia_key = os.environ.get("NVIDIA_API_KEY")
    groq_key = os.environ.get("GROQ_API_KEY")
    hf_token = os.environ.get("HF_TOKEN")
    cohere_key = os.environ.get("COHERE_API_KEY")

    db_url = os.environ.get("DATABASE_URL") or os.environ.get("Database_url") or "sqlite+aiosqlite:///./data/lexsi_production.db"
    turso_token = os.environ.get("TURSO_AUTH_TOKEN") or os.environ.get("TURSO_TOKEN")

    lf_pub = (
        os.environ.get("LANGFUSE_PUBLIC_KEY")
        or os.environ.get("Public_Key")
    )
    lf_sec = (
        os.environ.get("LANGFUSE_SECRET_KEY")
        or os.environ.get("Secret_Key")
    )
    lf_host = (
        os.environ.get("LANGFUSE_HOST")
        or os.environ.get("LANGFUSE_BASE_URL")
        or "https://cloud.langfuse.com"
    )

    enable_lf = os.environ.get("ENABLE_LANGFUSE_TRACING", "true").lower() in ("true", "1", "yes")

    return Settings(
        openrouter_api_key=openrouter_key,
        nvidia_api_key=nvidia_key,
        groq_api_key=groq_key,
        hf_token=hf_token,
        cohere_api_key=cohere_key,
        default_model_provider=os.environ.get("DEFAULT_MODEL_PROVIDER", "openrouter"),
        default_model_name=os.environ.get("DEFAULT_MODEL_NAME") or "nvidia/nemotron-3.5-lightning:free",
        fallback_model_provider=os.environ.get("FALLBACK_MODEL_PROVIDER", "openrouter"),
        fallback_model_name=os.environ.get("FALLBACK_MODEL_NAME", "google/gemma-4-31b-it:free"),
        model_temperature=float(os.environ.get("MODEL_TEMPERATURE", "0.0")),
        model_max_tokens=int(os.environ.get("MODEL_MAX_TOKENS", "2048")),
        model_timeout_seconds=float(os.environ.get("MODEL_TIMEOUT_SECONDS", "30.0")),
        model_max_retries=int(os.environ.get("MODEL_MAX_RETRIES", "3")),
        model_retry_backoff_factor=float(os.environ.get("MODEL_RETRY_BACKOFF_FACTOR", "1.5")),
        database_url=db_url,
        turso_auth_token=turso_token,
        allow_local_db_fallback=os.environ.get("ALLOW_LOCAL_DB_FALLBACK", "false").lower() in ("true", "1", "yes"),
        db_pool_size=int(os.environ.get("DB_POOL_SIZE", "20")),
        db_max_overflow=int(os.environ.get("DB_MAX_OVERFLOW", "10")),
        db_pool_timeout=float(os.environ.get("DB_POOL_TIMEOUT", "30.0")),
        db_echo_sql=os.environ.get("DB_ECHO_SQL", "false").lower() in ("true", "1", "yes"),
        real_dataset_path=os.environ.get("REAL_DATASET_PATH", "./data/enterprise_supply_chain.db"),
        enable_langfuse_tracing=enable_lf,
        langfuse_public_key=lf_pub,
        langfuse_secret_key=lf_sec,
        langfuse_host=lf_host,
        max_steps_per_episode=int(os.environ.get("MAX_STEPS_PER_EPISODE", "10")),
        per_call_tool_timeout=float(os.environ.get("PER_CALL_TOOL_TIMEOUT", "5.0")),
        rate_limit_rpm=int(os.environ.get("RATE_LIMIT_RPM", "60")),
        max_concurrent_sessions=int(os.environ.get("MAX_CONCURRENT_SESSIONS", "50")),
        enable_strict_redaction=os.environ.get("ENABLE_STRICT_REDACTION", "true").lower() in ("true", "1", "yes"),
        base_model_name_or_path=os.environ.get("BASE_MODEL_NAME_OR_PATH", "CohereLabs/aya-expanse-8b"),
        hf_repo_id=os.environ.get("HF_REPO_ID"),
        training_output_dir=os.environ.get("TRAINING_OUTPUT_DIR", "./results/checkpoints"),
    )
