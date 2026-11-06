"""Unit tests for configuration loader and environment settings."""

import os
from unittest.mock import patch

from src.utils.config import Settings, get_settings


def test_settings_defaults() -> None:
    """Verify production settings have valid defaults."""
    settings = Settings()
    assert settings.default_model_provider in ("openrouter", "nvidia")
    assert settings.model_temperature == 0.0
    assert settings.model_max_tokens > 0
    assert settings.db_pool_size > 0
    assert settings.max_steps_per_episode >= 5


def test_settings_alias_resolution() -> None:
    """Verify alias resolution for environment variable variants."""
    env_patch = {
        "SKIP_DOTENV": "1",
        "OPENROUTER_API": "test_or_key_123",
        "Database_url": "sqlite+aiosqlite:///:memory:",
        "LANGFUSE_BASE_URL": "https://custom.langfuse.com",
        "Public_Key": "pk-lf-test",
        "Secret_Key": "sk-lf-test",
    }
    # Clear explicit primary keys so alias takes precedence
    current_env = dict(os.environ)
    current_env.pop("LANGFUSE_HOST", None)
    current_env.pop("OPENROUTER_API_KEY", None)
    current_env.pop("DATABASE_URL", None)
    current_env.pop("LANGFUSE_PUBLIC_KEY", None)
    current_env.pop("LANGFUSE_SECRET_KEY", None)
    current_env.update(env_patch)

    with patch.dict(os.environ, current_env, clear=True):
        get_settings.cache_clear()
        settings = get_settings()
        assert settings.openrouter_api_key == "test_or_key_123"
        assert settings.database_url == "sqlite+aiosqlite:///:memory:"
        assert settings.langfuse_host == "https://custom.langfuse.com"
        assert settings.langfuse_public_key == "pk-lf-test"
        assert settings.langfuse_secret_key == "sk-lf-test"
        get_settings.cache_clear()
