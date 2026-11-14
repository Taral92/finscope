"""Application settings, loaded from environment variables and an optional `.env` file."""

import re
from decimal import Decimal
from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Deliberately loose: we only need to confirm a contact email is present, not validate it fully.
_EMAIL_PATTERN = re.compile(r"[\w.+-]+@[\w-]+(\.[\w-]+)+")


class Settings(BaseSettings):
    """FinScope configuration. Every field maps to a `FINSCOPE_`-prefixed env var."""

    model_config = SettingsConfigDict(
        env_prefix="FINSCOPE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        env_ignore_empty=True,  # KEY="" (as in .env.example) means "not set"
    )

    # SEC EDGAR rejects requests whose User-Agent lacks contact info,
    # e.g. "FinScope Jane Doe jane@example.com".
    sec_user_agent: str
    log_level: str = "INFO"
    # SEC's fair-access limit is 10 requests/second; staying a bit under leaves headroom.
    sec_max_requests_per_second: float = Field(default=8.0, gt=0, le=10)
    # Where downloaded filings and derived files are cached (relative to the working directory).
    data_dir: Path = Path("data")

    # LLM. The key keeps its conventional name OPENAI_API_KEY (no FINSCOPE_ prefix).
    # SecretStr masks it in repr/logs; read it only via .get_secret_value().
    openai_api_key: SecretStr | None = Field(default=None, validation_alias="OPENAI_API_KEY")
    # Model and standard-tier, short-context (<=272K input tokens) prices in USD per 1M tokens,
    # from https://developers.openai.com/api/docs/pricing, checked 2026-11-01.
    # Update all three together when changing the model; cost estimates are None without prices.
    llm_model: str = "gpt-6.1-terra"
    llm_input_price_per_mtok: Decimal | None = Decimal("1.00")
    llm_output_price_per_mtok: Decimal | None = Decimal("5.00")

    # Telegram alerts. The bot token (from @BotFather) is a password: SecretStr, never printed.
    # The chat ID says where alerts go; `python -m finscope telegram-chat-id` finds yours.
    telegram_bot_token: SecretStr | None = None
    telegram_chat_id: str | None = None

    @field_validator("sec_user_agent")
    @classmethod
    def _require_email(cls, value: str) -> str:
        if not _EMAIL_PATTERN.search(value):
            raise ValueError("must contain a contact email address, as required by SEC EDGAR")
        return value


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings, loading them on first call."""
    return Settings()
