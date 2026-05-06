"""
Configuration management for Zoo AI Chat.
Loads settings from .env file and provides typed access throughout the app.
"""
import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

# Load .env file from project root
_project_root = Path(__file__).parent.parent
load_dotenv(_project_root / ".env")


class Settings:
    """Application settings loaded from environment variables."""

    # DeepSeek API
    deepseek_api_key: str = os.getenv("DEEPSEEK_API_KEY", "")
    deepseek_base_url: str = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    deepseek_model: str = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

    # Application
    app_host: str = os.getenv("APP_HOST", "0.0.0.0")
    app_port: int = int(os.getenv("APP_PORT", "8000"))
    debug: bool = os.getenv("DEBUG", "true").lower() in ("true", "1", "yes")

    # LLM Settings
    temperature: float = float(os.getenv("TEMPERATURE", "0.7"))
    max_tokens: int = int(os.getenv("MAX_TOKENS", "512"))

    # Data paths
    data_dir: Path = _project_root / "data"
    faq_meetings_path: Path = data_dir / "faq_meetings.json"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
