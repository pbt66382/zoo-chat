"""
Zoo AI Chat 配置管理模块。
从 .env 文件加载配置，为整个应用提供类型化的配置访问。
"""
import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

# 从项目根目录加载 .env 文件
_project_root = Path(__file__).parent.parent
load_dotenv(_project_root / ".env")


class Settings:
    """从环境变量加载的应用配置。"""

    # DeepSeek API 配置
    deepseek_api_key: str = os.getenv("DEEPSEEK_API_KEY", "")
    deepseek_base_url: str = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    deepseek_model: str = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

    # Embedding 配置（复用 DeepSeek OpenAI 兼容接口）
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    embedding_dimension: int = int(os.getenv("EMBEDDING_DIMENSION", "1536"))

    # Milvus 向量数据库配置
    milvus_host: str = os.getenv("MILVUS_HOST", "localhost")
    milvus_port: int = int(os.getenv("MILVUS_PORT", "19530"))
    milvus_collection: str = os.getenv("MILVUS_COLLECTION", "zoo_faq_collection")

    # 应用配置
    app_host: str = os.getenv("APP_HOST", "0.0.0.0")
    app_port: int = int(os.getenv("APP_PORT", "8000"))
    debug: bool = os.getenv("DEBUG", "true").lower() in ("true", "1", "yes")

    # LLM 配置
    temperature: float = float(os.getenv("TEMPERATURE", "0.7"))
    max_tokens: int = int(os.getenv("MAX_TOKENS", "512"))

    # 数据文件路径
    data_dir: Path = _project_root / "data"
    faq_meetings_path: Path = data_dir / "faq_meetings.json"


@lru_cache
def get_settings() -> Settings:
    """获取缓存的配置实例（单例模式）。"""
    return Settings()
