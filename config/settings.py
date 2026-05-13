"""
Zoo AI Chat 配置管理模块。
从 .env 文件加载配置，为整个应用提供类型化的配置访问。
"""
import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

_project_root = Path(__file__).parent.parent
load_dotenv(_project_root / ".env")

# HuggingFace 主站在国内代理后常被屏蔽，默认走 hf-mirror.com 镜像，
# 用户可通过 .env 中的 HF_ENDPOINT 覆盖。必须在导入 huggingface 库前设置。
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


def _resolve_bool_env(name: str, default: bool) -> bool:
    """支持 true/1/yes 与 false/0/no 写法的布尔解析；其他值落回默认。"""
    v = os.getenv(name, "").strip().lower()
    if v in ("true", "1", "yes"):
        return True
    if v in ("false", "0", "no"):
        return False
    return default


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
    debug: bool = _resolve_bool_env("DEBUG", True)

    # LLM 配置
    temperature: float = float(os.getenv("TEMPERATURE", "0.7"))
    max_tokens: int = int(os.getenv("MAX_TOKENS", "512"))

    # 数据文件路径
    data_dir: Path = _project_root / "data"
    faq_meetings_path: Path = data_dir / "faq_meetings.json"

    # 日志相关
    log_dir: Path = _project_root / "logs"
    app_log_path: Path = log_dir / "app.log"
    log_level: str = os.getenv("LOG_LEVEL", "INFO").upper()
    log_file_max_bytes: int = int(os.getenv("LOG_FILE_MAX_BYTES", str(10 * 1024 * 1024)))
    log_file_backup_count: int = int(os.getenv("LOG_FILE_BACKUP_COUNT", "3"))

    # RAG 日志配置
    rag_log_path: Path = _project_root / "logs" / "rag_logs.yaml"
    rag_log_level: str = os.getenv("RAG_LOG_LEVEL", "full")

    # /api/logs/tail 是否对外暴露：默认在 DEBUG=true 时开启
    log_http_tail: bool = _resolve_bool_env("LOG_HTTP_TAIL", debug)

    # Phase 4：产品线检测
    product_detection_enabled: bool = _resolve_bool_env("PRODUCT_DETECTION_ENABLED", True)
    default_collection: str = os.getenv("DEFAULT_COLLECTION", "zoo_faq_meetings")

    # Phase 4：Agent 模式
    agent_mode_enabled: bool = _resolve_bool_env("AGENT_MODE_ENABLED", False)
    agent_max_iterations: int = int(os.getenv("AGENT_MAX_ITERATIONS", "5"))

    # Phase 5：召回策略（vector | bm25 | hybrid）
    retrieval_strategy: str = os.getenv("RETRIEVAL_STRATEGY", "vector")
    rerank_enabled: bool = _resolve_bool_env("RERANK_ENABLED", False)
    bm25_weight: float = float(os.getenv("BM25_WEIGHT", "0.3"))
    vector_weight: float = float(os.getenv("VECTOR_WEIGHT", "0.7"))
    top_k: int = int(os.getenv("TOP_K", "3"))
    rerank_top_n: int = int(os.getenv("RERANK_TOP_N", "10"))
    rerank_model: str = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-base")


@lru_cache
def get_settings() -> Settings:
    """获取缓存的配置实例（单例模式）。"""
    return Settings()
