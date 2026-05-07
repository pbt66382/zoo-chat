"""
Embedding 客户端封装模块。
使用 OpenAI 兼容接口连接 DeepSeek Embedding 服务。
"""
from langchain_huggingface import HuggingFaceEmbeddings

from config.settings import get_settings


def get_embedding_client():
    """
    获取配置好的 Embedding 客户端（HuggingFace 本地模型）。

    返回:
        配置好的 HuggingFaceEmbeddings 实例
    """
    settings = get_settings()

    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
