"""
LangChain FAQ Chain 模块 - Zoo 会议服务产品线（Phase 2 RAG）。
"""
from typing import Optional

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from app.llm.deepseek_client import get_llm
from app.llm.embedding_client import get_embedding_client
from config.settings import get_settings


SYSTEM_PROMPT_V2 = """你是一个专业的 Zoo 会议服务客服助手。

以下是与用户问题相关的 FAQ 参考内容：
{context}

请根据以上参考内容回答用户问题。如果找不到相关信息，请礼貌告知用户。
{history}"""


def _retrieve_docs(query: str, top_k: int = 3) -> list[Document]:
    """Milvus 向量检索。"""
    from pymilvus import MilvusClient

    settings = get_settings()
    embeddings = get_embedding_client()
    vector = embeddings.embed_query(query)
    client = MilvusClient(uri=f"http://{settings.milvus_host}:{settings.milvus_port}")
    results = client.search(
        collection_name=settings.milvus_collection,
        data=[vector],
        limit=top_k,
        output_fields=["faq_id", "tags", "text"],
    )
    docs = []
    for hit in results[0]:
        entity = hit.get("entity", {})
        docs.append(Document(
            page_content=entity.get("text", ""),
            metadata={
                "faq_id": entity.get("faq_id", "?"),
                "tags": entity.get("tags", ""),
                "score": hit.get("distance", 0.0),
            },
        ))
    return docs


def _format_context(docs: list[Document]) -> str:
    if not docs:
        return "（未找到相关 FAQ，请告知用户暂无相关信息）"
    lines = []
    for doc in docs:
        faq_id = doc.metadata.get("faq_id", "?")
        lines.append(f"【问题 {faq_id}】{doc.page_content}")
    return "\n".join(lines)


def build_rag_chain(history: str = ""):
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT_V2),
        ("human", "{question}"),
    ])
    chain = (
        {
            "context": lambda query: _format_context(_retrieve_docs(query)),
            "question": RunnablePassthrough(),
            "history": lambda _: f"【历史对话】\n{history}" if history else "【历史对话】（无）",
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


def invoke_rag_chain(question: str, history: str = "") -> str:
    chain = build_rag_chain(history=history)
    return chain.invoke(question)


# Phase 1 旧代码（向后兼容）
from data import FAQ_MEETINGS


def _build_faq_context() -> str:
    lines = []
    for faq in FAQ_MEETINGS:
        lines.append(f"【问题{faq['id']}】{faq['question']}")
        lines.append(f"【回答】{faq['answer']}")
        lines.append("")
    return "\n".join(lines)


FAQ_CONTEXT = _build_faq_context()


def find_faq_by_question(question: str) -> Optional[dict]:
    q_lower = question.lower()
    for faq in FAQ_MEETINGS:
        keywords = [tag.lower() for tag in faq.get("tags", [])]
        if any(kw in q_lower for kw in keywords):
            return faq
    return None
