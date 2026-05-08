"""
LangChain FAQ Chain 模块 - Zoo 会议服务产品线（Phase 2 RAG）。
"""
import time
from typing import Optional

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from app.llm.deepseek_client import get_llm
from app.llm.embedding_client import get_embedding_client
from app.utils.rag_logger import RAGLog, log_rag_invocation
from config.settings import get_settings


SYSTEM_PROMPT_V2 = """你是一个专业的 Zoo 会议服务客服助手。

以下是与用户问题相关的 FAQ 参考内容：
{context}

请根据以上参考内容回答用户问题。如果找不到相关信息，请礼貌告知用户。
{history}"""


def _retrieve_docs(query: str, top_k: int = 3) -> tuple[list[Document], float, float]:
    """
    Milvus 向量检索。

    Returns:
        (docs, embedding_latency_ms, retrieval_latency_ms)
    """
    from pymilvus import MilvusClient

    settings = get_settings()
    embeddings = get_embedding_client()

    t0 = time.perf_counter()
    vector = embeddings.embed_query(query)
    t1 = time.perf_counter()
    embedding_latency_ms = (t1 - t0) * 1000

    client = MilvusClient(uri=f"http://{settings.milvus_host}:{settings.milvus_port}")
    t2 = time.perf_counter()
    results = client.search(
        collection_name=settings.milvus_collection,
        data=[vector],
        limit=top_k,
        output_fields=["faq_id", "tags", "text"],
    )
    t3 = time.perf_counter()
    retrieval_latency_ms = (t3 - t2) * 1000

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
    return docs, embedding_latency_ms, retrieval_latency_ms


def _format_context(docs: list[Document]) -> str:
    if not docs:
        return "（未找到相关 FAQ，请告知用户暂无相关信息）"
    lines = []
    for doc in docs:
        faq_id = doc.metadata.get("faq_id", "?")
        lines.append(f"【问题 {faq_id}】{doc.page_content}")
    return "\n".join(lines)


def build_rag_chain(history: str = "", prefetched_docs: list[Document] | None = None):
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT_V2),
        ("human", "{question}"),
    ])

    def _get_context(_):
        if prefetched_docs is not None:
            return _format_context(prefetched_docs)
        return _format_context(_retrieve_docs(_)[0])

    chain = (
        {
            "context": _get_context,
            "question": RunnablePassthrough(),
            "history": lambda _: f"【历史对话】\n{history}" if history else "【历史对话】（无）",
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


def invoke_rag_chain(question: str, history: str = "") -> str:
    """
    执行 RAG 流程，返回 LLM 生成的回答。
    流程结束后自动将召回情况写入日志文件。
    """
    from datetime import datetime, timezone

    t_start = time.perf_counter()
    request_id = f"req_{int(t_start * 1000)}"

    docs, embedding_ms, retrieval_ms = _retrieve_docs(question)

    chain = build_rag_chain(history=history, prefetched_docs=docs)
    answer = chain.invoke(question)

    t_end = time.perf_counter()
    total_ms = (t_end - t_start) * 1000

    scores = [d.metadata.get("score", 0.0) for d in docs]
    top_score = scores[0] if scores else 0.0
    score_gap = scores[0] - scores[1] if len(scores) >= 2 else 0.0

    log = RAGLog(
        timestamp=datetime.now(timezone.utc).isoformat(),
        request_id=request_id,
        question=question,
        history=history or "（无）",
        recalled_faqs=[
            {"faq_id": d.metadata.get("faq_id", "?"), "score": round(d.metadata.get("score", 0.0), 4), "tags": d.metadata.get("tags", "")}
            for d in docs
        ],
        retrieval_latency_ms=round(retrieval_ms, 2),
        embedding_latency_ms=round(embedding_ms, 2),
        total_latency_ms=round(total_ms, 2),
        llm_answer_length=len(answer),
        top_score=round(top_score, 4),
        score_gap=round(score_gap, 4),
    )
    log_rag_invocation(log)

    return answer


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
