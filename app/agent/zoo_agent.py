"""
Zoo ReAct Agent（Phase 4）。

核心概念
--------
* Agent = LLM + 工具（Tools） + 规划能力（ReAct 循环）
* ReAct = Reason（思考当前状态，下一步该做什么）+ Act（调用工具）+ Observe（观察结果）
* Tool Calling = LLM 输出结构化的工具调用指令，框架解析并执行，再把结果返回给 LLM

与 Phase 3 Pipeline 的区别
--------------------------
Pipeline：固定顺序执行 Step（意图 → 路由 → 槽位 → 检索 → 生成），每步之间无条件分支。
Agent：LLM 自主决定"接下来调用哪个工具"，可以多次检索、调用不同工具后再生成答案，
       具备动态规划能力，适合需要多步推理的复杂问题。

工具列表
--------
query_product_doc   语义检索产品知识库（RAG）
query_faq_exact     关键词精确匹配 FAQ 数据库
check_order_status  查询订单状态（模拟 API）
create_ticket       创建技术支持工单（模拟）
escalate_to_human   转接人工客服
"""
from __future__ import annotations

import json
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool

from app.llm.deepseek_client import get_llm
from app.llm.embedding_client import get_embedding_client
from config.settings import get_settings

logger = logging.getLogger("zoo_chat.agent")

# ---------------------------------------------------------------------------
# Agent 系统提示词
# ---------------------------------------------------------------------------
_AGENT_SYSTEM_PROMPT = """你是 Zoo 智能客服助手，负责解答关于 Zoo {product_name} 产品的用户问题。

可用工具：
- query_product_doc：在产品知识库中语义检索，获取最相关的文档片段。优先使用此工具。
- query_faq_exact：精确关键词匹配 FAQ，适合含有具体型号、错误码的问题。
- check_order_status：查询 Zoo 订单状态，用户询问订单/物流时使用。
- create_ticket：为无法立即解决的问题创建技术支持工单。
- escalate_to_human：当问题复杂、用户情绪激动或你没有把握时，转接人工客服。

工作原则：
1. 优先调用 query_product_doc 检索知识库，基于检索结果回答，不要编造信息。
2. 如果检索结果不足，可追加调用 query_faq_exact 补充查找。
3. 若最终仍无法回答，主动调用 create_ticket 或 escalate_to_human。
4. 回答用通俗亲切的中文，控制在 200 字以内。
5. 不要在最终回答中提到"工具"、"检索"、"知识库"等内部术语。"""


# ---------------------------------------------------------------------------
# Tool 工厂函数：构建绑定了特定 collection 的工具集
# ---------------------------------------------------------------------------

def _make_tools(collection: str) -> list:
    """为特定产品集合创建工具列表。"""
    settings = get_settings()
    milvus_uri = f"http://{settings.milvus_host}:{settings.milvus_port}"

    @tool
    def query_product_doc(query: str) -> str:
        """语义检索产品知识库，返回最相关的 FAQ 内容。使用场景：用户询问产品功能、操作步骤、故障排查时。"""
        try:
            from pymilvus import MilvusClient
            embeddings = get_embedding_client()
            vector = embeddings.embed_query(query)
            client = MilvusClient(uri=milvus_uri)
            results = client.search(
                collection_name=collection,
                data=[vector],
                limit=3,
                output_fields=["faq_id", "text"],
            )
            if not results or not results[0]:
                return "知识库中未找到相关内容。"
            items = []
            for hit in results[0]:
                entity = hit.get("entity", {})
                score = round(hit.get("distance", 0.0), 4)
                items.append(f"[相关度 {score}] {entity.get('text', '')}")
            return "\n\n".join(items)
        except Exception as e:
            logger.exception("query_product_doc_failed")
            return f"检索失败：{e}"

    @tool
    def query_faq_exact(keywords: str) -> str:
        """按关键词精确匹配 FAQ 数据库（适合包含型号、错误码等精确词的问题）。"""
        try:
            from pymilvus import MilvusClient
            client = MilvusClient(uri=milvus_uri)
            results = client.query(
                collection_name=collection,
                filter=f'text like "%{keywords}%"',
                output_fields=["faq_id", "text"],
                limit=3,
            )
            if not results:
                return f"未找到包含关键词「{keywords}」的 FAQ。"
            return "\n\n".join(f"【FAQ {r.get('faq_id', '?')}】{r.get('text', '')}" for r in results)
        except Exception as e:
            logger.exception("query_faq_exact_failed")
            return f"精确匹配失败：{e}"

    @tool
    def check_order_status(order_id: str) -> str:
        """查询 Zoo 产品订单状态（输入订单号，如 ZO-20240513-001）。"""
        # 模拟订单查询 API（实际项目中替换为真实 API 调用）
        mock_orders = {
            "ZO-20240513-001": {"status": "已发货", "carrier": "顺丰速运", "tracking": "SF1234567890", "eta": "明天"},
            "ZO-20240512-005": {"status": "已签收", "signed_at": "2024-05-12 14:30"},
            "ZO-20240514-010": {"status": "处理中", "note": "仓库备货中，预计 24 小时内发出"},
        }
        order = mock_orders.get(order_id.strip().upper())
        if order:
            details = "、".join(f"{k}: {v}" for k, v in order.items())
            return f"订单 {order_id} 状态：{details}"
        return f"未找到订单号 {order_id}，请确认订单号是否正确，或联系客服查询。"

    @tool
    def create_ticket(summary: str, priority: str = "normal") -> str:
        """为无法通过文档解决的问题创建技术支持工单。priority 可选 low/normal/high/urgent。"""
        ticket_id = f"TKT-{int(time.time())}-{uuid.uuid4().hex[:4].upper()}"
        logger.info("ticket_created id=%s priority=%s summary=%s", ticket_id, priority, summary[:50])
        priority_map = {"low": "低", "normal": "普通", "high": "高", "urgent": "紧急"}
        priority_cn = priority_map.get(priority, "普通")
        return (
            f"✅ 工单已创建！工单号：{ticket_id}\n"
            f"优先级：{priority_cn}\n"
            f"问题描述：{summary}\n"
            f"技术支持团队将在 2 个工作日内联系您，紧急问题将在 4 小时内响应。"
        )

    @tool
    def escalate_to_human(reason: str) -> str:
        """转接人工客服。当问题复杂、用户情绪激动或 AI 无法可靠回答时使用。"""
        logger.info("escalate_to_human reason=%s", reason[:80])
        return (
            f"正在为您转接人工客服（原因：{reason}）。\n"
            f"人工客服工作时间：周一至周五 9:00–18:00。\n"
            f"您也可以拨打服务热线：400-888-9999，或通过 zoo.cn/support 发起在线咨询。"
        )

    return [query_product_doc, query_faq_exact, check_order_status, create_ticket, escalate_to_human]


# ---------------------------------------------------------------------------
# AgentResult 数据类
# ---------------------------------------------------------------------------

@dataclass
class AgentResult:
    answer: str
    steps: list[dict[str, Any]] = field(default_factory=list)
    total_ms: float = 0.0
    tool_calls: int = 0


# ---------------------------------------------------------------------------
# ZooAgent 主类
# ---------------------------------------------------------------------------

class ZooAgent:
    """
    Zoo 客服 Agent，封装了 LangChain AgentExecutor。

    内部工作流（ReAct 循环）：
    1. LLM 收到用户问题 + 工具描述 → 决定调用哪个工具
    2. 框架执行工具，返回结果（Observation）
    3. LLM 看到 Observation，决定下一步（再调工具 or 生成最终答案）
    4. 重复 2–3 直到 LLM 输出最终答案 or 达到 max_iterations
    """

    def __init__(self, product_name: str, collection: str) -> None:
        self.product_name = product_name
        self.collection = collection
        self._executor: AgentExecutor | None = None

    def _get_executor(self) -> AgentExecutor:
        if self._executor is not None:
            return self._executor

        settings = get_settings()
        llm = get_llm(temperature=0.3)
        tools = _make_tools(self.collection)

        # Tool Calling Agent 使用模型内置的 function calling 能力
        # 比 ReAct（文本解析方式）更稳定，DeepSeek 原生支持
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=_AGENT_SYSTEM_PROMPT.format(product_name=self.product_name)),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ])

        agent = create_tool_calling_agent(llm, tools, prompt)
        self._executor = AgentExecutor(
            agent=agent,
            tools=tools,
            max_iterations=settings.agent_max_iterations,
            return_intermediate_steps=True,
            handle_parsing_errors=True,
            verbose=settings.debug,
        )
        return self._executor

    def chat(
        self,
        question: str,
        history: list[dict[str, str]] | None = None,
    ) -> AgentResult:
        """
        同步执行 Agent。

        history 格式：[{"role": "user"/"assistant", "content": "..."}]
        """
        t0 = time.perf_counter()

        # 将 history 转换为 LangChain 消息格式
        chat_history = []
        for msg in (history or []):
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                chat_history.append(HumanMessage(content=content))
            elif role == "assistant":
                chat_history.append(AIMessage(content=content))

        try:
            executor = self._get_executor()
            result = executor.invoke({
                "input": question,
                "chat_history": chat_history,
            })
            answer = result.get("output", "抱歉，我暂时无法回答您的问题，请联系人工客服。")
            intermediate = result.get("intermediate_steps", [])

            steps = []
            for action, observation in intermediate:
                steps.append({
                    "tool": getattr(action, "tool", "unknown"),
                    "input": getattr(action, "tool_input", {}),
                    "output": str(observation)[:200],
                })

        except Exception:
            logger.exception("agent_chat_failed question=%s", question[:50])
            answer = "抱歉，处理您的问题时遇到了错误，请稍后再试或联系人工客服。"
            steps = []

        total_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "agent_done product=%s question_len=%d steps=%d total_ms=%.0f",
            self.product_name, len(question), len(steps), total_ms,
        )
        return AgentResult(
            answer=answer,
            steps=steps,
            total_ms=round(total_ms, 2),
            tool_calls=len(steps),
        )
