"""
Chat 处理流水线（Phase 3）。

将原本耦合在 ``faq_chain.py`` 中的"检索 + 生成"流程拆解为可组合的 Step：
意图识别 -> 路由 -> 槽位填充 -> 检索 -> 生成。

每个 Step 实现 ``async def run(ctx: ChatContext) -> None``，按顺序执行，
任一步置 ``ctx.answer`` 或 ``ctx.needs_followup=True`` 即短路返回。
"""
from app.pipeline.base import ChatContext, ChatPipeline, PipelineStep

__all__ = ["ChatContext", "ChatPipeline", "PipelineStep"]
