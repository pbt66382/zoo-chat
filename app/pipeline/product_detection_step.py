"""
ProductDetectionStep（Phase 4）：检测用户问题所属产品线，设置 ctx.milvus_collection。

* 若 session 已有产品线记录（同一对话中用户明确了产品），直接复用，跳过 LLM 调用。
* 检测结果写入 ctx.product_id / product_name / product_confidence / milvus_collection，
  下游 RetrievalStep 会读取 milvus_collection 选择对应向量库。
"""
from __future__ import annotations

import asyncio
import logging

from app.pipeline.base import ChatContext
from app.product.detector import get_product_detector

logger = logging.getLogger("zoo_chat.pipeline.product_detection")


class ProductDetectionStep:
    name = "product_detection"

    async def run(self, ctx: ChatContext) -> None:
        # 已有产品线（来自 session 持久化）—— 跳过
        if ctx.product_id and ctx.milvus_collection:
            logger.debug(
                "product_detection_skip request_id=%s product_id=%s (from_session)",
                ctx.request_id,
                ctx.product_id,
            )
            return

        result = await asyncio.to_thread(get_product_detector().detect, ctx.question)

        ctx.product_id = result.product_id
        ctx.product_name = result.product_name
        ctx.product_confidence = result.confidence
        ctx.milvus_collection = result.collection

        logger.info(
            "product_detected request_id=%s product=%s(%s) conf=%.2f collection=%s",
            ctx.request_id,
            result.product_id,
            result.product_name,
            result.confidence,
            result.collection,
        )
