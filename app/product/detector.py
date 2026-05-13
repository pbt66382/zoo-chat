"""
产品线检测器（Phase 4）：识别用户问题属于哪个 Zoo 产品线。

设计思路
--------
* 先做关键词快速预判（零延迟），再用 LLM 做语义确认。
* 两者结果一致时提升置信度；LLM 输出无效时退回关键词结果。
* 产品线结果在 session 内持久——同一轮对话无需重复检测。

产品线一览
----------
meetings  会议服务       Zoo Meetings / Zoom 功能
phone     可视电话       Zoo Phone 硬件话机
earbuds   耳机           Zoo 蓝牙/USB 耳机
mouse     鼠标           Zoo 无线鼠标
screen    会议大屏       Zoo Room 会议室大屏
calls     通话服务       Zoo Phone 云通话/呼叫中心
general   通用           账号/订阅/价格/其他
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from functools import lru_cache

from langchain_core.messages import HumanMessage, SystemMessage

from app.llm.deepseek_client import get_llm

logger = logging.getLogger("zoo_chat.product")

# ---------------------------------------------------------------------------
# 产品线定义（ID → 元数据）
# ---------------------------------------------------------------------------
PRODUCT_LINES: dict[str, dict] = {
    "meetings": {
        "name": "会议服务",
        "description": "Zoo Meetings 视频会议、会议发起/入会、屏幕共享、主持人控制、会议预约",
        "keywords": ["会议", "zoom", "入会", "共享屏幕", "主持人", "静音", "摄像头", "虚拟背景", "预约会议", "会议室"],
        "collection": "zoo_faq_meetings",
    },
    "phone": {
        "name": "可视电话",
        "description": "Zoo Phone 硬件可视话机、SIP 注册、固件升级、话机激活",
        "keywords": ["电话", "话机", "ip phone", "固件", "激活", "sip", "拨号", "分机", "外线", "座机"],
        "collection": "zoo_faq_phone",
    },
    "earbuds": {
        "name": "耳机",
        "description": "Zoo 蓝牙耳机、USB 耳机、配对连接、音质、降噪",
        "keywords": ["耳机", "蓝牙", "配对", "音质", "耳麦", "降噪", "usb 耳机", "无线耳机"],
        "collection": "zoo_faq_earbuds",
    },
    "mouse": {
        "name": "鼠标",
        "description": "Zoo 无线鼠标、大屏控制、批注笔、指针",
        "keywords": ["鼠标", "无线鼠标", "批注", "指针", "遥控", "大屏控制"],
        "collection": "zoo_faq_mouse",
    },
    "screen": {
        "name": "会议大屏",
        "description": "Zoo Room 会议室大屏、投屏、白板、触摸屏、无法入会",
        "keywords": ["大屏", "投屏", "白板", "会议室大屏", "触摸屏", "room", "无线投屏"],
        "collection": "zoo_faq_screen",
    },
    "calls": {
        "name": "通话服务",
        "description": "Zoo 云通话、呼叫中心、语音服务、IVR、中继线路",
        "keywords": ["通话服务", "呼叫中心", "云通话", "ivr", "语音信箱", "中继", "坐席"],
        "collection": "zoo_faq_calls",
    },
    "general": {
        "name": "通用",
        "description": "账号管理、登录、订阅、价格、通用问题",
        "keywords": ["账号", "登录", "密码", "订阅", "价格", "注册", "发票", "退款"],
        "collection": "zoo_faq_meetings",   # 通用问题走会议库
    },
}

_DETECTION_PROMPT = """你是 Zoo 智能客服产品线分类器。根据用户问题，从以下产品线中选择最匹配的一个。

产品线列表：
{product_list}

规则：
1. 只返回 JSON，不要有任何其他文字。
2. confidence 为 0.0~1.0，代表你对分类结果的把握程度。
3. 无法判断时选 general，confidence 给 0.5。

用户问题：{question}

返回格式：{{"product_id": "<产品ID>", "confidence": <0.0~1.0>}}"""


@dataclass
class ProductResult:
    product_id: str
    product_name: str
    confidence: float
    collection: str


class ProductDetector:
    """产品线检测器：关键词预判 + LLM 语义确认双重校验。"""

    def __init__(self, llm=None) -> None:
        self._llm = llm

    def _get_llm(self):
        if self._llm is None:
            self._llm = get_llm(temperature=0.0, max_tokens=64)
        return self._llm

    def _keyword_detect(self, question: str) -> str | None:
        """关键词匹配，返回命中最多关键词的产品 ID（排除 general）。"""
        question_lower = question.lower()
        best_pid, best_count = None, 0
        for pid, meta in PRODUCT_LINES.items():
            if pid == "general":
                continue
            count = sum(1 for kw in meta["keywords"] if kw in question_lower)
            if count > best_count:
                best_count, best_pid = count, pid
        return best_pid if best_count > 0 else None

    def detect(self, question: str) -> ProductResult:
        """检测产品线，返回 ProductResult。"""
        keyword_hit = self._keyword_detect(question)

        product_list = "\n".join(
            f"- {pid}: {meta['name']}（{meta['description']}）"
            for pid, meta in PRODUCT_LINES.items()
        )
        prompt = _DETECTION_PROMPT.format(product_list=product_list, question=question)

        try:
            resp = self._get_llm().invoke([
                SystemMessage(content="你是产品线分类器，只返回 JSON，不要有多余文字。"),
                HumanMessage(content=prompt),
            ])
            raw = getattr(resp, "content", str(resp)).strip()
            m = re.search(r'\{[^}]+\}', raw)
            if m:
                data = json.loads(m.group())
                pid = data.get("product_id", "general")
                conf = float(data.get("confidence", 0.5))
                if pid not in PRODUCT_LINES:
                    pid = keyword_hit or "general"
                    conf = 0.5
                # 两者一致时提升置信度
                if keyword_hit and keyword_hit == pid:
                    conf = min(1.0, conf + 0.15)
                elif keyword_hit and pid == "general":
                    pid, conf = keyword_hit, 0.7
            else:
                pid = keyword_hit or "general"
                conf = 0.5
        except Exception:
            logger.exception("product_detect_llm_failed, fallback to keyword")
            pid = keyword_hit or "general"
            conf = 0.5

        meta = PRODUCT_LINES[pid]
        logger.info("product_detected pid=%s conf=%.2f kw_hit=%s", pid, conf, keyword_hit)
        return ProductResult(
            product_id=pid,
            product_name=meta["name"],
            confidence=conf,
            collection=meta["collection"],
        )


@lru_cache(maxsize=1)
def get_product_detector() -> ProductDetector:
    return ProductDetector()
