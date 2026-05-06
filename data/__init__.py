"""
Zoo 会议服务产品线 FAQ 数据模块。
从 faq_meetings.json 加载并导出 FAQ_MEETINGS 列表。
"""
import json
from pathlib import Path

_data_path = Path(__file__).parent / "faq_meetings.json"

with open(_data_path, "r", encoding="utf-8") as f:
    FAQ_MEETINGS = json.load(f)
