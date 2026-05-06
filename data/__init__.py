"""
FAQ data module for Zoo Meetings product line.
Exports the FAQ_MEETINGS list from faq_meetings.json.
"""
import json
from pathlib import Path

_data_path = Path(__file__).parent / "faq_meetings.json"

with open(_data_path, "r", encoding="utf-8") as f:
    FAQ_MEETINGS = json.load(f)
