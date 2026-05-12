"""
槽位（Slot）定义：仅故障排查类意图需要追加结构化信息后再去检索。

通过收集 ``device_type`` / ``scenario`` / ``network_type`` 等关键信息，
让 RAG 检索时能拼出更具区分度的 query，提升召回精度。
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SlotDef:
    name: str
    question: str  # 追问用户的话术
    examples: tuple[str, ...] = ()  # 给提示用


# 各意图的必填槽位（按顺序追问）
SLOT_SCHEMA: dict[str, tuple[SlotDef, ...]] = {
    "troubleshoot_audio": (
        SlotDef(
            name="device_type",
            question="请问您使用的是哪种音频设备？例如：电脑内置麦克风、外接耳机、蓝牙耳机、专业麦克风等。",
            examples=("电脑内置麦克风", "蓝牙耳机", "USB 耳麦"),
        ),
        SlotDef(
            name="scenario",
            question="请问问题出现在什么时候？仅在 Zoo 会议中出现，还是其他应用也有？是入会就没有声音，还是会议中途突然没声音？",
            examples=("仅 Zoo 会议中", "会议中途突然没声音", "所有应用都没声音"),
        ),
    ),
    "troubleshoot_video": (
        SlotDef(
            name="device_type",
            question="请问您使用的是哪种摄像头？例如：笔记本内置摄像头、外接 USB 摄像头、手机摄像头、专业相机等。",
            examples=("笔记本内置摄像头", "外接 USB 摄像头", "手机前置摄像头"),
        ),
        SlotDef(
            name="scenario",
            question="请问具体表现是什么？是完全打不开、画面很模糊、画面卡顿，还是其他应用能用但 Zoo 不能用？",
            examples=("完全打不开", "画面卡顿", "画面模糊", "Zoo 中不能用其他应用可以"),
        ),
    ),
    "troubleshoot_network": (
        SlotDef(
            name="network_type",
            question="请问您当前使用的是哪种网络？Wi-Fi、有线网络、4G/5G 移动网络，还是 VPN？",
            examples=("公司 Wi-Fi", "家里有线", "手机热点 5G", "VPN"),
        ),
        SlotDef(
            name="scenario",
            question="请问问题表现是什么？是完全连不上会议、入会后频繁掉线，还是画面/声音一直卡顿？",
            examples=("完全连不上", "频繁掉线", "声音和画面卡顿", "网络速度慢"),
        ),
    ),
}


def get_slot_schema(intent_id: str) -> tuple[SlotDef, ...]:
    return SLOT_SCHEMA.get(intent_id, ())


def next_missing_slot(intent_id: str, slots: dict[str, str]) -> SlotDef | None:
    """按 schema 定义的顺序找第一个还没填的槽位。"""
    for slot in get_slot_schema(intent_id):
        if not slots.get(slot.name):
            return slot
    return None
