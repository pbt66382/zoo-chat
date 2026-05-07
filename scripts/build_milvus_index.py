"""
构建 Milvus 向量索引脚本。
一次性运行，将 FAQ 数据导入 Milvus collection。
使用 pymilvus 2.x 原生 API，绕过 langchain_milvus ORM 兼容性问题。
"""
from pymilvus import MilvusClient, DataType

from app.llm.embedding_client import get_embedding_client
from data import FAQ_MEETINGS
from config.settings import get_settings


def build_milvus_index():
    """将 FAQ 数据导入 Milvus collection（幂等：先删除旧 collection 再重建）。"""
    settings = get_settings()
    embeddings = get_embedding_client()

    print(f"正在连接 Milvus ({settings.milvus_host}:{settings.milvus_port})...")
    client = MilvusClient(uri=f"http://{settings.milvus_host}:{settings.milvus_port}")

    if client.has_collection(settings.milvus_collection):
        print(f"删除旧 collection '{settings.milvus_collection}'...")
        client.drop_collection(settings.milvus_collection)

    dim = settings.embedding_dimension
    print(f"创建 collection '{settings.milvus_collection}' (dim={dim})...")
    client.create_collection(
        collection_name=settings.milvus_collection,
        dimension=dim,
        auto_id=True,
        enable_dynamic_field=True,
        vector_field_name="vector",
        index_params=[
            {
                "field_name": "vector",
                "index_type": "IVF_FLAT",
                "metric_type": "IP",
                "params": {"nlist": 128},
            }
        ],
    )

    # 构造要存入的文本和 metadata
    # 构造增强版检索文本：原始问答 + tags + 常见中文问法变体
    # 这样 all-MiniLM-L6-v2（英文优化）也能通过 tag/similarity 命中中文 query
    VARIATIONS = {
        1:  ["新建", "发起", "召开会议"],
        2:  ["加入", "进入", "参加"],
        3:  ["共享屏幕", "屏幕分享"],
        4:  ["预约", "预定", "安排"],
        5:  ["听不到", "没声音", "声音问题"],
        6:  ["麦克风", "说话对方听不到", "收音"],
        7:  ["开启摄像头", "打开视频", "视频"],
        8:  ["模糊", "画质差", "画面不清楚"],
        9:  ["邀请", "增加参会人", "拉人"],
        10: ["录制", "录屏", "录像"],
        11: ["查看录制", "录像存放", "回放"],
        12: ["密码", "会议码"],
        13: ["等候室", "等待室"],
        14: ["视图", "发言人", "切换画面"],
        15: ["白板", "共享白板"],
        16: ["举手", "举手发言"],
        17: ["虚拟背景", "美颜", "背景虚化"],
        18: ["断网", "网络断了", "网络中断", "连不上", "掉线"],
        19: ["修改时间", "改时间", "调整时间"],
        20: ["删除", "取消会议"],
        21: ["没有声音", "音频问题", "播放无声"],
        22: ["移出", "踢人", "移除参会人"],
        23: ["同声传译", "翻译", "多语言"],
        24: ["投票", "问卷调查"],
        25: ["会议纪要", "会议记录", "笔记"],
    }

    # 常见中文 FAQ 回答内容的关键词补充（让 embedding 更具区分性）
    ANSWER_KEYWORDS = {
        1:  "新建会议 即时会议 预约会议 会议链接",
        2:  "加入会议 会议号 邀请链接 会议邀请",
        3:  "共享屏幕 应用窗口 桌面",
        4:  "预约会议 时间 参会人 邮件提醒",
        5:  "听不到声音 音频 扬声器 耳机",
        6:  "麦克风 说话 收音 音频设置",
        7:  "视频 摄像头 开启视频",
        8:  "视频画质 模糊 分辨率 带宽",
        9:  "邀请参会人 拉人 邮箱 手机号",
        10: "录制会议 录屏 录像 保存",
        11: "录制文件 云端 我的录制 下载 分享",
        12: "会议密码 安全 会议码",
        13: "等候室 主持人 批准 进入会议室",
        14: "发言人视图 切换画面 画廊视图",
        15: "共享白板 画笔 标注",
        16: "举手发言 互动",
        17: "虚拟背景 背景图片 视频设置",
        18: "网络中断 断网 重连 检查网络 会议链接 重新加入",
        19: "修改时间 调整会议 重新预约",
        20: "删除会议 取消会议 通知参会人",
        21: "没有声音 音频 播放 扬声器",
        22: "移出参会人 踢人 移除",
        23: "同声传译 语言 翻译 声道",
        24: "会议投票 发起投票 问卷",
        25: "会议纪要 会议记录 会议总结 AI摘要",
    }

    def build_enriched_text(faq: dict) -> str:
        faq_id = faq["id"]
        variations = " ".join(VARIATIONS.get(faq_id, []))
        tags = " ".join(faq.get("tags", []))
        answer_kws = ANSWER_KEYWORDS.get(faq_id, "")
        original = f"{faq['question']} {faq['answer']}"
        return " ".join(filter(None, [variations, tags, answer_kws, original]))

    texts = [build_enriched_text(faq) for faq in FAQ_MEETINGS]
    metadatas = [
        {"faq_id": faq["id"], "tags": ",".join(faq.get("tags", []))}
        for faq in FAQ_MEETINGS
    ]

    print(f"正在计算 {len(texts)} 条 FAQ 的向量嵌入...")
    vectors = embeddings.embed_documents(texts)

    print(f"正在写入 {len(texts)} 条 FAQ 到 Milvus...")
    data = [
        {"vector": vector, "text": texts[i], **metadatas[i]}
        for i, vector in enumerate(vectors)
    ]
    client.insert(settings.milvus_collection, data)

    client.flush(settings.milvus_collection)
    count = client.query(settings.milvus_collection, output_fields=["count(*)"])
    print(f"Milvus 索引构建完成！共 {count[0]['count(*)']} 条 FAQ 已入库。")


if __name__ == "__main__":
    build_milvus_index()
