# Zoo Chat - Phase 2 RAG 升级设计文档

> 日期：2026-05-07（初稿） / 2026-05-12（同步现状）
> Phase 1 → Phase 2 完整升级记录

---

## 一、升级目标

Phase 1 采用**全量 FAQ 注入**方式：将全部 FAQ 直接拼入 Prompt，LLM 从中检索回答。
这种方式存在明显缺陷：FAQ 数量增长时 Prompt 膨胀、检索精度低、LLM 容易"幻觉"。

Phase 2 引入**向量检索增强生成（RAG）**架构，改为只召回最相关的 Top-3 条 FAQ 再送 LLM 生成回答。

---

## 二、架构对比

### Phase 1 架构

```
用户输入 → find_faq_by_question（关键词 tag 匹配）
         → 拼接所有匹配 FAQ → Prompt → DeepSeek LLM → 回答
```

特点：
- FAQ 全部注入 Prompt，无向量检索
- 匹配逻辑：遍历 FAQ tags，逐条做字符串 `in` 判断
- 无向量数据库依赖

### Phase 2 架构（实际落地）

```
用户输入 → Embedding（BAAI/bge-m3，1024 维）
         → Milvus 向量相似度检索 Top-3
         → 组装 context + history → Prompt → DeepSeek LLM → 回答
```

新增组件：
- **Milvus** 向量数据库（存储 150 条 FAQ 的向量）
- **HuggingFace Embedding** 本地模型 `BAAI/bge-m3`（中文友好，1024 维）
- **LangChain RAG Chain** 替代原有的全量注入逻辑
- 检索增强：每条 FAQ 入库时拼接"问法变体 + tags + 答案关键词"提升召回命中

---

## 三、新增依赖

### Python 包（requirements.txt）

| 包名 | 版本 | 用途 |
|------|------|------|
| `langchain` / `langchain-core` / `langchain-community` | >= 0.3.0 | LangChain 核心框架 |
| `openai` | >= 1.0.0 | `ChatOpenAI` 内部依赖（DeepSeek 用 OpenAI 兼容接口） |
| `langchain-huggingface` | 最新 | HuggingFace Embedding 封装 |
| `sentence-transformers` | >= 2.2.0 | bge-m3 模型加载所需 |
| `pymilvus` | >= 2.3.0 | Milvus Python SDK（直接使用，不通过 langchain-milvus） |
| `langchain-milvus` | >= 0.1.0 | 备用，本期未直接使用 |
| `fastapi` | >= 0.110.0 | Web 框架 |
| `uvicorn[standard]` | >= 0.29.0 | ASGI 服务器 |
| `httpx` | >= 0.27.0 | HTTP 客户端 |
| `pyyaml` | >= 6.0 | RAG 日志 YAML 序列化 |

> 注：`langchain_community.chat_models.ChatOpenAI` 仍依赖 `openai` SDK，必须显式安装；
> 之前的 requirements.txt 注释错写"无需 openai"，已修正。

### 向量数据库（Docker）

Phase 2 使用 Milvus All-in-One 三件套：

| 容器名 | 镜像 | 端口 | 用途 |
|--------|------|------|------|
| `milvus-etcd` | quay.io/coreos/etcd:v3.5.5 | 2379 | 元数据存储 |
| `milvus-minio` | minio/minio:RELEASE.2023-03-20T20-16-18Z | 9000 | 对象存储（S3 兼容） |
| `milvus-standalone` | milvusdb/milvus:v2.3.3 | 9091 / 19530 | Milvus 主服务 |

启动命令：

```bash
docker compose -f docker-compose-milvus.yml up -d
```

---

## 四、文件清单

### 核心代码

| 文件 | 说明 |
|------|------|
| `app/llm/embedding_client.py` | HuggingFace Embedding 客户端封装 |
| `app/llm/deepseek_client.py` | DeepSeek LLM 客户端（OpenAI 兼容） |
| `app/chains/faq_chain.py` | Phase 2 RAG Chain（Phase 3 后大部分逻辑迁移到 `app/pipeline/`） |
| `app/api/chat.py` | `POST /api/chat` 接口 |
| `app/main.py` | FastAPI 应用入口 |
| `app/utils/rag_logger.py` | RAG 调用结构化日志（YAML） |

### 脚本与配置

| 文件 | 说明 |
|------|------|
| `scripts/build_milvus_index.py` | 索引构建脚本：把 150 条 FAQ + 增强词向量化后写入 Milvus |
| `data/faq_meetings.json` | 150 条 Zoo 会议服务 FAQ 数据 |
| `data/faq_enrichment.json` | 每条 FAQ 的"问法变体 + 答案关键词"（Phase 3 重构时从脚本抽出） |
| `docker-compose-milvus.yml` | Milvus 容器编排 |
| `config/settings.py` | 集中配置：DeepSeek / Embedding / Milvus / 日志 |
| `.env` | 敏感配置（API key）+ 模型/端点覆盖 |

---

## 五、数据流详解

### 索引构建流程（一次性）

```
data/faq_meetings.json（150 条 FAQ）
  + data/faq_enrichment.json（变体 + 答案关键词）
  → scripts/build_milvus_index.py
    → HuggingFaceEmbeddings(BAAI/bge-m3, 1024 维)
    → 每条 FAQ 拼接 "变体 + tags + 答案关键词 + 原始问答"
    → pymilvus（IVF_FLAT + 内积度量）
    → Milvus collection: zoo_faq_collection
```

enriched 检索文本示例（FAQ ID=18，网络中断）：

```
断网 网络断了 网络中断 网络 连不上 掉线
troubleshoot_connectivity network 断网
网络中断 断网 重连 检查网络 会议链接 重新加入
会议中断网了怎么办 如果网络中断：1) 检查本地网络...
```

### 推理流程（每次请求）

```
1. POST /api/chat { "message": "网络断了怎么办" }
2. RetrievalStep:
     - embed_query("网络断了怎么办") → 1024 维向量
     - Milvus.search(top_k=3) → Top-3 相似 FAQ
3. GenerationStep:
     - 组装 system prompt：意图提示 + Top-3 FAQ + 历史
     - DeepSeek LLM 生成回答
4. ChatService:
     - 写 RAG 日志（recalled_faqs / latency / top_score）
     - 返回 { "answer": "...", "session_id": "...", "latency_ms": ... }
```

---

## 六、环境变量配置

`.env` 关键项：

```bash
# DeepSeek API
DEEPSEEK_API_KEY=sk-...
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat

# Embedding（HuggingFace 本地模型，1024 维）
EMBEDDING_MODEL=BAAI/bge-m3
EMBEDDING_DIMENSION=1024

# Milvus
MILVUS_HOST=localhost
MILVUS_PORT=19530
MILVUS_COLLECTION=zoo_faq_collection

# RAG 日志
RAG_LOG_LEVEL=full

# HuggingFace 镜像（settings.py 已默认设置 hf-mirror.com）
# 主站如能直连可改回 https://huggingface.co
# HF_ENDPOINT=https://huggingface.co
```

---

## 七、FAQ 数据结构

`data/faq_meetings.json` 每条 FAQ：

| 字段 | 说明 |
|------|------|
| `id` | 唯一标识（1-150） |
| `question` | 标准问法 |
| `answer` | 回答内容 |
| `tags` | 语义标签（英文 snake_case + 中文混合，第一个 snake_case 标签即"意图标签"） |

`data/faq_enrichment.json`：

```json
{
  "variations": {
    "1": ["新建", "发起", "召开会议"],
    ...
  },
  "answer_keywords": {
    "1": "新建会议 即时会议 预约会议 会议链接",
    ...
  }
}
```

---

## 八、与 Phase 1 的差异

| 维度 | Phase 1 | Phase 2 |
|------|---------|---------|
| 检索方式 | 字符串 tag 匹配（全量遍历） | 向量相似度检索（Milvus） |
| Prompt 注入量 | 全部 25 条 FAQ | 仅 Top-3 相关 FAQ |
| FAQ 数量 | 25 条 | 150 条 |
| 向量数据库 | 无 | Milvus（本地 Docker） |
| Embedding 模型 | 无 | BAAI/bge-m3（1024 维，本地推理） |
| LLM 调用方式 | 直接调用 | LangChain Chain 封装 |
| 依赖 | DeepSeek API | DeepSeek API + Milvus + HuggingFace |

---

## 九、Phase 2 已完成事项

- 150 条 FAQ 全量入库（覆盖会议创建/加入/共享/音视频/网络/账户/录制/计费/集成等场景）
- 检索增强：变体 + tags + 答案关键词三段式拼接
- RAG Chain 完整 LCEL 实现 + 结构化日志
- HF mirror 默认配置，免去手动设置 `HF_ENDPOINT`

## 十、Phase 2 → Phase 3 迁移注意

Phase 3 的 Pipeline/Step 架构接管了原本 `faq_chain.py` 中的检索 + 生成逻辑。
旧的 `invoke_rag_chain` 仍保留作向后兼容入口（评估脚本/单元测试）。
新增内容详见 `wiki/PHASE3_DESIGN.md`。
