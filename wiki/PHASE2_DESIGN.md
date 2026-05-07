# Zoo Chat - Phase 2 RAG 升级设计文档

> 日期：2026-05-07
> Phase 1 → Phase 2 完整升级记录

---

## 一、升级目标

Phase 1 采用**全量 FAQ 注入**方式：将全部 25 条 FAQ 直接拼入 Prompt，LLM 从中检索回答。
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

### Phase 2 架构

```
用户输入 → Embedding（all-MiniLM-L6-v2）
         → Milvus 向量相似度检索 Top-3
         → 组装 context + history → Prompt → DeepSeek LLM → 回答
```

新增组件：
- **Milvus** 向量数据库（存储 25 条 FAQ 的向量）
- **HuggingFace Embedding** 本地模型（免费，无需 API key）
- **LangChain RAG Chain** 替代原有的全量注入逻辑

---

## 三、新增依赖

### Python 包（requirements.txt）

| 包名 | 版本 | 用途 |
|------|------|------|
| `langchain` | >= 0.3.0 | LangChain 核心框架 |
| `langchain-core` | >= 0.3.0 | LangChain 核心抽象 |
| `langchain-community` | >= 0.3.0 | ChatOpenAI（DeepSeek 兼容） |
| `langchain-huggingface` | 最新 | HuggingFace Embedding 封装 |
| `pymilvus` | >= 2.3.0 | Milvus Python SDK |
| `langchain-milvus` | >= 0.1.0 | LangChain × Milvus 集成（未直接使用，仅作兼容备用） |
| `fastapi` | >= 0.110.0 | Web 框架（已有） |
| `uvicorn[standard]` | >= 0.29.0 | ASGI 服务器（已有） |
| `httpx` | >= 0.27.0 | HTTP 客户端（已有） |

> 注：`langchain-openai` 已废弃，DeepSeek 通过 `langchain-community` 中的 `ChatOpenAI` 接入（OpenAI 兼容接口）。

### 向量数据库（Docker）

Phase 2 新增三个 Docker 容器（Milvus All-in-One）：

| 容器名 | 镜像 | 端口 | 用途 |
|--------|------|------|------|
| `milvus-etcd` | quay.io/coreos/etcd:v3.5.5 | 2379 | 元数据存储 |
| `milvus-minio` | minio/minio:RELEASE.2023-03-20T20-16-18Z | 9091 / 9000 | 对象存储（S3 兼容） |
| `milvus-standalone` | milvusdb/milvus:v2.3.3 | 9091 / 19530 | Milvus 主服务 |

启动命令：
```bash
docker compose -f scripts/docker-compose-milvus.yml up -d
```

---

## 四、新增文件

### 核心代码

| 文件 | 说明 |
|------|------|
| `app/llm/embedding_client.py` | HuggingFace Embedding 客户端封装，返回 `HuggingFaceEmbeddings` 实例 |
| `app/chains/faq_chain.py` | Phase 2 RAG Chain 实现：`build_rag_chain()` + `invoke_rag_chain()` |
| `app/api/chat.py` | `POST /api/chat` 接口，接收 `message` + `history`，返回 `answer` + `session_id` + `latency_ms` |
| `app/main.py` | FastAPI 应用入口，注册路由和 CORS |

### 脚本与配置

| 文件 | 说明 |
|------|------|
| `scripts/build_milvus_index.py` | 索引构建脚本：将 `data/faq_meetings.json` 中的 25 条 FAQ 转换为向量存入 Milvus |
| `scripts/docker-compose-milvus.yml` | Milvus 容器编排配置 |
| `config/settings.py` | 新增 `embedding_model`、`embedding_dimension`、`milvus_host/port/collection` 配置项 |
| `.env` | 新增 Embedding 和 Milvus 相关环境变量 |

---

## 五、数据流详解

### 索引构建流程（一次性）

```
data/faq_meetings.json（25 条 FAQ）
  → build_milvus_index.py
    → HuggingFaceEmbeddings（all-MiniLM-L6-v2, 384 维）
    → 每条 FAQ 构建 enriched 检索文本：
      "问法变体 tags 回答关键词 原始问答"
    → pymilvus（绕过 langchain_milvus ORM）
    → Milvus collection: zoo_faq_collection（IVF_FLAT 索引）
```

enriched 检索文本示例（FAQ ID=18）：
```
断网 网络断了 网络中断 网络 连不上 掉线
troubleshoot_connectivity network 断网
网络中断 断网 重连 检查网络 会议链接 重新加入
会议中断网了怎么办 如果网络中断：1) 检查本地网络连接；2) Zoo 会自动尝试重连；3) 可点击会议链接重新加入；4) 如果重新加入失败，系统会自动保存聊天记录。
```

### 推理流程（每次请求）

```
1. POST /api/chat { "message": "网络断了怎么办" }
2. _retrieve_docs(query):
     - embed_query("网络断了怎么办") → 384 维向量
     - Milvus search(collection, top_k=3) → 返回 Top-3 相似文档
     - 封装为 langchain Document 返回
3. _format_context(docs):
     - 将 Top-3 文档格式化为：
       【问题 18】断网 网络...问题：会议中断网了怎么办...
4. build_rag_chain() 组装 LangChain chain：
     - context = _format_context(_retrieve_docs(question))
     - history = format_chat_history(request.history)
     - Prompt: SYSTEM_PROMPT_V2（角色设定 + context + history）
5. chain.invoke(question) → DeepSeek LLM 生成回答
6. 返回 { "answer": "...", "session_id": "...", "latency_ms": ... }
```

---

## 六、环境变量配置

`.env` 文件 Phase 2 新增项：

```bash
# Embedding（HuggingFace 本地模型，免费离线，无需 API key）
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DIMENSION=384

# Milvus 向量数据库
MILVUS_HOST=localhost
MILVUS_PORT=19530
MILVUS_COLLECTION=zoo_faq_collection
```

DeepSeek API 配置（Phase 1 已有，Phase 2 继续使用）：
```bash
DEEPSEEK_API_KEY=sk-...
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
```

---

## 七、FAQ 数据

共 150 条 Zoo 会议服务 FAQ，存储在 `data/faq_meetings.json`，每条包含：

| 字段 | 说明 |
|------|------|
| `id` | 唯一标识（1-25） |
| `question` | 标准问法 |
| `answer` | 回答内容 |
| `tags` | 语义标签（英文 + 中文混合） |

---

## 八、与 Phase 1 的差异

| 维度 | Phase 1 | Phase 2 |
|------|---------|---------|
| 检索方式 | 字符串 tag 匹配（全量遍历） | 向量相似度检索（Milvus） |
| Prompt 注入量 | 全部 25 条 FAQ | 仅 Top-3 相关 FAQ |
| 向量数据库 | 无 | Milvus（本地 Docker） |
| Embedding 模型 | 无 | HuggingFace all-MiniLM-L6-v2（本地） |
| LLM 调用方式 | 直接调用 | LangChain Chain 封装 |
| 依赖 | DeepSeek API | DeepSeek API + Milvus + HuggingFace |

---

## 九、后续优化方向（TODO）

1. **中文 Embedding 模型**：当前使用的 `all-MiniLM-L6-v2` 对中文支持有限，可考虑切换为 `shibing624/text2vec-base-chinese` 等中文模型
2. **关键词路由混合检索**：在向量检索前加一层精确关键词匹配，提升"断网"、"麦克风"等明确意图的召回准确率
3. **FAQ 扩充**：当前仅 25 条，覆盖场景有限，可按需扩充至 50-100 条
4. **Streaming 响应**：接入 LLM streaming 输出，提升用户体验
5. **Session Memory 持久化**：当前会话历史仅在请求内传递，可接入 Redis 持久化
