# Zoo AI Chat

Zoo 多产品线 AI 智能客服系统的学习型项目，目标是掌握 AI Agent、RAG、LangChain、意图识别、召回策略、SSE 流式输出等核心能力。

---

## 项目阶段

| 阶段 | 重点 | 状态 |
|------|------|------|
| Phase 0 | Python 环境准备 | ✅ 完成 |
| Phase 1 | 最小 FAQ 机器人（DeepSeek + LangChain + FastAPI） | ✅ 完成 |
| Phase 2 | RAG + 向量检索（Milvus） | ✅ 完成 |
| Phase 3 | 意图识别 + 槽位填充 + Pipeline 架构 | ✅ 完成 |
| Phase 4 | Agent 化升级 + 多产品线扩展 | ✅ 完成 |
| Phase 5 | 召回策略优化（BM25 + Hybrid + Reranker + A/B Test） | ✅ 完成 |
| **Phase 6** | **SSE 流式输出 + React TypeScript 前端** | **✅ 完成** |
| Phase 7 | LangGraph 多 Agent 编排 | 计划中 |
| Phase 8 | 生产化部署（Docker Compose + CI/CD） | 计划中 |

---

## Phase 6：SSE 流式输出 + React 前端

当前版本核心特性：

- **SSE 流式回答**：FastAPI `StreamingResponse` + LangChain `astream()`，首 token 延迟低
- **React TypeScript 前端**：Vite 构建，完整类型安全
- **Markdown 渲染**：`react-markdown` + `rehype-highlight`，支持代码块语法高亮
- **流式游标**：回答生成中显示 `▌` 闪烁游标，完成后消失
- **产品线徽章**：自动识别并显示当前对话所属产品线（7 种颜色编码）
- **快捷提问**：5 个常见问题一键填充
- **延迟展示**：状态栏显示每次回答的端到端耗时（ms）

---

## 快速启动

### 前置依赖

- Python 3.10+
- Node.js 18+
- Docker（运行 Milvus）
- DeepSeek API Key（写入 `.env`）

### 1. 启动 Milvus

```bash
cd infra && docker compose up -d
```

### 2. 构建向量索引（首次或重建）

```bash
python scripts/build_milvus_index.py
```

### 3. 启动后端

```bash
source .venv/bin/activate
python -m app.main
# 或：uvicorn app.main:app --reload --port 8000
```

### 4. 构建并访问前端

```bash
# 方式 A：直接使用已构建产物（推荐）
# 后端已自动 serve frontend/dist/，直接打开：
open http://localhost:8000/static/index.html

# 方式 B：前端开发模式（热更新）
cd frontend && npm install && npm run dev
# 打开 http://localhost:5173
```

---

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/` | 服务信息 |
| GET | `/health` | 健康检查 |
| POST | `/api/chat` | 同步问答（Phase 1 兼容） |
| POST | `/api/chat/stream` | **SSE 流式问答（Phase 6）** |
| GET | `/static/index.html` | React 前端 |

### SSE 流式接口示例

```bash
curl -X POST http://localhost:8000/api/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "如何共享屏幕", "session_id": null}' \
  --no-buffer
```

SSE 事件格式：

```
data: {"type": "chunk", "content": "您好，共享屏幕"}
data: {"type": "chunk", "content": "的步骤如下..."}
data: {"type": "done", "session_id": "xxx", "product_id": "meetings", "latency_ms": 1234.5}
```

---

## 项目结构

```
zoo-chat/
├── app/
│   ├── main.py                    # FastAPI 入口（自动 serve frontend/dist）
│   ├── api/
│   │   ├── chat.py               # /api/chat + /api/chat/stream
│   │   └── logs_route.py         # RAG 日志查询
│   ├── services/
│   │   └── chat_service.py       # ChatService.ask() / ChatService.stream()
│   ├── pipeline/                  # Phase 3 Pipeline 架构
│   │   ├── steps/                # 各处理步骤（意图、槽位、检索、生成…）
│   │   └── chat_pipeline.py
│   ├── agents/                    # Phase 4 ZooAgent（ReAct）
│   ├── retrieval/                 # Phase 5 BM25 + Hybrid + Reranker
│   └── middleware/
├── frontend/
│   ├── src/
│   │   ├── App.tsx               # 主组件（SSE fetch 循环）
│   │   ├── components/
│   │   │   ├── ChatMessage.tsx   # Markdown 渲染 + 流式游标
│   │   │   ├── ChatInput.tsx     # 输入框 + 快捷提问
│   │   │   └── ProductBadge.tsx  # 产品线徽章
│   │   ├── types.ts              # Message / ChatMeta / SSEEvent
│   │   └── index.css             # 深色主题 CSS
│   ├── dist/                      # Vite 构建产物（已提交）
│   ├── package.json
│   └── vite.config.ts            # 开发代理 /api → :8000
├── data/                          # FAQ JSON（5 条产品线，220 条）
├── scripts/
│   └── build_milvus_index.py     # 向量索引构建
├── wiki/                          # 各阶段设计文档
├── infra/
│   └── docker-compose.yml        # Milvus + etcd + MinIO
├── config/settings.py
├── requirements.txt
└── zoo_ai_project_plan.md        # 完整项目计划
```

---

## 技术栈

| 层 | 技术 |
|----|------|
| LLM | DeepSeek（OpenAI 兼容接口） |
| 编排 | LangChain LCEL + Pipeline 架构 |
| 向量库 | Milvus（5 条产品线独立 Collection） |
| 关键词检索 | BM25（rank-bm25） |
| 混合召回 | RRF 融合 + Cross-Encoder Reranker |
| API | FastAPI + SSE StreamingResponse |
| 前端 | React 18 + TypeScript + Vite |
| Markdown | react-markdown + rehype-highlight |
| 容器 | Docker Compose（Milvus 基础设施） |

---

## Wiki 文档

| 文档 | 内容 |
|------|------|
| [PHASE1_DESIGN.md](wiki/PHASE1_DESIGN.md) | 最小 FAQ 机器人设计 |
| [PHASE2_DESIGN.md](wiki/PHASE2_DESIGN.md) | RAG + Milvus 向量检索 |
| [PHASE3_DESIGN.md](wiki/PHASE3_DESIGN.md) | 意图识别 + Pipeline 架构 |
| [PHASE4_DESIGN.md](wiki/PHASE4_DESIGN.md) | ZooAgent + 多产品线工具 |
| [PHASE5_DESIGN.md](wiki/PHASE5_DESIGN.md) | BM25 + Hybrid + Reranker + A/B |
| [PHASE6_DESIGN.md](wiki/PHASE6_DESIGN.md) | SSE 流式输出 + React 前端 |

---

## 参考

- 项目计划：[`zoo_ai_project_plan.md`](./zoo_ai_project_plan.md)
- 项目规则：[`.cursor/rules/project_rules.md`](.cursor/rules/project_rules.md)
