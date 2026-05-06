# Zoo AI Chat

Zoo 多产品线 AI 智能客服系统的学习型项目，目标是掌握 AI Agent、RAG、LangChain、意图识别、召回策略等核心能力。

---

## 项目阶段

本项目遵循 5 阶段学习路线。当前处于 **Phase 1**。

| 阶段 | 重点 | 状态 |
|------|------|------|
| Phase 0 | Python 环境准备 | - |
| **Phase 1** | **最小 FAQ 机器人**（DeepSeek + LangChain + FastAPI） | **进行中** |
| Phase 2 | RAG + 向量检索（FAISS/Chroma） | 计划中 |
| Phase 3 | 意图识别 + 对话管理 | 计划中 |
| Phase 4 | Agent 化升级 + 多产品线扩展 | 计划中 |
| Phase 5 | 召回策略优化与调优 | 计划中 |

---

## Phase 1：最小 FAQ 机器人

基于 **Zoo 会议服务**产品线的 FAQ 问答机器人，核心技术栈：

- **DeepSeek**（LLM 底座，OpenAI 兼容接口）
- **LangChain**（LCEL 流水线：PromptTemplate -> LLM -> OutputParser）
- **FastAPI**（REST API + 静态前端）

### 快速启动

```bash
# 1. 创建虚拟环境
python3 -m venv .venv
source .venv/bin/activate

# 2. 安装依赖
pip install -r requirements.txt

# 3. 启动服务
python -m app.main
# 或：uvicorn app.main:app --reload --port 8000

# 4. 打开浏览器
open http://localhost:8000/static/index.html
```

### 项目结构（Phase 1）

```
zoo-chat/
├── requirements.txt              # Python 依赖
├── .env                         # API Key（禁止提交到 git）
├── README.md
├── NOTES.md                     # 学习笔记
├── zoo_ai_project_plan.md       # 项目计划（由 HTML 转换）
├── config/
│   └── settings.py             # 配置管理
├── data/
│   ├── __init__.py            # 加载 faq_meetings.json
│   └── faq_meetings.json      # 会议服务 FAQ 数据（25 条）
├── app/
│   ├── main.py                # FastAPI 入口
│   ├── llm/
│   │   └── deepseek_client.py  # DeepSeek LLM 封装
│   ├── chains/
│   │   └── faq_chain.py       # LangChain LCEL Chain（核心逻辑）
│   └── api/
│       └── chat.py            # POST /api/chat 路由
├── frontend/
│   └── index.html             # 极简对话 UI
└── tests/
    └── test_faq_chain.py     # 单元测试（13 个，全部通过）
```

### API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/` | 服务信息 |
| GET | `/health` | 健康检查 |
| POST | `/api/chat` | 发送消息，返回 AI 回答 |
| GET | `/static/index.html` | 前端对话页面 |

### 请求示例

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "如何共享屏幕"}'
```

### 核心概念（Phase 1）

| 概念 | 说明 |
|------|------|
| **Chain** | 流水线：用 `\|` 管道符将每个步骤串联起来 |
| **PromptTemplate** | 可复用的提示词骨架，`{}` 占位符由输入填充 |
| **LCEL** | LangChain Expression Language，现代 LangChain 的推荐写法 |
| DeepSeek | OpenAI 兼容接口，`base_url` 设为 `https://api.deepseek.com` |

---

## 项目设计背景

### 目标

构建一个统一的 AI 客服入口：

- 理解用户意图（intent），而不只是识别 SKU
- 自动路由到正确的产品上下文（会议 / 设备 / 电话）
- 基于可信知识（文档、FAQ）回答，必要时调用实时系统（订单状态等）
- 置信度低或涉及敏感问题时，平滑转人工

### 为什么"通用"很难

不同产品的症状（如"会议无声音"vs"耳机无法配对"）、数据（账号 vs 序列号）和解决路径（App 设置 vs 驱动 vs 换货）都不同。一套 Prompt 不够用，需要**共享编排 + 分域知识与工具**。

### 推荐方案（高层）

- **A. 共享对话层**：意图识别、槽位填充、产品路由、品牌语调策略
- **B. 分域知识**：每条产品线独立索引，RAG 检索，版本感知答案
- **C. 工具集成**：订单查询、设备注册、工单创建、转人工
- **D. 质量保障**：引用溯源、PII 处理、回归测试集

### 成功指标

- 有质量的自主解决率（非简单拒答）
- 首次解决率和平均处理时间
- 错误产品率（答非所问的比例）接近零

---

## 参考文档

- 项目计划：[`zoo_ai_project_plan.md`](./zoo_ai_project_plan.md)
- 学习笔记：[`NOTES.md`](./NOTES.md)
- 项目规则：[`.cursor/rules/project_rules.md`](.cursor/rules/project_rules.md)
