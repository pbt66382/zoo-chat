# Zoo 多产品线 AI 智能客服系统

> 学习型项目计划文档 · 版本 2.0 · 2026-05-13

---

## 项目元信息

| 属性 | 内容 |
|------|------|
| 目标 | 系统掌握 AI Agent、RAG、LangChain、意图识别、召回策略、生产化部署等核心能力 |
| 前置要求 | Java 基础，了解基本编程概念 |
| 开发语言 | Python（AI/ML 主流生态）+ 少量 Java 对接（可选） |
| 学习周期 | 8 个阶段，约 26 周，循序渐进 |
| 核心原则 | 做完每一步都能讲清楚原理，不做"黑箱"交付 |

---

## 目录

1. [第 0 阶段：Python 基础与环境准备](#第-0-阶段python-基础与环境准备)
2. [第 1 阶段：构建最小可运行的 FAQ 机器人](#第-1-阶段构建最小可运行的-faq-机器人)
3. [第 2 阶段：引入 RAG 与向量检索](#第-2-阶段引入-rag-与向量检索)
4. [第 3 阶段：意图识别与对话管理](#第-3-阶段意图识别与对话管理)
5. [第 4 阶段：Agent 化升级 + 多产品线扩展](#第-4-阶段agent-化升级--多产品线扩展)
6. [第 5 阶段：召回策略优化与调优](#第-5-阶段召回策略优化与调优)
7. [第 6 阶段：流式输出 + 前端体验升级](#第-6-阶段流式输出--前端体验升级)
8. [第 7 阶段：LangGraph 多 Agent 协作](#第-7-阶段langgraph-多-agent-协作)
9. [第 8 阶段：生产化部署与可观测性](#第-8-阶段生产化部署与可观测性)
10. [学习路径总结](#学习路径总结)
11. [技术栈全景图](#技术栈全景图)

---

## 第 0 阶段：Python 基础与环境准备

**预计时间：第 1~2 周**

作为有 Java 基础的开发者，需要先熟悉 Python 语法及 AI 开发工具链。这部分不需要深入，快速上手即可。

### 学习目标

- Python 基本语法（数据类型、函数、类、异常处理）
- pip / conda 环境管理
- requests、json、pathlib 等标准库使用
- Jupyter Notebook / VSCode Python 插件
- 了解什么是 API 调用（类比 Java 的 HTTP Client）

### 任务清单

1. 安装 Python 3.10+，使用 conda 或 venv 创建独立环境
2. 完成 Python 基础语法练习
3. 写一个 Java vs Python 对照小脚本，理解两者语法的对应关系
4. 学会用 pip 安装包：`pip install langchain langchain-community faiss-cpu`

### Java 背景快速对照

| Java | Python |
|------|--------|
| `class` | `class`（语法略有不同） |
| `List<String>` | `list` |
| `Map<String, Object>` | `dict` |
| `Optional` | `None` 检查 |
| `@RestController` | Flask/FastAPI `@app.route` |

### 阶段产出

本地可运行的 Python 环境，能独立写一个读取文件、处理数据的小脚本。

---

## 第 1 阶段：构建最小可运行的 FAQ 机器人

**预计时间：第 3~5 周**

这是整个项目的 MVP（最小可行产品）。你将学会：如何用 LangChain 串联 LLM 与知识库、如何构建一个完整对话流程。

### 核心技术点

- LangChain
- LLM API 调用
- Prompt 工程
- FastAPI / Flask
- 简单文本匹配

### Week 3：调用 LLM，实现最简对话

1. 注册 DeepSeek 账号，获取 API Key
2. 用 Python 调用 LLM，完成最简单的"问答"：用户问，LLM 答
3. 学习 **Prompt 工程**：什么是 system prompt、user prompt、few-shot example
4. 写一个带上下文的对话函数，模拟多轮对话（Java 里的 session 概念）

### Week 4：接入 LangChain，构建 Chain

1. 学习 LangChain 核心概念：
   - **Model I/O**：PromptTemplate、LLM、OutputParser
   - **Chain**：把多个步骤串联成一个 pipeline
   - **Memory**：保存对话历史
2. 用 LangChain 构建一个 FAQ 问答 Chain
3. 为 Zoo 的**会议服务（Meetings）**产品线，手写 20~30 条 FAQ
4. 让 LLM 只基于这些 FAQ 回答，超出范围的礼貌拒绝

### Week 5：包装成 Web 服务

1. 学习 FastAPI 基础：路由定义、请求体、响应体
2. 用 FastAPI 包装你的 FAQ 机器人
3. 前端做一个极简 HTML 页面，调用后端 API，实现实时对话

### 核心概念（必须理解透）

| 概念 | 含义 |
|------|------|
| **Chain** | 工序流水线。每个工序（Prompt → LLM → Parser）是一个环节，多个环节串联成完整流程 |
| **PromptTemplate** | 带占位符的提示词骨架，让同一套模板复用不同输入 |
| **Memory** | 对话历史存储，类似 Java 里维护一个 `List<Message>` 存会话上下文 |
| **LCEL** | LangChain Expression Language，LangChain 新版本推荐的新写法，用 `|` 管道符串联步骤 |

### 阶段产出

一个能跑起来的 Web 服务，输入"如何共享屏幕"，AI 返回基于 FAQ 的准确回答。前端页面可以聊天。

---

## 第 2 阶段：引入 RAG 与向量检索

**预计时间：第 6~9 周**

这是 AI 客服系统的**核心能力**。你将学会：如何把文档知识存入向量数据库，查询时如何精准召回相关内容（Recall）。

### 核心技术点

- Embedding
- 向量数据库
- FAISS / Chroma
- LangChain RAG Chain
- 文档切分（Chunking）

### Week 6：理解 Embedding 与向量检索原理

1. 理解什么是**文本 Embedding**：把文字变成一串数字向量，语义相近的文字向量也相近
2. 用 DeepSeek/BGE 对 Zoo 会议服务的产品手册段落生成向量
3. 用**余弦相似度**手工计算两条文本的相似度，理解向量距离的物理意义
4. 理解：为什么不能用关键词匹配（BM25）做语义搜索？Embedding 解决了什么问题？

### Week 7：搭建向量数据库

1. 选型向量数据库（初学推荐）：
   - **ChromaDB**（最简单，Python 原生，适合本地学习）
   - **FAISS**（Facebook 开源，支持海量向量，推荐生产级入门）
2. 准备 Zoo 知识文档（会议服务产品手册、历史工单、技术故障排查文档）
3. 实现**文档切分（Chunking）**策略：
   - 按固定长度切（如 500 字一段）
   - 按段落切（更语义完整，推荐）
   - 重叠切分（相邻 chunk 重叠 50~100 字，减少上下文断裂）
4. 生成每个 chunk 的向量，存入 FAISS/Chroma

### Week 8：实现 RAG 检索链

1. 构建 LangChain RAG Chain 四步走：
   - ① **问题 Embedding**：用户问题 → 向量
   - ② **向量检索**：在向量库中搜索 Top K 相关 chunk
   - ③ **上下文组装**：将用户问题 + 召回的 chunk 组装成 Prompt
   - ④ **LLM 生成**：让 LLM 基于检索到的内容回答
2. 用 LangChain 的 `RetrievalQA` Chain 或 LCEL 写法实现上述流程
3. 添加**引用来源**功能：让 LLM 的回答中标注"参考了哪篇文档的第几段"
4. 对比实验：直接让 LLM 回答 vs 加了 RAG 之后回答，测试效果差异

### Week 9：扩展到更多产品线

1. 为 Zoo 的**通话服务（Phone）**准备第二批文档，重复 Week 7~8 的流程
2. 学习**多产品线知识库隔离**：不同产品的知识存不同 collection，避免跨产品干扰召回
3. 用 Zoo 的产品场景做端到端测试

### 召回策略核心概念

| 概念 | 含义 |
|------|------|
| **召回（Recall）** | 从知识库中把所有可能相关的候选都找出来。recall 低 = 好的答案根本没被召回，后面的 LLM 再强也没用 |
| **精准度（Precision）** | 召回的结果中，有多少是真正相关的。precision 低 = 召回了大量无关内容，LLM 被噪声干扰 |
| **BM25** | 基于关键词的传统召回算法，RAG 中作为辅助补充 |
| **Hybrid Search** | 向量检索 + BM25 混合使用，兼顾语义和关键词 |

### 阶段产出

基于真实文档的 RAG 机器人。能自己准备文档、切分、存入向量库，并能说出每个环节的原理。

---

## 第 3 阶段：意图识别与对话管理

**预计时间：第 10~13 周**

你将学会：如何让 AI 理解用户到底想问什么（意图识别），如何在多轮对话中记住上下文（对话管理），以及如何设计一个能真正解决问题的对话流程。

### 核心技术点

- 意图识别
- 多分类模型 / LLM Classification
- 槽位填充（Slot Filling）
- 对话状态管理
- 多轮追问

### Week 10：意图识别

1. 理解意图识别（Intent Detection）的业务价值：不同意图 → 不同处理流程
2. 设计 Zoo 会议服务的意图分类体系：

| 意图 ID | 意图名称 | 示例问法 |
|---------|---------|---------|
| greet | 问候/寒暄 | "你好" |
| meeting_create | 创建会议 | "怎么发起一个会议" |
| screen_share | 共享屏幕 | "如何共享我的屏幕" |
| troubleshoot_audio | 音频故障 | "对方听不到我的声音" |
| schedule_meeting | 预约会议 | "我想预约明天下午3点的会" |
| general_inquiry | 通用咨询 | "Zoom是什么" |
| out_of_scope | 范围外 | "今天天气怎么样" |

3. 用**LLM 做意图分类**（最简方案）
4. 如果想进阶：了解 BERT/RoBERTa 微调做意图分类的流程

### Week 11：多轮对话与槽位填充

1. 理解"槽位填充（Slot Filling）"：收集解决用户问题所需的必要信息
2. 用 LangChain 的 `ConversationBufferMemory` 或 `ConversationSummaryMemory` 存储对话历史
3. 设计对话流程图
4. 用 LangChain 的 **Agent** 机制处理条件分支

### Week 12：产品识别

1. 在用户第一句话就判断产品类别：会议服务 / 可视电话 / 耳机 / 鼠标 / 会议大屏 / 通话服务 / 通用
2. 用 LLM 分类 + 关键词规则双重校验
3. 产品确定后，加载对应知识库 collection，减少无关召回

### Week 13：集成测试 + 对话质量评估

1. 端到端测试：完整走一遍产品识别 → 意图识别 → 槽位填充 → RAG 检索 → 回答生成
2. 建立评估集（50~100 条真实用户问法）
3. 评估指标：意图准确率 > 90%、产品识别率 > 95%、回答准确率 > 80%、对话完成率 > 70%

### 阶段产出

一个完整的多轮对话客服机器人：能自动判断产品线 → 判断意图 → 收集必要信息 → RAG 检索 → 生成回答。

---

## 第 4 阶段：Agent 化升级 + 多产品线扩展

**预计时间：第 14~16 周**

你将理解什么是**AI Agent**，如何让 AI 主动调用工具、规划行动，并将其落地到 Zoo 的 6 大产品线。

### 核心技术点

- LangChain Agent
- ReAct 模式
- Tool Calling / Function Calling
- 多产品线架构

### Week 14：深入理解 AI Agent

1. 学习 AI Agent 的核心概念：
   - **Agent** = LLM + 工具 + 规划能力。能自主决定"我要做什么"
   - **Tool** = Agent 可以调用的外部能力（查数据库、调用 API、计算器、搜索）
   - **ReAct**（Reason + Act）= LLM 思考"我现在有什么信息，接下来该做什么"，然后执行 action，再观察结果
   - **Tool Calling / Function Calling** = LLM 输出结构化的工具调用指令，程序解析并执行
2. 用 LangChain Agent 重构第 3 阶段的代码
3. 为 Agent 添加"自我检查"能力：生成回答后，让 Agent 判断置信度，低则转人工

### Week 15：扩展到全部 6 条产品线

| 产品线 | 典型意图 | 知识文档来源 |
|--------|---------|------------|
| 可视电话 | 设备激活、固件升级、音视频配置 | 产品手册 PDF |
| 耳机 | 连接配对、音质问题、认证查询 | FAQ + 工单 |
| 鼠标 | 大屏控制、批注操作、配对问题 | 快速入门指南 |
| 会议大屏 | 投屏、白板使用、无法入会 | 运维文档 |

### Week 16：添加工具调用能力（Function Calling）

1. 设计并实现几个实用 Tool：
   - **QueryProductDoc**：查 Zoo 产品文档（RAG）
   - **QueryFAQ**：查 FAQ 数据库（精确匹配）
   - **CheckOrderStatus**：查订单状态（模拟 API 调用）
   - **CreateTicket**：创建人工工单
   - **EscalateToHuman**：判断是否需要转人工
2. 用 OpenAI Function Calling 或 LangChain Agent 的 tool_call 机制实现工具选择

### 阶段产出

一个 Agent 化的 Zoo 全产品线客服系统。能自动选择工具、自主决策下一步。

---

## 第 5 阶段：召回策略优化与调优

**预计时间：第 17~18 周**

这是提升系统质量的关键阶段。你将学会：如何量化评估召回质量、如何优化 chunk 策略、如何做 A/B 测试，以及如何持续迭代。

### 核心技术点

- 召回率评估
- Chunk 策略调优
- Hybrid Search
- 重排序（ReRank）
- A/B 测试
- Prompt 调优

### Week 17：召回质量分析与优化

1. 建立召回评估集（100 条测试问题，每条标注"期望召回的文档"）
2. 分析召回失败的 case
3. 实现并对比召回策略：

| 策略 | 做法 | 适用场景 |
|------|------|---------|
| 纯向量检索 | 只靠 Embedding 相似度 | 语义相近、表述多样的问题 |
| BM25 | 关键词精确匹配 | 含具体型号、错误码的问题 |
| Hybrid Search | 向量 + BM25 加权混合 | 通用场景，推荐默认方案 |
| Rerank | 先召回 Top 20，再用 Cross-Encoder 重排 | 召回量大、精确度要求高 |

4. 中文场景推荐 embedding 模型：`BGE-large-zh`、`m3e-large`、`text2vec-base-chinese`

### Week 18：Prompt 调优 + A/B 测试 + 上线总结

1. Prompt 调优（迭代式）
2. A/B 测试设计（轻量版）
3. 系统总结：整理项目架构图、写出每个阶段的技术选型理由、记录踩坑经历和解决方案
4. 可选：部署到云端（阿里云函数计算、Railway、Render 等免费额度）

### 阶段产出

一份完整的召回策略对比实验报告 + 调优记录 + 项目复盘文档。

---

## 第 6 阶段：流式输出 + 前端体验升级

**预计时间：第 19~20 周**

用户等待 AI 回答时看到的是一整段文字突然出现，体验不够好。真实产品都是**逐字流式输出**（打字机效果）。这一阶段你将学会：SSE 流式协议、前端实时渲染、以及如何让 React 组件消费流数据。

### 核心技术点

- Server-Sent Events（SSE）
- LangChain 流式生成（`astream`）
- React + TypeScript（前端重构）
- Markdown 实时渲染
- 打字机动画效果

### Week 19：后端流式接口

1. 理解 **SSE（Server-Sent Events）** 原理：
   - 服务端持续向客户端推送文本块（chunk）
   - 类比 Java 的 `ResponseBodyEmitter` 或 Servlet 3.0 异步响应
   - 对比 WebSocket：SSE 单向推送、更简单、HTTP 原生支持

2. 激活已预留的 `GenerationStep.stream()` 接口，接入 FastAPI `StreamingResponse`：

   ```python
   @router.post("/api/chat/stream")
   async def chat_stream(request: ChatRequest):
       async def event_generator():
           async for chunk in service.stream(question=request.message):
               yield f"data: {json.dumps({'chunk': chunk})}\n\n"
       return StreamingResponse(event_generator(), media_type="text/event-stream")
   ```

3. 在 `GenerationStep` 中用 LangChain `astream` 替换 `invoke`，实现逐 token 输出

4. 处理流式场景下的 Pipeline 特殊性：产品检测 / 意图识别 / 槽位收集仍走非流式，只有**最终生成阶段**流式输出

### Week 20：React 前端重构

1. 用 **Vite + React + TypeScript** 重构现有极简 HTML 前端

2. 核心功能实现：
   - `EventSource` / `fetch` + `ReadableStream` 消费 SSE
   - 打字机效果：`useState` 累积 chunk，`useEffect` 滚动到底部
   - **Markdown 渲染**：用 `react-markdown` + `rehype-highlight` 渲染代码块和列表
   - 消息气泡：用户消息右对齐，AI 回答左对齐，流式时末尾加光标 `▌`

3. 产品线标签：在对话框顶部显示当前检测到的产品线（会议服务 / 耳机 / …）

4. 前端状态设计：

   ```
   ChatState:
     messages: Message[]       // 历史消息列表
     isStreaming: boolean       // 是否正在流式输出
     currentChunk: string       // 当前正在累积的 AI 回答
     sessionId: string | null   // 会话 ID
     productLine: string | null // 当前产品线
   ```

### 核心概念

| 概念 | 含义 |
|------|------|
| **SSE** | HTTP 长连接，服务端持续推送 `data: ...\n\n` 格式文本，客户端用 `EventSource` 订阅 |
| **astream** | LangChain 异步流式生成器，每次 `yield` 一个 token 块，无需等待全部生成完毕 |
| **ReadableStream** | Fetch API 的流式读取接口，配合 `TextDecoder` 逐块解析 SSE 数据 |
| **Hydration** | React 渲染流式内容时的状态同步：每收到一个 chunk 就更新 state，触发局部重渲染 |

### 阶段产出

流式输出的聊天界面：AI 回答逐字出现，支持 Markdown 渲染，显示当前产品线标签，交互体验接近生产级产品。

---

## 第 7 阶段：LangGraph 多 Agent 协作

**预计时间：第 21~23 周**

当前 Agent 是单一 LLM 自主规划，复杂问题（需要多步推理 + 人工确认 + 状态回溯）难以处理。**LangGraph** 是 LangChain 官方推出的有状态图编排框架，适合构建**多 Agent 协作**系统。你将学会：状态机思维、Supervisor 模式、Human-in-the-loop。

### 核心技术点

- LangGraph StateGraph
- Supervisor Agent 模式
- 有状态工作流（Stateful Workflow）
- Human-in-the-loop 暂停 / 恢复
- 子 Agent 专业化分工

### Week 21：理解 LangGraph 核心模型

1. 理解 LangGraph 与 LangChain Pipeline 的区别：

   | 维度 | Pipeline（Phase 3） | LangGraph（Phase 7） |
   |------|---------------------|---------------------|
   | 流程 | 线性，固定顺序 | 有向图，支持循环和条件分支 |
   | 状态 | `ChatContext` dataclass | `TypedDict` 全局状态图 |
   | 分支 | RouterStep 硬编码 | 边（Edge）上的条件函数 |
   | 中断 | 不支持 | `interrupt()` 暂停等人工审核 |

2. 用 LangGraph 实现一个最简工作流：`检索 → 判断置信度 → 高置信直接回答 / 低置信转人工`

3. 理解 **StateGraph** 三要素：
   - **Node**：执行具体逻辑的函数（类比 Pipeline 的 Step）
   - **Edge**：节点间的有向连接（可带条件）
   - **State**：在所有节点间共享的 TypedDict（类比 Java 的 ThreadLocal 上下文）

### Week 22：Supervisor + 专业子 Agent

1. 设计 Zoo 多 Agent 架构：

   ```
   用户输入
     └── SupervisorAgent（协调者）
           ├── 判断任务类型
           ├── 分发给专业子 Agent
           └── 合并结果，输出最终回答
   
   子 Agent 分工：
   ├── RetrievalAgent    → 负责知识库检索（向量 + BM25 + Reranker）
   ├── DiagnosisAgent    → 负责故障诊断（多轮追问 + 故障树推理）
   ├── OrderAgent        → 负责订单/物流查询
   └── EscalationAgent   → 负责判断是否转人工 + 创建工单
   ```

2. 用 LangGraph 的 `send()` 实现 Supervisor 动态路由到子 Agent

3. 子 Agent 之间可以**串行**（诊断 → 检索）也可以**并行**（同时查知识库和订单）

4. 学习 **Map-Reduce 模式**：将用户问题分解为子问题，并行让多个子 Agent 回答，最后合并

### Week 23：Human-in-the-loop + 长期记忆

1. **Human-in-the-loop**：在敏感操作（创建工单、退款）前暂停，等人工确认：

   ```python
   # LangGraph interrupt 机制
   @node
   def create_ticket_node(state):
       if state["ticket_amount"] > 1000:
           interrupt({"reason": "高额工单需人工审核", "draft": state["ticket_draft"]})
       # 人工审核后恢复，state 中会有审核结果
   ```

   类比 Java 的**工作流引擎**（Activiti/Flowable）的人工任务节点。

2. **长期记忆**：用 LangGraph 内置的 `MemorySaver` 或 `SqliteSaver` 持久化对话图状态，实现跨会话记忆：
   - 记住用户偏好（"这个用户惯用蓝牙耳机"）
   - 记住未解决的工单（"上次的问题解决了吗"）

3. 将现有 `ZooAgent` 逐步迁移到 LangGraph，保持 API 接口不变

### 核心概念

| 概念 | 含义 |
|------|------|
| **StateGraph** | 有向图编排，状态在节点间流转，支持条件边和循环 |
| **Supervisor Pattern** | 协调者 Agent 负责拆解任务 + 分发 + 汇总，子 Agent 各司其职 |
| **interrupt()** | 工作流暂停原语，等待外部输入（人工/系统）后继续，保留完整状态 |
| **Checkpointer** | 持久化图执行状态的快照，支持断点续跑和历史回溯 |

### 阶段产出

基于 LangGraph 的多 Agent 协作系统：Supervisor 分发任务、子 Agent 专业化执行、人工审核节点、跨会话长期记忆。能说清楚 LangGraph 和传统工作流引擎的异同。

---

## 第 8 阶段：生产化部署与可观测性

**预计时间：第 24~26 周**

前 7 个阶段都是在本地跑。这一阶段你将学会：如何把 AI 服务部署到云上、如何监控它、如何在不重启的情况下热更新知识库，以及如何建立数据飞轮持续改善系统质量。

### 核心技术点

- Redis Session 持久化
- Prometheus + Grafana 监控
- Docker 多容器编排
- GitHub Actions CI/CD
- 知识库热更新（Admin API）
- 用户反馈闭环（Data Flywheel）

### Week 24：Session 持久化 + Redis

1. 理解当前 Session 的缺陷：进程内存存储，服务重启丢失，无法多实例横向扩展

2. 用 Redis 替换 `app/memory/session.py` 中的内存字典：
   - `SessionState` 序列化为 JSON 存入 Redis，设置 TTL（1小时）
   - 类比 Java Spring Session + Redis 的方案
   - 关键接口 `get_or_create / save / delete` 不变，切换对上层透明

3. `docker-compose.yml` 新增 Redis 容器，与 Milvus 一起编排

4. 压力测试：用 `locust` 模拟 100 并发用户，验证多 worker 下 session 一致性

### Week 25：监控 + 可观测性

1. **结构化指标**（Prometheus）：

   | 指标 | 含义 |
   |------|------|
   | `zoo_request_total` | 请求总数（按产品线/意图分类） |
   | `zoo_request_latency_seconds` | 请求延迟 P50/P95/P99 |
   | `zoo_retrieval_recall_score` | 检索 Top-1 相关性分数 |
   | `zoo_intent_confidence` | 意图置信度分布 |
   | `zoo_escalation_rate` | 转人工率 |

2. 用 `prometheus-fastapi-instrumentator` 自动暴露 `/metrics` 端点

3. **Grafana Dashboard**：可视化延迟曲线、意图分布饼图、产品线流量热力图

4. **LangSmith 链路追踪**（可选）：集成 LangChain 官方追踪平台，每次 LLM 调用都有完整 Span 记录，方便 debug 幻觉问题

### Week 26：知识库管理 + 数据飞轮 + 云部署

1. **Admin API**（知识库热更新）：
   - `POST /admin/faq`：上传新 FAQ，自动 embedding 后写入 Milvus，无需重启
   - `DELETE /admin/faq/{id}`：软删除（标记 inactive），不影响正在服务的请求
   - `GET /admin/stats`：各产品线 FAQ 数量、最近召回频次 Top 10

2. **用户反馈闭环（Data Flywheel）**：

   ```
   用户点踩 / 反馈"回答不准"
     ↓
   写入 feedback 表（问题 + 回答 + 反馈类型）
     ↓
   每周批量分析：哪些问题被频繁差评？
     ↓
   补充 FAQ 数据 → 重建索引 → 系统自动变好
   ```

3. **Docker 多阶段构建**：将应用打包为单一镜像，分离 builder / runtime 层，镜像体积压缩 60%

4. **GitHub Actions CI/CD**：
   - Push 触发：运行 `pytest` → 构建 Docker 镜像 → 推送 Registry
   - 手动触发：一键部署到云服务器（阿里云 ECS / Railway / Render）

5. **云部署选项**：

   | 平台 | 成本 | 适合场景 |
   |------|------|---------|
   | Railway | 免费额度 5$/月 | 快速验证，无需运维 |
   | Render | 免费 tier（有冷启动） | 学习展示 |
   | 阿里云 ECS | 按量付费，~100元/月 | 国内访问稳定 |
   | Fly.io | 免费 3 个小实例 | 全球低延迟 |

### 核心概念

| 概念 | 含义 |
|------|------|
| **Data Flywheel** | 数据飞轮：用户使用产生数据 → 数据改善模型/知识库 → 系统变好吸引更多用户，形成正循环 |
| **Prometheus Pull Model** | Prometheus 主动拉取 `/metrics`，类比 Java Micrometer + Actuator |
| **Hot Reload** | 知识库更新无需停机，新旧 collection 版本切换对在飞请求透明 |
| **Blue-Green Deploy** | 蓝绿部署：新版本完全就绪后才切换流量，零停机 |

### 阶段产出

一个真正可以上线的 Zoo AI 客服系统：Redis 持久化 Session、Prometheus + Grafana 监控大盘、Admin 知识库管理界面、CI/CD 自动部署、数据反馈闭环。

---

## 学习路径总结

| 阶段 | 核心能力 | 关键技术 | 时间 |
|------|---------|---------|------|
| 第 0 阶段 | Python 基础 | Python 语法、环境管理 | Week 1~2 |
| 第 1 阶段 | LLM + LangChain 入门 | LangChain Chain、PromptTemplate、FastAPI | Week 3~5 |
| 第 2 阶段 | RAG 核心能力 | Embedding、向量库、文档切分、RAG Chain | Week 6~9 |
| 第 3 阶段 | 对话智能 | 意图识别、槽位填充、多轮对话、产品识别 | Week 10~13 |
| 第 4 阶段 | Agent 进阶 | LangChain Agent、Tool Calling、ReAct、多产品线 | Week 14~16 |
| 第 5 阶段 | 系统调优 | 召回评估、Hybrid Search、ReRank、A/B 测试 | Week 17~18 |
| 第 6 阶段 | 实时体验 | SSE 流式输出、React 前端、astream、Markdown 渲染 | Week 19~20 |
| 第 7 阶段 | 多 Agent 协作 | LangGraph StateGraph、Supervisor 模式、Human-in-the-loop、长期记忆 | Week 21~23 |
| 第 8 阶段 | 生产化 | Redis Session、Prometheus/Grafana、CI/CD、知识库热更新、数据飞轮 | Week 24~26 |

---

## 每阶段必须回答的 3 个问题

每个阶段完成后，用自己的话写下答案，写在项目的 `NOTES.md` 里：

1. 这个阶段的核心原理是什么？（用自己的语言，不要抄文档）
2. 我遇到了什么坑，是怎么解决的？
3. 这个技术和 Java 世界的什么概念类似？有什么区别？

---

## 推荐学习资源

- [LangChain 官方文档](https://python.langchain.com) — 必读，每个组件都有详细教程
- [LangChain 中文网](https://www.langchain.com.cn)
- [RAG 实战：构建企业级知识库问答系统](https://github.com)
- [OpenAI Function Calling 官方文档](https://platform.openai.com/docs/guides/function-calling)
- [BGE Embedding 模型](https://github.com/FlagOpen/FlagEmbedding)
- [FAISS 官方教程](https://github.com/facebookresearch/faiss)

---

## 技术栈全景图

| 层级 | 技术选型 |
|------|---------|
| 前端 | HTML + JS（Phase 1~5）→ React + TypeScript + Vite（Phase 6） |
| 后端 | Python FastAPI |
| AI 框架 | LangChain（Phase 1~5）→ LangGraph（Phase 7） |
| LLM | DeepSeek / 硅基流动 / 阿里通义（按需切换） |
| Embedding | BAAI/bge-m3（本地，1024 维） |
| 向量数据库 | Milvus（Docker，支持多 collection） |
| 关键词检索 | BM25（rank_bm25 + jieba 分词） |
| 精排模型 | BAAI/bge-reranker-base（Cross-Encoder） |
| Session 存储 | 进程内存（Phase 1~5）→ Redis（Phase 8） |
| 知识库 | FAQ JSON（Phase 1~4）→ Admin API 热更新（Phase 8） |
| 监控 | 结构化日志（Phase 2~5）→ Prometheus + Grafana（Phase 8） |
| 链路追踪 | RAG 日志（Phase 2）→ LangSmith（Phase 8，可选） |
| 评估 | Recall@K、MRR、McNemar A/B 检验 |
| CI/CD | GitHub Actions → Docker Registry → 云部署（Phase 8） |

---

> **核心原则**：每一步都能做出来，每一步都能讲清楚原理。你不需要一口气学完所有内容。Phase 0~5 是核心学习路径（18 周），完成后已具备完整的 AI 客服系统开发能力；Phase 6~8 是进阶扩展，按实际需要选择性完成，对标真实生产级系统的工程要求。
