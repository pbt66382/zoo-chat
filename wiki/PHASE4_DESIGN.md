# Zoo Chat - Phase 4 设计文档

> 日期：2026-05-13
> 主题：AI Agent 化升级 + 多产品线扩展

---

## 一、目标

按学习计划 Phase 4 完整完成：

1. **ZooAgent**：用 LangChain Tool Calling Agent 替代固定 Pipeline，实现 LLM 自主规划 + 工具调用（ReAct 模式）。
2. **多产品线扩展**：支持 6 条产品线（会议/可视电话/耳机/鼠标/会议大屏/通话服务）。
3. **产品线检测**：首轮问题自动判断产品，同一 session 内持久化，避免重复检测。
4. **双模执行**：Pipeline 模式（Phase 3）与 Agent 模式并存，通过请求参数或环境变量切换。

---

## 二、架构

### 双模式执行路径

```
POST /api/chat { use_agent: false (默认) }
  └── ChatService._run_pipeline()
        └── ProductDetectionStep → IntentStep → RouterStep → SlotFillingStep → RetrievalStep → GenerationStep

POST /api/chat { use_agent: true }
  └── ChatService._run_agent()
        └── ProductDetector.detect()（若 session 无缓存）
        └── ZooAgent.chat()
              └── LangChain AgentExecutor（Tool Calling Agent）
                    └── ReAct 循环：Think → Tool Call → Observe → ... → Final Answer
```

### 关键模块

| 模块 | 职责 |
|------|------|
| `app/agent/zoo_agent.py` | ZooAgent 主类 + Tool 工厂 + AgentExecutor 封装 |
| `app/product/detector.py` | ProductDetector：关键词预判 + LLM 语义确认双重校验 |
| `app/pipeline/product_detection_step.py` | Pipeline 中的产品检测步骤（session 命中则跳过） |
| `app/services/chat_service.py` | 双模分流入口，产品线 session 持久化 |
| `app/api/chat.py` | `use_agent` 请求参数 + `product_id/product_name/used_agent/agent_steps` 响应字段 |
| `config/settings.py` | `AGENT_MODE_ENABLED`、`AGENT_MAX_ITERATIONS` 配置项 |

---

## 三、ZooAgent 详解

### ReAct 与 Tool Calling 模式区别

| 模式 | 原理 | 优缺点 |
|------|------|--------|
| ReAct（文本解析） | LLM 输出 `Thought/Action/Observation` 文本，框架解析 | 通用但脆弱，解析易出错 |
| Tool Calling（结构化） | LLM 输出结构化 JSON 工具调用，框架直接执行 | 更稳定，DeepSeek 原生支持 |

本项目选用 `create_tool_calling_agent`，使用 DeepSeek 的 Function Calling 能力。

### 工具列表

| 工具 | 触发场景 | 返回 |
|------|---------|------|
| `query_product_doc` | 用户询问产品功能/操作/故障排查 | Milvus 向量检索 Top-3 FAQ 片段 |
| `query_faq_exact` | 包含型号、错误码等精确词 | Milvus filter 精确匹配 |
| `check_order_status` | 用户询问订单/物流 | 模拟订单 API 结果 |
| `create_ticket` | 问题无法立即解决 | 工单号 + 处理时效 |
| `escalate_to_human` | 问题复杂/用户情绪激动/AI 无把握 | 人工客服联系方式 |

工具通过 `_make_tools(collection)` 工厂函数创建，绑定了当前产品线对应的 Milvus collection。

### AgentResult 数据结构

```python
@dataclass
class AgentResult:
    answer: str
    steps: list[dict]    # 每步 {tool, input, output[:200]}
    total_ms: float
    tool_calls: int
```

---

## 四、产品线检测

### 支持的 6 + 1 条产品线

| product_id | 产品名 | Milvus collection |
|------------|-------|-------------------|
| `meetings` | 会议服务 | `zoo_faq_meetings` |
| `phone` | 可视电话 | `zoo_faq_phone` |
| `earbuds` | 耳机 | `zoo_faq_earbuds` |
| `mouse` | 鼠标 | `zoo_faq_mouse` |
| `screen` | 会议大屏 | `zoo_faq_screen` |
| `calls` | 通话服务 | `zoo_faq_calls` |
| `general` | 通用 | `zoo_faq_meetings`（通用问题走会议库） |

### 检测策略：双重校验

```
用户问题
  ├── 关键词快速预判（零 LLM 调用，遍历关键词命中最多的产品）
  └── LLM 语义分类（system prompt 列出所有产品线描述，输出 JSON）
        ↓
  规则融合：
  - 两者一致 → confidence + 0.15（最高 1.0）
  - LLM 选 general 但关键词命中其他产品 → 采用关键词结果，conf=0.7
  - LLM 解析失败 → 回退到关键词结果
```

### Session 内产品线持久化

同一 session 首轮检测后，将 `product_id` 和 `milvus_collection` 写入 `SessionState`，后续轮次直接从 session 读取，跳过 LLM 调用。

---

## 五、多产品线 FAQ 数据

| 文件 | 产品线 | 说明 |
|------|-------|------|
| `data/faq_meetings.json` | 会议服务 | 150 条，Phase 2 已建立索引 |
| `data/faq_phone.json` | 可视电话 | SIP 注册、固件升级、话机激活等 |
| `data/faq_earbuds.json` | 耳机 | 蓝牙配对、音质、降噪、USB 耳机 |
| `data/faq_mouse.json` | 鼠标 | 大屏控制、批注笔、配对 |
| `data/faq_screen.json` | 会议大屏 | 投屏、白板、触摸屏、无法入会 |

各产品线 FAQ 通过 `scripts/build_milvus_index.py` 索引到对应 collection。

---

## 六、API 协议变更

`POST /api/chat` 新增字段：

请求：

```json
{
  "message": "我的耳机蓝牙连不上",
  "session_id": "optional",
  "use_agent": true
}
```

响应（新增字段）：

```json
{
  "answer": "请确认耳机已进入配对模式...",
  "session_id": "session_xxx",
  "latency_ms": 1234.56,
  "product_id": "earbuds",
  "product_name": "耳机",
  "used_agent": true,
  "agent_steps": [
    {"tool": "query_product_doc", "input": {"query": "蓝牙耳机连不上"}, "output": "[相关度 0.9123] ..."}
  ]
}
```

---

## 七、环境变量配置（Phase 4 新增）

```bash
# Agent 模式（默认 false，用 Pipeline 模式）
AGENT_MODE_ENABLED=false

# Agent 最大工具调用轮次（防止死循环）
AGENT_MAX_ITERATIONS=5

# 产品线检测开关（默认 true）
PRODUCT_DETECTION_ENABLED=true

# 默认 Milvus collection（检测失败时的兜底）
DEFAULT_COLLECTION=zoo_faq_meetings
```

---

## 八、Pipeline 模式 vs Agent 模式 对比

| 维度 | Pipeline 模式 | Agent 模式 |
|------|-------------|----------|
| 执行顺序 | 固定（ProductDetection → Intent → ... → Generation） | LLM 自主决定工具调用顺序 |
| 可解释性 | 每 Step 独立日志，完全可追溯 | 通过 `agent_steps` 字段查看工具调用链 |
| 延迟 | 较低（1-2 次 LLM 调用） | 较高（可能 3-5 次 LLM 调用） |
| 适用场景 | 标准 FAQ 问答，流程确定 | 多步推理、需要订单查询/工单创建等工具组合 |
| 槽位填充 | ✅（Pipeline SlotFillingStep） | ❌（Agent 自行决策，无显式槽位状态） |

**推荐**：默认使用 Pipeline 模式；`use_agent=true` 留给需要工具组合的复杂请求。

---

## 九、Phase 4 完成清单

- [x] `ZooAgent` 实现：Tool Calling Agent + 5 个工具 + ReAct 循环
- [x] `ProductDetector` 实现：关键词 + LLM 双重校验，7 条产品线
- [x] `ProductDetectionStep` 集成到 Pipeline，支持 session 持久化
- [x] `ChatService` 双模切换：`use_agent` 参数 + `AGENT_MODE_ENABLED` 全局配置
- [x] API `ChatRequest.use_agent` / `ChatResponse.product_id/used_agent/agent_steps`
- [x] 4 条新产品线 FAQ 数据：phone / earbuds / mouse / screen
- [x] `config/settings.py` Phase 4 配置项

---

## 十、与 Phase 3 的关系

Phase 4 完全向后兼容 Phase 3。默认仍走 Pipeline 模式，新增的 `ProductDetectionStep` 插在 `IntentStep` 之前，旧接口（无 `use_agent` 字段）行为不变。Agent 模式是独立分支，不影响 Pipeline 代码路径。
