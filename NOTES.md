# Phase 1 学习笔记

> 来自 Java 背景开发者的理解，记录每个阶段的核心原理、踩坑经历和与 Java 世界的类比。

---

## Phase 1：构建最小可运行的 FAQ 机器人

### 1. Chain 的核心原理是什么？

**Chain = 工序流水线。**

类比 Java 的 `Stream API`，想象一条汽车生产线：
- 零件进来（用户提问）
- 每一道工序只做一件事（PromptTemplate 格式化、LLM 推理、OutputParser 解析）
- 每道工序的输出自动流入下一道工序
- 最终产出成品（回答文本）

LangChain 的 LCEL（LangChain Expression Language）用 `|` 管道符把这个流水线表达得非常直观：

```python
chain = (
    {"faq_context": RunnablePassthrough(), "question": RunnablePassthrough()}
    | prompt                      # 第一道工序：组装提示词
    | llm                         # 第二道工序：LLM 推理
    | StrOutputParser()           # 第三道工序：解析输出
)
```

关键点：
- `RunnablePassthrough()` = 把输入原封不动传给下一步
- 每一步都是**可组合的单元**，可以单独测试、替换、复用
- 和 Java Stream 的 `.map().filter().collect()` 概念几乎一模一样

### 2. 遇到了什么坑？怎么解决？

#### 坑 1：DeepSeek 的 API 端点格式

DeepSeek 使用 OpenAI 兼容接口，但 `base_url` 必须指向 `https://api.deepseek.com`，不能用 `https://api.deepseek.com/v1` 作为 base_url，因为 LangChain 内部会自动拼接 `/v1/chat/completions`。

**解决**：在 `ChatOpenAI` 中设置 `base_url="https://api.deepseek.com"`，LangChain 会自动处理版本路径。

#### 坑 2：Prompt 中 FAQ 上下文太长

第一次实现时，直接把所有 FAQ 塞进 system prompt，结果 token 消耗巨大，而且 LLM 容易"走神"（忽略部分内容）。

**解决**：
- Phase 1：FAQ 直接内联在 prompt 里（适合少量 FAQ）
- Phase 2：引入向量检索（embedding search），只召回最相关的 3~5 条 FAQ，大幅减少上下文长度

#### 坑 3：FastAPI 的 CORS 配置

前端调用 `POST /api/chat` 时，浏览器报 CORS 错误。

**解决**：在 `app/main.py` 中添加 `CORSMiddleware`，允许所有 origins（开发环境，生产环境要限制）：

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### 3. 这个技术和 Java 的什么概念类似？有什么区别？

| Python (LangChain) | Java 对应概念 | 关键区别 |
|---|---|---|
| `PromptTemplate` | `String.format()` / MessageFormat | 支持变量类型检查和模板复用，比字符串拼接安全 |
| `Chain` | `Stream` pipeline / `Pipeline` design pattern | 都是流水线组合，但 Chain 的每一步可以是 LLM（不确定的） |
| `LLM` | `HttpClient` 调用远程服务 | Java 的 HTTP 调用是确定性的；LLM 是概率性的，相同的输入可能有不同输出 |
| `ChatOpenAI` | RestTemplate / WebClient | 接口类似，但 ChatOpenAI 返回的是自然语言，不是结构化 JSON |
| `FastAPI @router.post` | `@RestController` | 非常相似，FastAPI 的优势是自动生成 OpenAPI 文档 |
| `.env` 配置 | Spring 的 `@ConfigurationProperties` | 理念相同，Spring 更企业化，Python 更轻量 |
| `lru_cache` | Spring `@Cacheable` | 作用相同，但 Python 的装饰器语法更简洁 |

**最核心的区别**：Java 是**确定性**的，LLM 是**概率性**的。在 Java 里，同样的输入一定有同样的输出；LLM 每次调用可能略有不同（由 temperature 控制）。这个思维转换是最重要的。

---

## 继续学习建议

- 阅读 LangChain 官方文档的 "LCEL" 章节，理解 Runnable 接口
- 学习 Prompt 工程：system prompt、few-shot examples、output format control
- Phase 2 将引入向量数据库（FAISS/Chroma），彻底改变 FAQ 的检索方式
