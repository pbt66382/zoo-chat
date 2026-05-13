# Zoo Chat - Phase 5 设计文档

> 日期：2026-05-13
> 主题：召回策略优化与调优

---

## 一、目标

按学习计划 Phase 5 完整完成：

1. **多策略召回**：实现纯向量检索、BM25 关键词检索、Hybrid Search（RRF 融合）三种策略，支持运行时切换。
2. **Reranker 精排**：Cross-Encoder 两阶段精排（可选，通过 `RERANK_ENABLED` 开关）。
3. **召回评估**：Recall@K + MRR 指标，多产品线评估集（`recall_eval.jsonl`）。
4. **A/B 测试**：策略对比实验 + McNemar 统计显著性检验 + Markdown 报告生成。

---

## 二、架构

### 召回策略分层

```
用户问题
  ↓
RetrievalStep（app/pipeline/retrieval_step.py）
  ├── strategy=vector  → _vector_search()      Milvus 向量检索
  ├── strategy=bm25    → BM25Retriever.search() BM25 关键词检索
  └── strategy=hybrid  → _hybrid_search()
        ├── _vector_search(top_k * 3)           粗排：向量 Top-N
        ├── BM25Retriever.search(top_k * 3)     粗排：BM25 Top-N
        ├── RRF 融合排序                         精排：互倒排名融合
        └── [可选] CrossEncoderReranker.rerank() 精排：Cross-Encoder 重排
```

### 关键模块

| 模块 | 职责 |
|------|------|
| `app/pipeline/retrieval_step.py` | Pipeline 中的检索步骤，读取 `RETRIEVAL_STRATEGY` 选择策略 |
| `app/retrieval/bm25_retriever.py` | BM25Okapi 中文检索器，jieba 分词，内存缓存 |
| `app/retrieval/reranker.py` | CrossEncoderReranker，sentence-transformers 实现 |
| `scripts/eval_recall.py` | 召回质量评估：Recall@K + MRR + per-product 分析 |
| `scripts/ab_test.py` | A/B 对比实验：多策略并行 + McNemar 检验 + Markdown 报告 |
| `data/eval/recall_eval.jsonl` | 评估数据集，每条含 `question / product / expected_faq_ids` |

---

## 三、BM25 检索器

### 原理

BM25（Best Match 25）是 TF-IDF 的升级版：

```
score(q, d) = Σ IDF(t) × tf × (k1+1) / (tf + k1 × (1 - b + b×|d|/avgdl))
```

- `k1=1.5`：词频饱和参数，避免高频词无限加分
- `b=0.75`：文档长度归一化，长文档词频打折
- 中文处理：优先用 `jieba` 分词，未安装时回退到字符级切分

### 适用场景

- 问题包含精确型号（如"E-1204 错误"）或错误码
- 包含品牌名、专有名词（向量检索语义空间可能分散）
- 作为 Hybrid Search 的补充信号

### 缓存策略

每个 collection 的 BM25 索引按需构建后缓存在进程内存（`_CACHE` 字典）。FAQ 数据更新后需调用 `BM25Retriever.clear_cache()` 或重启服务。

---

## 四、Hybrid Search（RRF 融合）

### 两阶段策略

```
第一阶段（粗排，快）：
  - 向量检索召回 Top-K×3 候选
  - BM25 检索召回 Top-K×3 候选

第二阶段（精排，准）：
  - RRF 融合：score(d) = Σ 1/(k + rank_i)  k=60
  - [可选] Cross-Encoder 对 RRF Top-N 重排
```

### 为什么用 RRF 而非加权融合

| 方案 | 问题 |
|------|------|
| 加权融合（0.7×向量 + 0.3×BM25） | 两种分数量级不同，需要归一化，权重敏感 |
| RRF | 只看排名不看分数，无需归一化，对单一列表排名更鲁棒 |

RRF 是召回融合的工业界标准方案，无需调参即有良好效果。

---

## 五、Cross-Encoder Reranker

### Bi-Encoder vs Cross-Encoder

| 维度 | Bi-Encoder（向量检索） | Cross-Encoder（精排） |
|------|---------------------|---------------------|
| 编码方式 | 查询和文档分别编码 | 查询+文档拼接后一起编码 |
| 速度 | 快（O(1) 向量查找） | 慢（每对需一次前向传播） |
| 精度 | 较低（无词级交互） | 高（注意力机制看全文交互） |
| 适用 | 大规模粗排 | 小规模精排（Top-20 → Top-3） |

### 推荐两阶段

```
向量检索 Top-20（粗排，<100ms）→ Cross-Encoder 重排 Top-3（精排，~200ms）
```

### 推荐模型

- 中文：`BAAI/bge-reranker-base`（与 bge-m3 embedding 搭配效果好）
- 更高精度：`BAAI/bge-reranker-large`（约 4x 延迟）

通过 `RERANK_MODEL` 环境变量配置，默认 `BAAI/bge-reranker-base`。

---

## 六、环境变量配置（Phase 5 新增）

```bash
# 召回策略：vector | bm25 | hybrid（默认 vector）
RETRIEVAL_STRATEGY=vector

# 是否启用 Cross-Encoder Reranker（默认 false）
RERANK_ENABLED=false

# Reranker 模型（RERANK_ENABLED=true 时生效）
RERANK_MODEL=BAAI/bge-reranker-base

# Reranker 粗排候选数（对 top_n 个候选重排）
RERANK_TOP_N=10

# Top-K 召回数量（默认 3）
TOP_K=3

# Hybrid Search 权重参考（当前实现用 RRF，权重暂未生效）
BM25_WEIGHT=0.3
VECTOR_WEIGHT=0.7
```

---

## 七、评估工具

### eval_recall.py

```bash
# 基础用法：向量检索评估
python scripts/eval_recall.py

# 指定策略和 Top-K
python scripts/eval_recall.py --strategy hybrid --top-k 5

# 只评估指定产品线
python scripts/eval_recall.py --product meetings phone

# 显示失败案例
python scripts/eval_recall.py --verbose
```

**指标**：

| 指标 | 含义 |
|------|------|
| `Recall@K` | Top-K 中命中期望 FAQ 的比例 |
| `MRR` | Mean Reciprocal Rank，衡量期望 FAQ 的平均排名（1/rank） |
| `per_product` | 各产品线的 Recall@K 分布 |

### ab_test.py

```bash
# 对比全部策略（vector / bm25 / hybrid）
python scripts/ab_test.py

# 只对比两种策略
python scripts/ab_test.py --strategies vector hybrid

# 调整 Top-K，保存报告
python scripts/ab_test.py --top-k 5 --output wiki/PHASE5_AB_REPORT.md
```

**McNemar 检验**：

- 针对二元结果（命中/未命中）的配对检验
- `χ² = (|b-c| - 1)² / (b+c)`，使用 Yates 连续性修正
- `p < 0.05` 表示两策略差异显著，可置信地选择更优方案

---

## 八、评估数据集格式

`data/eval/recall_eval.jsonl`（每行一条 JSON）：

```jsonl
{"question": "蓝牙耳机连不上手机怎么办", "product": "earbuds", "expected_faq_ids": [3, 7]}
{"question": "如何升级话机固件", "product": "phone", "expected_faq_ids": [12]}
```

| 字段 | 说明 |
|------|------|
| `question` | 测试问题 |
| `product` | 产品线 ID（meetings/phone/earbuds/mouse/screen） |
| `expected_faq_ids` | 期望召回的 FAQ ID 列表（至少命中其中一个即为 hit） |

---

## 九、召回策略选型建议

| 场景 | 推荐策略 | 理由 |
|------|---------|------|
| 语义相近、表述多样的问题 | `vector` | Embedding 捕获语义 |
| 含型号、错误码等精确词 | `bm25` | 关键词精确匹配 |
| 通用客服场景（推荐默认） | `hybrid` | 兼顾语义和关键词，RRF 无需调参 |
| 对精确度要求极高 | `hybrid + reranker` | 最高精度，延迟增加 ~200ms |

---

## 十、Phase 5 完成清单

- [x] `BM25Retriever`：jieba 分词 + rank_bm25，5 个产品线 collection 支持，内存缓存
- [x] `CrossEncoderReranker`：sentence-transformers CrossEncoder，`lru_cache` 单例
- [x] `RetrievalStep` 三策略支持：`vector` / `bm25` / `hybrid`，`RETRIEVAL_STRATEGY` 控制
- [x] `_hybrid_search`：RRF 融合 + 可选 Reranker 精排
- [x] `scripts/eval_recall.py`：Recall@K + MRR + per-product + verbose 失败分析
- [x] `scripts/ab_test.py`：多策略并行评估 + McNemar 检验 + Markdown 报告生成
- [x] `data/eval/recall_eval.jsonl`：多产品线评估集
- [x] `config/settings.py` Phase 5 配置项（RETRIEVAL_STRATEGY / RERANK_* / TOP_K）

---

## 十一、运行 A/B 实验

```bash
# 确保 Milvus 服务已启动
docker compose -f docker-compose-milvus.yml up -d

# 确保各产品线 collection 已建立索引
python scripts/build_milvus_index.py

# 运行 A/B 对比实验，生成报告
python scripts/ab_test.py --top-k 3 --output wiki/PHASE5_AB_REPORT.md
```

报告将包含：Recall@K 对比表、各策略置信区间（Wilson score interval）、McNemar 显著性检验结论。
