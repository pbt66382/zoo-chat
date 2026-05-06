"""
LangChain FAQ Chain 模块 - Zoo 会议服务产品线。
使用 LCEL（LangChain Expression Language）构建流水线：
  PromptTemplate -> ChatOpenAI -> StrOutputParser
"""
from typing import Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from app.llm.deepseek_client import get_llm
from data import FAQ_MEETINGS


def _build_faq_context() -> str:
    """将所有 FAQ 问答对拼接成供 Prompt 使用的字符串。"""
    lines = []
    for faq in FAQ_MEETINGS:
        lines.append(f"【问题{faq['id']}】{faq['question']}")
        lines.append(f"【回答】{faq['answer']}")
        lines.append("")
    return "\n".join(lines)


FAQ_CONTEXT = _build_faq_context()


# 系统 Prompt 模板
SYSTEM_PROMPT = """你是一个专业的 Zoo 会议服务客服助手。请根据以下 FAQ 知识库回答用户的问题。

重要规则：
1. 只根据提供的 FAQ 内容回答，不要编造信息
2. 如果用户问题在 FAQ 中找不到相关信息，请礼貌地说："抱歉，这个问题我暂时无法回答，建议您联系人工客服获取帮助。"
3. 回答要简洁、专业、友好
4. 如果用户的问题涉及多个 FAQ 主题，合并相关内容回答

以下是 FAQ 知识库：
{faq_context}
"""


# 用户 Prompt 模板
USER_PROMPT = """用户问题: {question}

请根据上面的 FAQ 知识库回答用户的问题。"""


def build_faq_chain():
    """
    使用 LCEL（LangChain Expression Language）构建 FAQ Chain。

    流水线用 | 管道符串联：
    1. PromptTemplate 组合 system prompt + user question
    2. LLM（DeepSeek）生成回答
    3. StrOutputParser 提取字符串输出

    返回:
        一个可执行的 LangChain chain
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", USER_PROMPT),
    ])

    llm = get_llm()

    # 使用 LCEL 管道符构建 Chain
    # 每个 | 将上一步的输出传给下一步
    chain = (
        {
            "faq_context": RunnablePassthrough(),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


def invoke_faq_chain(question: str) -> str:
    """
    调用 FAQ Chain 回答单个问题的便捷函数。

    参数:
        question: 用户的问题

    返回:
        Chain 生成的回答文本
    """
    chain = build_faq_chain()
    result = chain.invoke({
        "faq_context": FAQ_CONTEXT,
        "question": question,
    })
    return result


def find_faq_by_question(question: str) -> Optional[dict]:
    """
    简单的关键词匹配 FAQ 检索（Phase 1，向量检索将在 Phase 2 引入）。

    参数:
        question: 用户的问题

    返回:
        匹配的 FAQ 字典，未找到则返回 None
    """
    q_lower = question.lower()
    for faq in FAQ_MEETINGS:
        keywords = [tag.lower() for tag in faq.get("tags", [])]
        if any(kw in q_lower for kw in keywords):
            return faq
    return None
