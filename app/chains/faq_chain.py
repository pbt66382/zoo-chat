"""
LangChain FAQ Chain for Zoo Meetings product line.
Uses LCEL (LangChain Expression Language) to build the pipeline:
  PromptTemplate -> ChatOpenAI -> StrOutputParser
"""
from typing import Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from app.llm.deepseek_client import get_llm
from data import FAQ_MEETINGS


# Build FAQ context string from the loaded FAQ data
def _build_faq_context() -> str:
    """Build a formatted string of all FAQ Q&A pairs for the prompt."""
    lines = []
    for faq in FAQ_MEETINGS:
        lines.append(f"【问题{faq['id']}】{faq['question']}")
        lines.append(f"【回答】{faq['answer']}")
        lines.append("")
    return "\n".join(lines)


FAQ_CONTEXT = _build_faq_context()


# System prompt template for the FAQ chatbot
SYSTEM_PROMPT = """你是一个专业的 Zoo 会议服务客服助手。请根据以下 FAQ 知识库回答用户的问题。

重要规则：
1. 只根据提供的 FAQ 内容回答，不要编造信息
2. 如果用户问题在 FAQ 中找不到相关信息，请礼貌地说："抱歉，这个问题我暂时无法回答，建议您联系人工客服获取帮助。"
3. 回答要简洁、专业、友好
4. 如果用户的问题涉及多个 FAQ 主题，合并相关内容回答

以下是 FAQ 知识库：
{faq_context}
"""


# User prompt template
USER_PROMPT = """用户问题: {question}

请根据上面的 FAQ 知识库回答用户的问题。"""


def build_faq_chain():
    """
    Build the FAQ Chain using LCEL (LangChain Expression Language).
    
    The chain is constructed as a pipeline using the | operator:
    1. PromptTemplate combines system prompt + user question
    2. LLM (DeepSeek) generates the answer
    3. StrOutputParser extracts the string output
    
    Returns:
        A runnable LangChain chain ready to be invoked
    """
    # Create prompt template by combining system and user prompts
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", USER_PROMPT),
    ])
    
    # Get the LLM instance
    llm = get_llm()
    
    # Build the chain using LCEL pipe operator
    # Each | passes the output of the previous step as input to the next
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
    Convenience function to invoke the FAQ chain with a single question.
    
    Args:
        question: The user's question
        
    Returns:
        The generated answer from the FAQ chain
    """
    chain = build_faq_chain()
    result = chain.invoke({
        "faq_context": FAQ_CONTEXT,
        "question": question,
    })
    return result


def find_faq_by_question(question: str) -> Optional[dict]:
    """
    Simple keyword-based FAQ lookup (Phase 1, no vector search yet).
    This will be replaced by RAG in Phase 2.
    
    Args:
        question: User's question
        
    Returns:
        Matching FAQ dict or None
    """
    q_lower = question.lower()
    for faq in FAQ_MEETINGS:
        # Simple keyword matching
        keywords = [tag.lower() for tag in faq.get("tags", [])]
        if any(kw in q_lower for kw in keywords):
            return faq
    return None
