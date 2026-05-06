"""
FastAPI 应用入口 - Zoo AI Chat Phase 1。
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from app.api.chat import router as chat_router
from config.settings import get_settings


# 创建 FastAPI 应用实例
app = FastAPI(
    title="Zoo AI Chat - Phase 1",
    description="Zoo 会议服务最小 FAQ 机器人，使用 DeepSeek + LangChain + FastAPI",
    version="1.0.0",
)

# 配置 CORS，允许前端跨域调用 API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境请限制具体的域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载静态文件目录（前端页面）
frontend_dir = Path(__file__).parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")

# 注册路由
app.include_router(chat_router)


@app.get("/")
def root():
    """根路径 - 返回服务信息。"""
    return {
        "message": "Zoo AI Chat API 已启动",
        "docs": "/docs",
        "frontend": "/static/index.html",
    }


@app.get("/health")
def health_check():
    """健康检查接口，供监控使用。"""
    settings = get_settings()
    return {
        "status": "healthy",
        "version": "1.0.0",
        "model": settings.deepseek_model,
    }


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.debug,
    )
