"""
FastAPI 应用入口 - Zoo AI Chat。
"""
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.chat import router as chat_router
from app.api.logs_route import router as logs_router
from app.logging_setup import configure_logging
from app.middleware.request_logging import RequestContextMiddleware
from config.settings import get_settings

configure_logging()

app = FastAPI(
    title="Zoo AI Chat - Phase 3",
    description="Zoo 会议服务 AI 客服：意图识别 + 槽位填充 + RAG（DeepSeek + LangChain + Milvus）",
    version="3.0.0",
)

app.add_middleware(RequestContextMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

frontend_dir = Path(__file__).parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")

app.include_router(chat_router)
app.include_router(logs_router)


@app.get("/")
def root():
    return {
        "message": "Zoo AI Chat API 已启动",
        "docs": "/docs",
        "frontend": "/static/index.html",
    }


@app.get("/health")
def health_check():
    settings = get_settings()
    return {
        "status": "healthy",
        "version": "3.0.0",
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
