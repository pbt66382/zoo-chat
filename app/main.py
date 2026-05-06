"""
FastAPI application entry point for Zoo AI Chat - Phase 1.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from app.api.chat import router as chat_router
from config.settings import get_settings


# Create FastAPI app
app = FastAPI(
    title="Zoo AI Chat - Phase 1",
    description="Minimal FAQ chatbot for Zoo Meetings product line using DeepSeek + LangChain + FastAPI",
    version="1.0.0",
)

# Configure CORS to allow frontend to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for frontend
frontend_dir = Path(__file__).parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")

# Include routers
app.include_router(chat_router)


@app.get("/")
def root():
    """Root endpoint - redirect to frontend."""
    return {
        "message": "Zoo AI Chat API is running",
        "docs": "/docs",
        "frontend": "/static/index.html",
    }


@app.get("/health")
def health_check():
    """Health check endpoint for monitoring."""
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
