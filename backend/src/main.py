from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from core.config import settings
from modules.tools.router import router as tools_router


main_app = FastAPI(
    title="Tool Recognition API",
    description="API для сервиса анализа инструментов",
    version="1.0.0"
)

main_app.include_router(tools_router, prefix=settings.api.tools, tags=["Tools"])


main_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"],
)

@main_app.get("/")
async def root():
    return {
        "message": "Tool Recognition Service", 
        "version": "1.0.0",
        "status": "active"
    }

@main_app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(
        "main:main_app", 
        host=settings.run.host, 
        port=settings.run.port, 
        reload=True
    )