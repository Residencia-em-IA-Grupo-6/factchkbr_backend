from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1.router import router as api_router
from app.config import get_settings
from app.core.registry import registry


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Ciclo de vida: inicializa a descoberta de analisadores no Registry."""
    registry.auto_discover("app.analyzers")
    yield


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Rota exigida pelo Bot do Telegram: /api/analyze e /api/health
    app.include_router(api_router, prefix="/api")

    # Rota raiz de healthcheck
    @app.get("/health", tags=["Infraestrutura"])
    async def root_health():
        return {"status": "ok", "app": settings.APP_NAME}

    return app


app = create_app()
