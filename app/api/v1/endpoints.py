from fastapi import APIRouter, Depends, Request, status
from app.config import Settings, get_settings
from app.core.orchestrator import FactCheckOrchestrator
from app.core.registry import registry
from app.schemas.analysis import (
    AnalyzeRequest,
    AnalyzeResponse,
    HealthResponse,
)

router = APIRouter()


def get_orchestrator(
    request: Request,
    settings: Settings = Depends(get_settings),
) -> FactCheckOrchestrator:
    if hasattr(request.app.state, "orchestrator") and request.app.state.orchestrator is not None:
        return request.app.state.orchestrator
    return FactCheckOrchestrator(settings=settings)


@router.post(
    "/analyze",
    response_model=AnalyzeResponse,
    status_code=status.HTTP_200_OK,
    summary="Análise de Veracidade",
    description="Endpoint consumido pelo Bot do Telegram para verificar a veracidade de uma mensagem."
)
async def analyze(
    request: AnalyzeRequest,
    orchestrator: FactCheckOrchestrator = Depends(get_orchestrator)
) -> AnalyzeResponse:
    """Recebe a solicitação e delega ao orquestrador."""
    return await orchestrator.analyze(text=request.text, urls=request.urls)


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Status da Aplicação",
    description="Verifica a integridade da API e lista os analisadores ativos e disponíveis."
)
async def health(
    settings: Settings = Depends(get_settings),
    orchestrator: FactCheckOrchestrator = Depends(get_orchestrator)
) -> HealthResponse:
    active = [a.name for a in orchestrator.get_active_analyzers()]
    available = registry.list_available()

    return HealthResponse(
        status="ok",
        version=settings.APP_VERSION,
        active_analyzers=active,
        available_analyzers=available
    )
