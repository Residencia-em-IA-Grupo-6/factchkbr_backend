from fastapi import APIRouter, Depends, Request, status
from app.config import Settings, get_settings
from app.core.orchestrator import FactCheckOrchestrator
from app.core.registry import registry
from app.schemas.analysis import (
    AnalyzeRequest,
    AnalyzeResponse,
    HealthResponse,
)
from app.services.telegram_service import TelegramService, get_telegram_service

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
    settings: Settings = Depends(get_settings),
    orchestrator: FactCheckOrchestrator = Depends(get_orchestrator),
    telegram_service: TelegramService = Depends(get_telegram_service),
) -> AnalyzeResponse:
    """Recebe a solicitação, delega ao orquestrador e formata para o Telegram."""
    response = await orchestrator.analyze(text=request.text, urls=request.urls)

    # Gera a mensagem formatada para exibição no Telegram
    formatted_telegram = telegram_service.format_analysis_for_telegram(response, original_text=request.text)
    response.telegram_formatted_text = formatted_telegram

    # Entrega opcional e direta ao Telegram se chat_id foi fornecido
    if request.chat_id is not None:
        should_deliver = request.deliver_to_telegram or (
            settings.TELEGRAM_AUTO_REPLY and telegram_service.is_configured
        )
        if should_deliver and telegram_service.is_configured:
            delivery = await telegram_service.send_message(
                chat_id=request.chat_id,
                text=formatted_telegram,
            )
            response.telegram_delivery = delivery

    return response


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
