"""
Endpoints dedicados à integração com o Telegram Bot API.
Oferece recebimento via Webhook, entrega de mensagens e gerenciamento do webhook.
"""

import logging
from typing import Any
from fastapi import APIRouter, BackgroundTasks, Depends, Header, HTTPException, Request, status

from app.api.v1.endpoints import get_orchestrator
from app.config import Settings, get_settings
from app.core.orchestrator import FactCheckOrchestrator
from app.schemas.analysis import TelegramDeliveryStatus
from app.schemas.telegram import (
    TelegramSendMessageRequest,
    TelegramSetWebhookRequest,
    TelegramUpdate,
)
from app.services.telegram_service import TelegramService, get_telegram_service

logger = logging.getLogger("factchkbr.api.telegram")
router = APIRouter(prefix="/telegram", tags=["Telegram Bot"])


@router.post(
    "/webhook",
    status_code=status.HTTP_200_OK,
    summary="Receber Updates do Telegram (Webhook)",
    description="Endpoint registrado junto à Telegram Bot API para receber mensagens e comandos de usuários."
)
async def telegram_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    x_telegram_bot_api_secret_token: str | None = Header(default=None),
    settings: Settings = Depends(get_settings),
    telegram_service: TelegramService = Depends(get_telegram_service),
    orchestrator: FactCheckOrchestrator = Depends(get_orchestrator),
) -> dict[str, bool]:
    """
    Recebe o payload enviado pelos servidores do Telegram.
    Valida a autenticidade do webhook e processa a checagem em background
    para responder ao Telegram dentro do prazo de timeout de 2 segundos.
    """
    # Validação de segurança via Secret Token (se configurado)
    expected_secret = settings.TELEGRAM_WEBHOOK_SECRET
    if expected_secret:
        if not x_telegram_bot_api_secret_token or x_telegram_bot_api_secret_token != expected_secret:
            logger.warning("Tentativa de webhook do Telegram com token secreto inválido.")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Invalid secret token."
            )

    try:
        body = await request.json()
    except Exception as e:
        logger.warning("Falha ao deserializar JSON do webhook do Telegram: %s", e)
        return {"ok": False}

    # Despacha o processamento analítico para background, retornando 200 OK imediatamente ao Telegram
    background_tasks.add_task(
        telegram_service.process_incoming_update,
        update_data=body,
        orchestrator=orchestrator,
    )

    return {"ok": True}


@router.post(
    "/send",
    response_model=TelegramDeliveryStatus,
    status_code=status.HTTP_200_OK,
    summary="Entregar Mensagem no Telegram",
    description="Envia diretamente uma mensagem de texto (com formatação HTML) para um chat_id do Telegram."
)
async def send_telegram_message(
    payload: TelegramSendMessageRequest,
    telegram_service: TelegramService = Depends(get_telegram_service),
) -> TelegramDeliveryStatus:
    """Entrega manual de mensagem a um chat do Telegram."""
    return await telegram_service.send_message(
        chat_id=payload.chat_id,
        text=payload.text,
        parse_mode=payload.parse_mode,
        reply_to_message_id=payload.reply_to_message_id,
        disable_web_page_preview=payload.disable_web_page_preview,
    )


@router.post(
    "/set-webhook",
    summary="Configurar Webhook no Telegram",
    description="Registra a URL pública da sua API nos servidores do Telegram."
)
async def set_telegram_webhook(
    payload: TelegramSetWebhookRequest,
    telegram_service: TelegramService = Depends(get_telegram_service),
) -> dict[str, Any]:
    """Configura o webhook do bot chamando a Telegram Bot API."""
    return await telegram_service.set_webhook(
        url=payload.url,
        secret_token=payload.secret_token,
        drop_pending_updates=payload.drop_pending_updates,
    )


@router.get(
    "/webhook-info",
    summary="Informações do Webhook Atual",
    description="Consulta os servidores do Telegram para verificar o status e métricas do webhook configurado."
)
async def get_telegram_webhook_info(
    telegram_service: TelegramService = Depends(get_telegram_service),
) -> dict[str, Any]:
    """Retorna os metadados do webhook registrado no Telegram."""
    return await telegram_service.get_webhook_info()


@router.post(
    "/delete-webhook",
    summary="Remover Webhook do Telegram",
    description="Desativa o recebimento via webhook (necessário caso deseje rodar o bot via long-polling)."
)
async def delete_telegram_webhook(
    drop_pending_updates: bool = False,
    telegram_service: TelegramService = Depends(get_telegram_service),
) -> dict[str, Any]:
    """Remove o webhook cadastrado no Telegram."""
    return await telegram_service.delete_webhook(drop_pending_updates=drop_pending_updates)
