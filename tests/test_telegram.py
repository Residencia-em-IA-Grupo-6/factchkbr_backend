"""
Testes unitários e de integração para a camada de comunicação do Bot do Telegram.
"""

from unittest.mock import AsyncMock, patch
import pytest
from fastapi.testclient import TestClient
from app.config import Settings
from app.main import app
from app.schemas.analysis import AnalyzeResponse, SubClaimAnalysis, Verdict
from app.services.telegram_service import TelegramService, get_telegram_service

client = TestClient(app)


def test_telegram_format_analysis_html():
    """Valida se a formatação para o Telegram gera HTML bem estruturado e com todas as seções."""
    service = TelegramService()

    analysis = AnalyzeResponse(
        claim="Água com limão emagrece em 3 dias",
        verdict=Verdict.FAKE,
        confidence=0.85,
        summary="A alegação não possui comprovação científica ou médica reconhecida.",
        reasons=[
            "Não há evidências clínicas que sustentem perda de gordura acelerada por ingestão de limão.",
            "Conselho Federal de Nutrição e OMS alertam para dietas milagrosas sem respaldo.",
        ],
        sources=[
            "https://saude.abril.com.br/nutricao/agua-com-limao-emagrece",
            "Anvisa - Informações Regulatórias",
        ],
        sub_claims=[
            SubClaimAnalysis(
                statement="Água com limão emagrece",
                verdict=Verdict.FAKE,
                confidence=0.85,
                justification="Sem respaldo clínico.",
                sources=["Anvisa"],
            ),
            SubClaimAnalysis(
                statement="Limão queima gordura localizada",
                verdict=Verdict.FAKE,
                confidence=0.90,
                justification="Mecanismo biológico inexistente.",
                sources=["Europe PMC"],
            ),
        ],
    )

    formatted = service.format_analysis_for_telegram(analysis)

    assert "FACTCHKBR - VERIFICAÇÃO FACTUAL" in formatted
    assert "❌ <b>FAKE / BOATO</b>" in formatted
    assert "85.0%" in formatted
    assert "Água com limão emagrece em 3 dias" in formatted
    assert "A alegação não possui comprovação científica" in formatted
    assert "Detalhamento por Alegação" in formatted
    assert "Principais Fundamentos" in formatted
    assert "Fontes Consultadas" in formatted
    assert "https://saude.abril.com.br" in formatted


def test_telegram_format_html_escaping():
    """Garante que caracteres especiais (<, >, &) sejam devidamente escapados para evitar erro no Telegram."""
    service = TelegramService()

    analysis = AnalyzeResponse(
        claim="Vacinas contêm microchips & substâncias com pH < 2 e pH > 12",
        verdict=Verdict.FAKE,
        confidence=0.95,
        summary="Texto contendo tags fictícias como <script>alert(1)</script> e & entidades.",
        reasons=["Razão com < e > e &."],
        sources=["Fonte <Oficial> & Segura"],
    )

    formatted = service.format_analysis_for_telegram(analysis)
    assert "<script>" not in formatted
    assert "&lt;script&gt;" in formatted
    assert "pH &lt; 2" in formatted
    assert "pH &gt; 12" in formatted
    assert "&amp;" in formatted


@pytest.mark.asyncio
async def test_telegram_send_message_unconfigured():
    """Garante comportamento defensivo caso TELEGRAM_BOT_TOKEN não esteja setado."""
    custom_settings = Settings(TELEGRAM_BOT_TOKEN=None)
    service = TelegramService(settings=custom_settings)

    status = await service.send_message(chat_id=123456, text="Teste")
    assert not status.delivered
    assert "não configurado" in (status.error or "")


@pytest.mark.asyncio
async def test_telegram_send_message_success():
    """Valida envio de mensagem com resposta de sucesso simulada da API do Telegram."""
    from unittest.mock import MagicMock
    custom_settings = Settings(TELEGRAM_BOT_TOKEN="fake_token_12345")
    service = TelegramService(settings=custom_settings)

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "ok": True,
        "result": {
            "message_id": 9988,
            "chat": {"id": 123456},
            "text": "Mensagem recebida!",
        }
    }

    with patch.object(service._get_client(), "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_resp
        res = await service.send_message(chat_id=123456, text="<b>Teste</b>")
        assert res.delivered
        assert res.message_id == 9988
        assert res.error is None


@pytest.mark.asyncio
async def test_telegram_process_start_command():
    """Garante resposta de boas-vindas ao receber o comando /start."""
    custom_settings = Settings(TELEGRAM_BOT_TOKEN="fake_token_12345")
    service = TelegramService(settings=custom_settings)

    mock_send = AsyncMock()
    mock_send.return_value = AsyncMock(delivered=True, message_id=101)

    update = {
        "update_id": 1,
        "message": {
            "message_id": 55,
            "chat": {"id": 12345, "type": "private"},
            "text": "/start",
        }
    }

    mock_orchestrator = AsyncMock()

    with patch.object(service, "send_message", mock_send):
        await service.process_incoming_update(update, mock_orchestrator)
        mock_send.assert_called_once()
        args, kwargs = mock_send.call_args
        assert kwargs["chat_id"] == 12345
        assert "Seja bem-vindo ao FactChkBR" in kwargs["text"]
        # /start não deve acionar o orquestrador pesado
        mock_orchestrator.analyze.assert_not_called()


@pytest.mark.asyncio
async def test_telegram_process_health_claim_update():
    """Valida processamento de alegação enviada pelo Telegram, com notificação inicial e entrega final."""
    custom_settings = Settings(TELEGRAM_BOT_TOKEN="fake_token_12345")
    service = TelegramService(settings=custom_settings)

    mock_send = AsyncMock()
    mock_send.return_value = AsyncMock(delivered=True, message_id=102)
    mock_action = AsyncMock(return_value=True)

    fake_analysis = AnalyzeResponse(
        claim="Água com limão emagrece",
        verdict=Verdict.FAKE,
        confidence=0.85,
        summary="Afirmação desprovida de comprovação científica.",
        reasons=["Sem base clínica."],
        sources=["Anvisa"],
    )
    mock_orchestrator = AsyncMock()
    mock_orchestrator.analyze.return_value = fake_analysis

    update = {
        "update_id": 2,
        "message": {
            "message_id": 88,
            "chat": {"id": 98765, "type": "private"},
            "text": "Água com limão emagrece",
        }
    }

    with patch.object(service, "send_message", mock_send), \
         patch.object(service, "send_chat_action", mock_action):
        await service.process_incoming_update(update, mock_orchestrator)

        # 1. Deve ter enviado a ação typing
        mock_action.assert_called_with(chat_id=98765, action="typing")
        # 2. Deve ter chamado o orchestrator com o texto da mensagem
        mock_orchestrator.analyze.assert_called_once()
        # 3. Deve ter chamado send_message pelo menos duas vezes (ack de processamento + resultado final)
        assert mock_send.call_count == 2
        # A última mensagem é o veredito formatado
        last_call_text = mock_send.call_args_list[-1][1]["text"]
        assert "FACTCHKBR - VERIFICAÇÃO FACTUAL" in last_call_text
        assert "FAKE / BOATO" in last_call_text


def test_telegram_webhook_endpoint_security():
    """Testa validação de secret token configurado no webhook."""
    from app.config import get_settings
    custom_settings = Settings(
        TELEGRAM_BOT_TOKEN="fake_token",
        TELEGRAM_WEBHOOK_SECRET="super_secret_token_123",
    )
    app.dependency_overrides[get_settings] = lambda: custom_settings

    try:
        update_payload = {
            "update_id": 3,
            "message": {
                "message_id": 99,
                "chat": {"id": 111, "type": "private"},
                "text": "Teste",
            }
        }

        # 1. Sem o header correto: deve retornar 403 Forbidden
        bad_resp = client.post(
            "/api/telegram/webhook",
            json=update_payload,
            headers={"X-Telegram-Bot-Api-Secret-Token": "wrong_secret"}
        )
        assert bad_resp.status_code == 403

        # 2. Com o header correto: deve aceitar com 200 OK
        ok_resp = client.post(
            "/api/telegram/webhook",
            json=update_payload,
            headers={"X-Telegram-Bot-Api-Secret-Token": "super_secret_token_123"}
        )
        assert ok_resp.status_code == 200
        assert ok_resp.json() == {"ok": True}
    finally:
        app.dependency_overrides.pop(get_settings, None)


def test_analyze_endpoint_returns_telegram_formatted_text():
    """Valida se o endpoint padrão /api/analyze inclui o campo telegram_formatted_text."""
    payload = {
        "text": "O governo anunciou novo plano de incentivo fiscal.",
        "urls": [],
        "user_id": 123,
        "chat_id": 456,
        "deliver_to_telegram": False,
    }

    resp = client.post("/api/analyze", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert "telegram_formatted_text" in data
    assert data["telegram_formatted_text"] is not None
    assert "FACTCHKBR - VERIFICAÇÃO FACTUAL" in data["telegram_formatted_text"]
    assert "Veredito:" in data["telegram_formatted_text"]
