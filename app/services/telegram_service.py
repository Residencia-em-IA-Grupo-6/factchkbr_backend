"""
Serviço de Integração com o Bot do Telegram.
Gerencia a comunicação assíncrona com a Telegram Bot API,
formatação de relatórios em HTML enriquecido e processamento de updates/webhooks.
"""

import html
import logging
import re
from typing import Any
import httpx

from app.config import Settings, get_settings
from app.core.orchestrator import FactCheckOrchestrator
from app.schemas.analysis import AnalyzeResponse, TelegramDeliveryStatus, Verdict
from app.schemas.telegram import TelegramMessage, TelegramUpdate

logger = logging.getLogger("factchkbr.telegram")

URL_REGEX = re.compile(r"https?://(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b[-a-zA-Z0-9()@:%_\+.~#?&//=]*")


class TelegramService:
    """Gerencia recebimento, formatação e entrega de mensagens via Telegram Bot API."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._http_client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(timeout=15.0)
        return self._http_client

    async def aclose(self) -> None:
        """Encerra o cliente HTTP subjacente."""
        if self._http_client and not self._http_client.is_closed:
            await self._http_client.aclose()

    @property
    def bot_token(self) -> str | None:
        return self.settings.TELEGRAM_BOT_TOKEN

    @property
    def is_configured(self) -> bool:
        """Verifica se o token do bot foi configurado."""
        return bool(self.bot_token and len(self.bot_token.strip()) > 5)

    def _build_url(self, method: str) -> str:
        base = self.settings.TELEGRAM_API_BASE_URL.rstrip("/")
        token = (self.bot_token or "").strip()
        return f"{base}/bot{token}/{method}"

    # =========================================================================
    # 1. FORMATAÇÃO VISUAL ENRIQUECIDA (HTML PARA TELEGRAM)
    # =========================================================================

    def format_analysis_for_telegram(self, analysis: AnalyzeResponse, original_text: str = "") -> str:
        """
        Converte o contrato AnalyzeResponse em mensagem HTML limpa e informativa,
        otimizada para leitura no aplicativo do Telegram.
        """
        verdict = analysis.verdict
        verdict_str = verdict.value if hasattr(verdict, "value") else str(verdict)

        verdict_headers = {
            Verdict.VERDADEIRO: ("✅", "VERDADEIRO", "Informação com respaldo factual ou científico comprovado."),
            Verdict.FAKE: ("❌", "FAKE / BOATO", "Alegação falsa, desmentida ou sem respaldo oficial/científico."),
            Verdict.SUSPEITO: ("⚠️", "SUSPEITO / DISTORCIDO", "Conteúdo com partes verdadeiras misturadas a imprecisões ou fora de contexto."),
            Verdict.INCONCLUSIVO: ("❓", "INCONCLUSIVO", "Não há dados suficientes ou registros oficiais para confirmar ou refutar."),
        }

        icon, label, subtitle = verdict_headers.get(
            verdict,
            ("🔍", verdict_str, "Análise concluída.")
        )

        conf_pct = round(analysis.confidence * 100, 1)

        # Barra visual de confiança
        filled_bars = int(round(conf_pct / 10))
        conf_bar = "█" * filled_bars + "░" * (10 - filled_bars)

        claim_clean = html.escape(analysis.claim or original_text[:120])
        summary_clean = html.escape(analysis.summary or "")

        lines = [
            "⚖️ <b>FACTCHKBR - VERIFICAÇÃO FACTUAL</b>",
            "───────────────────────────",
            f"🎯 <b>Alegação:</b> <i>\"{claim_clean}\"</i>\n",
            f"<b>Veredito:</b> {icon} <b>{label}</b>",
            f"<i>{subtitle}</i>\n",
            f"📊 <b>Grau de Confiança:</b> <code>{conf_pct:.1f}%</code>",
            f"<code>[{conf_bar}]</code>\n",
            f"📝 <b>Síntese Analítica:</b>\n{summary_clean}\n",
        ]

        # Se houver discriminação de proposições atômicas
        if analysis.sub_claims and len(analysis.sub_claims) > 1:
            lines.append("🧬 <b>Detalhamento por Alegação:</b>")
            for idx, sc in enumerate(analysis.sub_claims, 1):
                sc_v = sc.verdict
                sc_icon = "✅" if sc_v == Verdict.VERDADEIRO else ("❌" if sc_v == Verdict.FAKE else ("⚠️" if sc_v == Verdict.SUSPEITO else "❓"))
                sc_stmt = html.escape(sc.statement)
                sc_just = html.escape(sc.justification)
                sc_c = round(sc.confidence * 100)
                lines.append(f"  {sc_icon} <b>[{idx}] {sc_stmt}</b> (<code>{sc_c}%</code>)")
                if sc_just:
                    lines.append(f"     └─ <i>{sc_just}</i>")
            lines.append("")

        # Razões e fundamentos principais (até 3 mais relevantes)
        if analysis.reasons:
            lines.append("🔍 <b>Principais Fundamentos:</b>")
            for r in analysis.reasons[:3]:
                r_clean = html.escape(r)
                lines.append(f"• {r_clean}")
            lines.append("")

        # Fontes consultadas
        if analysis.sources:
            lines.append("🌐 <b>Fontes Consultadas:</b>")
            for s in analysis.sources[:4]:
                s_clean = html.escape(s)
                if s.startswith("http://") or s.startswith("https://"):
                    lines.append(f"• <a href=\"{s_clean}\">{s_clean[:45]}...</a>")
                else:
                    lines.append(f"• {s_clean}")
            lines.append("")

        lines.append("───────────────────────────")
        lines.append("🤖 <i>FactChkBR • Verificação automatizada de desinformação e saúde pública.</i>")

        return "\n".join(lines)

    # =========================================================================
    # 2. ENTREGA DE MENSAGENS VIA TELEGRAM BOT API
    # =========================================================================

    async def send_message(
        self,
        chat_id: int | str,
        text: str,
        parse_mode: str = "HTML",
        reply_to_message_id: int | None = None,
        disable_web_page_preview: bool = True,
    ) -> TelegramDeliveryStatus:
        """
        Envia uma mensagem de texto formatada a um chat ou usuário do Telegram.
        """
        if not self.is_configured:
            logger.warning("Tentativa de envio ao Telegram sem TELEGRAM_BOT_TOKEN configurado.")
            return TelegramDeliveryStatus(
                delivered=False,
                error="TELEGRAM_BOT_TOKEN não configurado no backend."
            )

        client = self._get_client()
        url = self._build_url("sendMessage")
        payload: dict[str, Any] = {
            "chat_id": chat_id,
            "text": text,
            "parse_mode": parse_mode,
            "disable_web_page_preview": disable_web_page_preview,
        }
        if reply_to_message_id:
            payload["reply_to_message_id"] = reply_to_message_id

        try:
            resp = await client.post(url, json=payload)
            data = resp.json()
            if resp.status_code == 200 and data.get("ok"):
                msg_id = data.get("result", {}).get("message_id")
                logger.info("Mensagem entregue com sucesso ao Telegram (chat_id: %s, message_id: %s)", chat_id, msg_id)
                return TelegramDeliveryStatus(delivered=True, message_id=msg_id)

            err = data.get("description", f"Status HTTP {resp.status_code}")
            logger.error("Erro ao enviar mensagem ao Telegram: %s", err)
            return TelegramDeliveryStatus(delivered=False, error=err)
        except Exception as e:
            logger.exception("Falha de rede ao conectar à API do Telegram: %s", e)
            return TelegramDeliveryStatus(delivered=False, error=str(e))

    async def send_chat_action(self, chat_id: int | str, action: str = "typing") -> bool:
        """Envia indicador de status (ex: 'typing') para informar ao usuário que o bot está processando."""
        if not self.is_configured:
            return False

        client = self._get_client()
        url = self._build_url("sendChatAction")
        try:
            resp = await client.post(url, json={"chat_id": chat_id, "action": action})
            return resp.status_code == 200
        except Exception:
            return False

    async def set_webhook(
        self,
        url: str,
        secret_token: str | None = None,
        drop_pending_updates: bool = False,
    ) -> dict[str, Any]:
        """Configura a URL do webhook nos servidores do Telegram."""
        if not self.is_configured:
            return {"ok": False, "description": "TELEGRAM_BOT_TOKEN não configurado"}

        client = self._get_client()
        endpoint = self._build_url("setWebhook")
        payload: dict[str, Any] = {
            "url": url,
            "drop_pending_updates": drop_pending_updates,
        }
        if secret_token:
            payload["secret_token"] = secret_token
        elif self.settings.TELEGRAM_WEBHOOK_SECRET:
            payload["secret_token"] = self.settings.TELEGRAM_WEBHOOK_SECRET

        resp = await client.post(endpoint, json=payload)
        return resp.json()

    async def get_webhook_info(self) -> dict[str, Any]:
        """Consulta o status atual do webhook junto aos servidores do Telegram."""
        if not self.is_configured:
            return {"ok": False, "description": "TELEGRAM_BOT_TOKEN não configurado"}

        client = self._get_client()
        endpoint = self._build_url("getWebhookInfo")
        resp = await client.get(endpoint)
        return resp.json()

    async def delete_webhook(self, drop_pending_updates: bool = False) -> dict[str, Any]:
        """Remove o webhook configurado no Telegram."""
        if not self.is_configured:
            return {"ok": False, "description": "TELEGRAM_BOT_TOKEN não configurado"}

        client = self._get_client()
        endpoint = self._build_url("deleteWebhook")
        resp = await client.post(endpoint, json={"drop_pending_updates": drop_pending_updates})
        return resp.json()

    # =========================================================================
    # 3. PROCESSAMENTO DE ATUALIZAÇÕES RECEBIDAS (WEBHOOK HANDLER)
    # =========================================================================

    async def process_incoming_update(
        self,
        update_data: dict[str, Any] | TelegramUpdate,
        orchestrator: FactCheckOrchestrator,
    ) -> TelegramDeliveryStatus | None:
        """
        Recebe um update do Telegram, extrai o texto/comando, executa o pipeline
        e devolve a análise formatada para o chat de origem.
        """
        if isinstance(update_data, dict):
            try:
                update = TelegramUpdate.model_validate(update_data)
            except Exception as e:
                logger.warning("Update do Telegram inválido ou não suportado: %s", e)
                return None
        else:
            update = update_data

        message = update.get_effective_message()
        if not message:
            return None

        chat_id = message.chat.id
        raw_text = (message.text or message.caption or "").strip()

        if not raw_text:
            return None

        # 1. Comandos de Sistema (/start, /help, /ajuda, /sobre)
        cmd = raw_text.split()[0].lower()
        if cmd in ("/start", "/iniciar"):
            welcome_msg = (
                "👋 <b>Olá! Seja bem-vindo ao FactChkBR!</b> 🩺🔍\n\n"
                "Eu sou o assistente inteligente de verificação de fatos e combate à desinformação, "
                "especializado em <b>saúde pública, tratamentos, medicamentos e alegações biomédicas</b>.\n\n"
                "💡 <b>Como funciona?</b>\n"
                "Basta me enviar ou encaminhar qualquer texto, corrente de WhatsApp, notícia ou áudio transcrito "
                "(ex.: <i>\"Água com limão emagrece\"</i>, <i>\"Chá de boldo cura gastrite\"</i>).\n\n"
                "🛡️ <b>Nosso pipeline analisa:</b>\n"
                "• 🏛️ Registros oficiais da Anvisa (medicamentos ativos e cancelados)\n"
                "• 🧪 Ensaios clínicos e artigos científicos (ClinicalTrials e Europe PMC)\n"
                "• 📰 Agências profissionais de checagem de fatos (IFCN)\n"
                "• 🧠 Modelos de tomada de decisão e linguagem natural\n\n"
                "Envie sua dúvida ou mensagem abaixo para começar! 👇"
            )
            return await self.send_message(
                chat_id=chat_id,
                text=welcome_msg,
                reply_to_message_id=message.message_id,
            )

        if cmd in ("/help", "/ajuda", "/sobre"):
            help_msg = (
                "ℹ️ <b>Guia de Uso do FactChkBR</b> 📖\n\n"
                "<b>Como interpretar os vereditos:</b>\n"
                "• ✅ <b>VERDADEIRO:</b> Informação confirmada por órgãos oficiais ou literatura médica.\n"
                "• ❌ <b>FAKE:</b> Boato, desinformação, promessa sem respaldo ou produto irregular.\n"
                "• ⚠️ <b>SUSPEITO:</b> Dados fora de contexto, exagero ou mistura de meias-verdades.\n"
                "• ❓ <b>INCONCLUSIVO:</b> Assunto recente ou escassez de dados confiáveis para cravar um lado.\n\n"
                "Envie qualquer texto ou link para realizarmos a checagem imediata."
            )
            return await self.send_message(
                chat_id=chat_id,
                text=help_msg,
                reply_to_message_id=message.message_id,
            )

        # 2. Notificação Proativa de Início do Processamento
        await self.send_chat_action(chat_id=chat_id, action="typing")

        ack_text = "⏳ <b>FactChkBR:</b> <i>Recebemos sua mensagem e iniciamos a checagem factual. Aguarde alguns instantes...</i>"
        await self.send_message(
            chat_id=chat_id,
            text=ack_text,
            reply_to_message_id=message.message_id,
        )

        # 3. Extração de URLs opcionais presentes no texto da mensagem
        extracted_urls = URL_REGEX.findall(raw_text)

        # 4. Execução do Pipeline de Checagem Factual
        try:
            analysis = await orchestrator.analyze(text=raw_text, urls=extracted_urls)
        except Exception as e:
            logger.exception("Erro durante a execução do orquestrador para Telegram: %s", e)
            error_msg = (
                "⚠️ <i>Desculpe, ocorreu uma instabilidade temporária ao processar esta mensagem. "
                "Por favor, tente novamente em instantes.</i>"
            )
            return await self.send_message(
                chat_id=chat_id,
                text=error_msg,
                reply_to_message_id=message.message_id,
            )

        # 5. Formatação e Entrega da Resposta Final
        formatted_html = self.format_analysis_for_telegram(analysis, original_text=raw_text)
        return await self.send_message(
            chat_id=chat_id,
            text=formatted_html,
            reply_to_message_id=message.message_id,
        )


# Singleton
_telegram_service: TelegramService | None = None


def get_telegram_service() -> TelegramService:
    """Retorna a instância singleton do serviço do Telegram."""
    global _telegram_service
    if _telegram_service is None:
        _telegram_service = TelegramService()
    return _telegram_service
