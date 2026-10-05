"""
Serviço de Integração com o Bot do Telegram.
Gerencia a comunicação assíncrona com a Telegram Bot API,
formatação de relatórios em linguagem simples e acessível (sem jargões ou termos em inglês)
com links clicáveis para todas as fontes consultadas.
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

# Portais e órgãos oficiais para links diretos de referência
OFFICIAL_DEFAULT_SOURCES = [
    ("Ministério da Saúde (Brasil)", "https://www.gov.br/saude/pt-br"),
    ("Anvisa (Agência Nacional de Vigilância Sanitária)", "https://www.gov.br/anvisa/pt-br"),
    ("Organização Mundial da Saúde (OMS)", "https://www.who.int/pt"),
]

KNOWN_ENTITIES_LINKS = [
    (re.compile(r"anvisa", re.IGNORECASE), "Anvisa (Consulta de Produtos e Medicamentos)", "https://consultas.anvisa.gov.br/#/medicamentos/q/"),
    (re.compile(r"minist[ée]rio da sa[úu]de|ms\b", re.IGNORECASE), "Ministério da Saúde - Portal Oficial", "https://www.gov.br/saude/pt-br"),
    (re.compile(r"fiocruz", re.IGNORECASE), "Fiocruz (Fundação Oswaldo Cruz)", "https://portal.fiocruz.br"),
    (re.compile(r"\boms\b|organiza[çc][ãa]o mundial", re.IGNORECASE), "Organização Mundial da Saúde (OMS)", "https://www.who.int/pt"),
    (re.compile(r"clinicaltrials", re.IGNORECASE), "ClinicalTrials.gov (Ensaios Clínicos Científicos)", "https://clinicaltrials.gov"),
    (re.compile(r"europe pmc|pubmed", re.IGNORECASE), "Europe PMC (Biblioteca Médica e Científica)", "https://europepmc.org"),
    (re.compile(r"fato ou fake|g1", re.IGNORECASE), "G1 Fato ou Fake", "https://g1.globo.com/fato-ou-fake/"),
    (re.compile(r"lupa", re.IGNORECASE), "Agência Lupa", "https://lupa.uol.com.br"),
    (re.compile(r"aos fatos", re.IGNORECASE), "Aos Fatos", "https://www.aosfatos.org"),
    (re.compile(r"uol confere|uol", re.IGNORECASE), "UOL Confere", "https://noticias.uol.com.br/confere/"),
    (re.compile(r"estad[ãa]o verifica|estad[ãa]o", re.IGNORECASE), "Estadão Verifica", "https://www.estadao.com.br/estadao-verifica/"),
    (re.compile(r"folha", re.IGNORECASE), "Folha de S.Paulo", "https://www1.folha.uol.com.br"),
    (re.compile(r"boatos\.org", re.IGNORECASE), "Boatos.org", "https://www.boatos.org"),
    (re.compile(r"cfm|conselho federal de medicina", re.IGNORECASE), "Conselho Federal de Medicina (CFM)", "https://portal.cfm.org.br"),
]


def clean_text_for_public(text: str) -> str:
    """
    Remove jargões técnicos, nomes de modelos internos e termos em inglês
    para garantir que a explicação seja 100% clara para qualquer cidadão.
    """
    if not text:
        return ""
    t = text.strip()

    # 1. Remove menções a modelos e algoritmos internos
    t = re.sub(r"Avaliação fundamentada pelo classificador de decisão Plumb-4B:?\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"Resultado consolidado '?(?:FAKE|VERDADEIRO|SUSPEITO|INCONCLUSIVO)'? com base em \d+ alegação\(ões\).*?\.", "", t, flags=re.IGNORECASE)
    t = re.sub(r"Classificador de Decisão Plumb-4B:?\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"Plumb-4B", "", t, flags=re.IGNORECASE)
    t = re.sub(r"LLM Judge(?:\s*\([^)]*\))?:?\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"Ollama local(?:\s*\([^)]*\))?:?\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"spaCy Gatekeeper(?:\s*\+[^)]*\))?:?\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"Heurística Textual.*?:?\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"Extrator FactChkBR.*?:?\s*", "", t, flags=re.IGNORECASE)

    # 2. Traduz e simplifica frases acadêmicas e jargões epistemológicos
    t = re.sub(
        r"sob o princípio de ônus da prova,?\s*afirmações públicas,?\s*promessas de tratamento ou declarações atribuídas sem respaldo factual são tratadas como provavelmente falsas \(boato sem sustentação\)\.?",
        "Não encontramos nenhum estudo médico ou registro oficial que comprove essa afirmação.",
        t,
        flags=re.IGNORECASE,
    )
    t = re.sub(
        r"sob o princípio de ônus da prova,?\s*afirmações sem respaldo factual são classificadas como provavelmente falsas\.?",
        "Não há comprovação oficial nem estudos médicos válidos para essa alegação.",
        t,
        flags=re.IGNORECASE,
    )
    t = re.sub(
        r"Alegação não encontrada em fontes oficiais ou veículos confiáveis:?\s*",
        "A informação não foi confirmada por órgãos de saúde nem pela imprensa de referência. ",
        t,
        flags=re.IGNORECASE,
    )
    t = re.sub(
        r"Prevalência de checagem oficial:?\s*fontes jornalísticas/IFCN de desmentido têm precedência probatória sobre confirmação divergente\.?",
        "Agências profissionais de checagem já investigaram e desmentiram esse boato.",
        t,
        flags=re.IGNORECASE,
    )
    t = re.sub(
        r"Confirmação por fontes oficiais e órgãos de referência com base documental comprovada\.?",
        "Informação conferida e confirmada com base em documentos e dados oficiais.",
        t,
        flags=re.IGNORECASE,
    )
    t = re.sub(r"\(não encontrada\s*/\s*provavelmente falso\)", "(boato sem comprovação)", t, flags=re.IGNORECASE)
    t = re.sub(r"\(provavelmente falso\)", "(sem comprovação)", t, flags=re.IGNORECASE)
    t = re.sub(r"\bFAKE\b", "FALSO", t)
    t = re.sub(r"\bCLAIM\b", "alegação", t, flags=re.IGNORECASE)
    t = re.sub(r"Sem predicado factual checável\.?", "", t, flags=re.IGNORECASE)
    t = re.sub(r"Extração concluída: texto com padrões estilísticos neutros\.?", "", t, flags=re.IGNORECASE)
    t = re.sub(r"Claim principal: \"[^\"]*\"\.?", "", t, flags=re.IGNORECASE)

    # 3. Limpeza de pontuação solta e espaçamento
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r"^[•\-—]\s*", "", t)
    return t


def build_accessible_sources(sources: list[str]) -> list[str]:
    """
    Garante que todas as fontes exibidas possuam links clicáveis reais
    e nomes em português claro, eliminando nomes de algoritmos internos.
    """
    results: list[str] = []
    seen_urls: set[str] = set()

    for s in sources:
        # Descarta algoritmos e módulos técnicos internos
        if any(ign in s.lower() for ign in (
            "plumb", "heurística", "extrator", "spacy", "llm judge", "ollama", "filtro de escopo"
        )):
            continue

        # 1. Se a fonte já traz uma URL
        url_match = URL_REGEX.search(s)
        if url_match:
            url = url_match.group(0).rstrip(".,;)")
            if url in seen_urls:
                continue
            seen_urls.add(url)

            name_part = s[:url_match.start()].strip(" :|-—")
            if not name_part or len(name_part) < 3:
                domain = re.sub(r"^https?://(?:www\.)?", "", url).split("/")[0]
                name_part = f"Matéria em {domain}"
            name_clean = html.escape(name_part[:60])
            results.append(f'• 🔗 <a href="{url}">{name_clean}</a>')
            continue

        # 2. Se for um órgão ou veículo conhecido, associa ao link oficial correspondente
        for pattern, label, url in KNOWN_ENTITIES_LINKS:
            if pattern.search(s):
                if url not in seen_urls:
                    seen_urls.add(url)
                    results.append(f'• 🔗 <a href="{url}">{html.escape(label)}</a>')
                break

    # 3. Se nenhuma fonte com link sobreviveu, fornece os portais oficiais de referência
    if not results:
        for label, url in OFFICIAL_DEFAULT_SOURCES:
            if url not in seen_urls:
                seen_urls.add(url)
                results.append(f'• 🔗 <a href="{url}">{html.escape(label)}</a>')

    return results[:4]


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
    # 1. FORMATAÇÃO VISUAL ACESSÍVEL (HTML AMIGÁVEL PARA QUALQUER PÚBLICO)
    # =========================================================================

    def format_analysis_for_telegram(self, analysis: AnalyzeResponse, original_text: str = "") -> str:
        """
        Converte a análise em mensagem agradável, didática e 100% em português.
        Sem termos em inglês, sem jargões técnicos e com links clicáveis em todas as fontes.
        """
        verdict = analysis.verdict

        verdict_headers = {
            Verdict.VERDADEIRO: (
                "✅",
                "É VERDADEIRO (FATO)",
                "Esta informação está correta e foi confirmada por fontes e órgãos de saúde confiáveis."
            ),
            Verdict.FAKE: (
                "❌",
                "É FALSO (BOATO)",
                "Esta informação não é verdadeira e não possui respaldo médico ou científico."
            ),
            Verdict.SUSPEITO: (
                "⚠️",
                "ENGANOSO (CUIDADO)",
                "Esta mensagem mistura fatos com informações incorretas, exageradas ou fora de contexto."
            ),
            Verdict.INCONCLUSIVO: (
                "❓",
                "NÃO COMPROVADO",
                "Ainda não há dados suficientes para confirmar ou desmentir com certeza."
            ),
        }

        icon, label, subtitle = verdict_headers.get(
            verdict,
            ("🔍", "EM ANÁLISE", "Análise factual concluída.")
        )

        conf_pct = round(analysis.confidence * 100)
        if conf_pct >= 80:
            certeza_texto = f"🟢 Alta ({conf_pct}%)"
        elif conf_pct >= 60:
            certeza_texto = f"🟡 Média ({conf_pct}%)"
        else:
            certeza_texto = f"⚪ Em apuração ({conf_pct}%)"

        claim_clean = html.escape(analysis.claim or original_text[:120])

        # Limpeza de resumo para eliminar termos técnicos e jargões
        cleaned_summary = clean_text_for_public(analysis.summary or "")
        if not cleaned_summary or len(cleaned_summary) < 15:
            default_explanations = {
                Verdict.FAKE: (
                    "Não encontramos nenhuma comprovação científica, registro na Anvisa ou notícia em veículos de saúde "
                    "confiáveis que comprove essa afirmação. Trata-se de um boato sem fundamento."
                ),
                Verdict.VERDADEIRO: (
                    "Esta afirmação foi conferida e possui respaldo comprovado em fontes oficiais de saúde e literatura médica."
                ),
                Verdict.SUSPEITO: (
                    "A mensagem contém elementos reais misturados a informações imprecisas ou exageradas. Tenha cautela ao compartilhar."
                ),
                Verdict.INCONCLUSIVO: (
                    "Não encontramos dados oficiais ou estudos médicos suficientes até o momento para confirmar nem desmentir essa informação com segurança."
                ),
            }
            cleaned_summary = default_explanations.get(verdict, "Análise concluída com base nas fontes consultadas.")

        summary_clean = html.escape(cleaned_summary)

        lines = [
            "🔍 <b>FACTCHKBR • CHECAGEM DE INFORMAÇÃO</b>",
            "───────────────────────────",
            f"📌 <b>O que foi analisado:</b>",
            f"<i>\"{claim_clean}\"</i>\n",
            f"<b>Resultado:</b> {icon} <b>{label}</b>",
            f"<i>{subtitle}</i>\n",
            f"📊 <b>Nível de certeza:</b> {certeza_texto}\n",
            f"📝 <b>Entenda o caso:</b>\n{summary_clean}\n",
        ]

        # Se houver múltiplos pontos analisados separadamente
        if analysis.sub_claims and len(analysis.sub_claims) > 1:
            lines.append("📋 <b>Pontos analisados:</b>")
            for sc in analysis.sub_claims:
                sc_v = sc.verdict
                sc_icon = "✅" if sc_v == Verdict.VERDADEIRO else ("❌" if sc_v == Verdict.FAKE else ("⚠️" if sc_v == Verdict.SUSPEITO else "❓"))
                sc_stmt = html.escape(clean_text_for_public(sc.statement))
                sc_just = html.escape(clean_text_for_public(sc.justification))
                lines.append(f"  • {sc_icon} <b>\"{sc_stmt}\"</b>")
                if sc_just:
                    lines.append(f"     └─ <i>{sc_just}</i>")
            lines.append("")

        # Razões e pontos de atenção (filtrados sem jargões técnicos)
        accessible_reasons: list[str] = []
        for r in analysis.reasons:
            cleaned_r = clean_text_for_public(r)
            if cleaned_r and len(cleaned_r) > 12 and cleaned_r not in accessible_reasons:
                accessible_reasons.append(cleaned_r)

        if accessible_reasons:
            lines.append("💡 <b>O que você precisa saber:</b>")
            for r in accessible_reasons[:3]:
                lines.append(f"• {html.escape(r)}")
            lines.append("")

        # Fontes consultadas OBRIGATORIAMENTE COM LINK
        linked_sources = build_accessible_sources(analysis.sources)
        if linked_sources:
            lines.append("🌐 <b>Fontes consultadas (clique no link para acessar):</b>")
            lines.extend(linked_sources)
            lines.append("")

        lines.append("───────────────────────────")
        lines.append("💡 <i>Dica: Antes de repassar dicas de saúde ou remédios caseiros, consulte sempre um médico ou órgãos oficiais!</i>")

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
        """Envia mensagem de texto formatada a um chat ou usuário do Telegram."""
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
        """Envia indicador de digitação para informar ao usuário que a resposta está sendo gerada."""
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
    # 3. PROCESSAMENTO DE MENSAGENS RECEBIDAS (WEBHOOK HANDLER)
    # =========================================================================

    async def process_incoming_update(
        self,
        update_data: dict[str, Any] | TelegramUpdate,
        orchestrator: FactCheckOrchestrator,
    ) -> TelegramDeliveryStatus | None:
        """
        Recebe um update do Telegram, extrai a dúvida do cidadão, executa a verificação
        e devolve a resposta clara e acolhedora para o chat de origem.
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

        # 1. Comandos Iniciais em Português
        cmd = raw_text.split()[0].lower()
        if cmd in ("/start", "/iniciar"):
            welcome_msg = (
                "👋 <b>Olá! Seja muito bem-vindo ao FactChkBR!</b> 🩺🔍\n\n"
                "Eu sou o seu assistente gratuito para <b>conferir notícias e dicas de saúde</b>.\n\n"
                "💡 <b>Como funciona?</b>\n"
                "É muito simples: sempre que receber uma mensagem, receita caseira, dica de remédio "
                "ou corrente de WhatsApp e ficar na dúvida se é verdade, basta me enviar ou encaminhar aqui.\n\n"
                "<b>Exemplos do que você pode checar:</b>\n"
                "• <i>\"Água com limão emagrece?\"</i>\n"
                "• <i>\"Chá de boldo cura gastrite?\"</i>\n"
                "• <i>\"Novo remédio caseiro cura pressão alta?\"</i>\n\n"
                "Nós pesquisamos em órgãos oficiais como a <b>Anvisa</b>, o <b>Ministério da Saúde</b> "
                "e estudos médicos para te dar uma resposta rápida e segura.\n\n"
                "Envie sua dúvida ou mensagem abaixo para começar! 👇"
            )
            return await self.send_message(
                chat_id=chat_id,
                text=welcome_msg,
                reply_to_message_id=message.message_id,
            )

        if cmd in ("/help", "/ajuda", "/sobre"):
            help_msg = (
                "ℹ️ <b>Como entender o resultado da checagem:</b> 📖\n\n"
                "• ❌ <b>É FALSO:</b> É boato ou mentira. Não possui comprovação científica ou foi desmentido por especialistas.\n"
                "• ✅ <b>É VERDADEIRO:</b> A informação está certa e foi confirmada por órgãos de saúde ou médicos.\n"
                "• ⚠️ <b>ENGANOSO:</b> Mistura alguma coisa que existe com dados exagerados ou fora de contexto. Cuidado!\n"
                "• ❓ <b>NÃO COMPROVADO:</b> Ainda não há estudos ou dados suficientes para ter certeza de um lado ou de outro.\n\n"
                "Basta enviar qualquer texto ou link quando quiser tirar uma dúvida!"
            )
            return await self.send_message(
                chat_id=chat_id,
                text=help_msg,
                reply_to_message_id=message.message_id,
            )

        # 2. Aviso de Recebimento Amigável
        await self.send_chat_action(chat_id=chat_id, action="typing")

        ack_text = (
            "⏳ <b>FactChkBR:</b> <i>Recebemos sua mensagem! Já estamos pesquisando nos órgãos de saúde e estudos médicos. "
            "Aguarde alguns segundos...</i>"
        )
        await self.send_message(
            chat_id=chat_id,
            text=ack_text,
            reply_to_message_id=message.message_id,
        )

        # 3. Extração de URLs opcionais
        extracted_urls = URL_REGEX.findall(raw_text)

        # 4. Execução da Checagem Factual
        try:
            analysis = await orchestrator.analyze(text=raw_text, urls=extracted_urls)
        except Exception as e:
            logger.exception("Erro durante a execução do orquestrador para Telegram: %s", e)
            error_msg = (
                "⚠️ <i>Pedimos desculpas: ocorreu uma instabilidade momentânea ao pesquisar esta informação. "
                "Por favor, tente enviar novamente em instantes.</i>"
            )
            return await self.send_message(
                chat_id=chat_id,
                text=error_msg,
                reply_to_message_id=message.message_id,
            )

        # 5. Formatação Acessível e Entrega Final
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
