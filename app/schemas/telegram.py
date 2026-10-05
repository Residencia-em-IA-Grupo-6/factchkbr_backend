"""
Schemas Pydantic para modelagem de payloads e updates da Telegram Bot API.
Em conformidade com a especificação oficial do Telegram:
https://core.telegram.org/bots/api#update
"""

from typing import Any
from pydantic import BaseModel, Field


class TelegramUser(BaseModel):
    id: int
    is_bot: bool = False
    first_name: str = ""
    last_name: str | None = None
    username: str | None = None


class TelegramChat(BaseModel):
    id: int
    type: str = "private"
    title: str | None = None
    username: str | None = None
    first_name: str | None = None


class TelegramMessage(BaseModel):
    message_id: int
    from_user: TelegramUser | None = Field(default=None, alias="from")
    chat: TelegramChat
    date: int = 0
    text: str | None = None
    caption: str | None = None
    reply_to_message: Any | None = None

    model_config = {"populate_by_name": True}


class TelegramUpdate(BaseModel):
    """Representa um Update emitido via Webhook pela Telegram Bot API."""
    update_id: int
    message: TelegramMessage | None = None
    edited_message: TelegramMessage | None = None
    channel_post: TelegramMessage | None = None

    def get_effective_message(self) -> TelegramMessage | None:
        """Retorna a mensagem ativa no update (seja nova, editada ou de canal)."""
        return self.message or self.edited_message or self.channel_post

    def get_effective_text(self) -> str | None:
        """Extrai o texto ou legenda presente no update."""
        msg = self.get_effective_message()
        if not msg:
            return None
        return msg.text or msg.caption


class TelegramSendMessageRequest(BaseModel):
    """Requisição direta para envio de mensagem a um chat do Telegram."""
    chat_id: int | str = Field(..., description="ID numérico ou @username do chat/canal destino")
    text: str = Field(..., description="Texto da mensagem (suporta HTML)")
    parse_mode: str = Field(default="HTML", description="Modo de formatação ('HTML' ou 'MarkdownV2')")
    reply_to_message_id: int | None = Field(default=None, description="ID da mensagem de referência para resposta")
    disable_web_page_preview: bool = Field(default=True, description="Desativa pré-visualização de links para mensagens mais compactas")


class TelegramSetWebhookRequest(BaseModel):
    """Requisição para configuração do Webhook junto aos servidores do Telegram."""
    url: str = Field(..., description="URL pública HTTPS onde a API receberá os updates (ex: https://meudominio.com/api/telegram/webhook)")
    secret_token: str | None = Field(default=None, description="Token secreto para validação do header X-Telegram-Bot-Api-Secret-Token")
    drop_pending_updates: bool = Field(default=False, description="Descarta mensagens acumuladas anteriores à configuração")
