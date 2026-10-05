from enum import Enum
from typing import Any
from pydantic import BaseModel, Field, field_validator


class Verdict(str, Enum):
    """Vereditos padronizados aceitos pela API."""
    VERDADEIRO = "VERDADEIRO"
    FAKE = "FAKE"
    SUSPEITO = "SUSPEITO"
    INCONCLUSIVO = "INCONCLUSIVO"


class AnalyzeRequest(BaseModel):
    """
    Payload de entrada enviado pelo Bot do Telegram para análise.
    """
    text: str = Field(
        ...,
        description="Texto completo da mensagem ou fato a ser verificado",
        min_length=1
    )
    urls: list[str] = Field(
        default_factory=list,
        description="URLs opcionais associadas"
    )
    user_id: int | None = Field(
        default=None,
        description="ID do usuário no Telegram"
    )
    chat_id: int | None = Field(
        default=None,
        description="ID do chat no Telegram"
    )


class SubClaimAnalysis(BaseModel):
    """Análise e checagem isolada de uma proposição factual específica."""
    statement: str = Field(..., description="Texto da proposição factual individual")
    verdict: Verdict = Field(..., description="Veredito específico desta proposição: VERDADEIRO, FAKE, SUSPEITO, INCONCLUSIVO")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Nível de confiança da análise desta proposição")
    justification: str = Field(..., description="Justificativa sucinta detalhando por que a proposição é verdadeira ou falsa")
    sources: list[str] = Field(default_factory=list, description="Fontes diretas consultadas para esta proposição")

    @field_validator("confidence", mode="before")
    @classmethod
    def normalize_confidence(cls, v: Any) -> float:
        if v is None:
            return 0.50
        try:
            val = float(v)
            if val > 1.0:
                val = val / 100.0
            return max(0.0, min(1.0, val))
        except (ValueError, TypeError):
            return 0.50


class AnalyzeResponse(BaseModel):
    """
    Payload de saída retornado para o Bot do Telegram.
    100% compatível com a especificação de produção.
    """
    claim: str = Field(
        ...,
        description="Resumo objetivo da afirmação/fato verificado"
    )
    verdict: Verdict = Field(
        ...,
        description="Veredito: VERDADEIRO | FAKE | SUSPEITO | INCONCLUSIVO"
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Nível de confiança entre 0.0 e 1.0"
    )
    summary: str = Field(
        ...,
        description="Resumo contextualizado da análise realizada pelos modelos"
    )
    reasons: list[str] = Field(
        default_factory=list,
        description="Motivos apontados pelos modelos"
    )
    sources: list[str] = Field(
        default_factory=list,
        description="Fontes ou referências consultadas"
    )
    sub_claims: list[SubClaimAnalysis] = Field(
        default_factory=list,
        description="Checagem discriminada por proposição atômica identificada no texto"
    )

    @field_validator("confidence", mode="before")
    @classmethod
    def normalize_confidence(cls, v: Any) -> float:
        if v is None:
            return 0.50
        try:
            val = float(v)
            if val > 1.0:
                val = val / 100.0
            return max(0.0, min(1.0, val))
        except (ValueError, TypeError):
            return 0.50


class AnalyzerResult(BaseModel):
    """
    Resultado individual produzido por um analisador antes do ensemble.
    O campo 'verdict' é opcional para acomodar analisadores de apoio/features (ex.: HeuristicAnalyzer).
    """
    analyzer_name: str
    verdict: Verdict | None = Field(
        default=None,
        description="Veredito parcial (ou None para analisadores de suporte/extração de features)"
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Grau de confiança da decisão (0.0 quando não emite veredito)"
    )
    reasons: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    claim: str | None = None
    summary: str | None = None
    raw_details: dict[str, Any] | None = None

    @field_validator("confidence", mode="before")
    @classmethod
    def normalize_confidence(cls, v: Any) -> float:
        if v is None:
            return 0.0
        try:
            val = float(v)
            if val > 1.0:
                val = val / 100.0
            return max(0.0, min(1.0, val))
        except (ValueError, TypeError):
            return 0.0


class HealthResponse(BaseModel):
    """Payload de status do sistema e verificadores ativos."""
    status: str
    version: str
    active_analyzers: list[str]
    available_analyzers: list[str]
