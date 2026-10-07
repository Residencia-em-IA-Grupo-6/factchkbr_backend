from typing import Any
from pydantic import BaseModel, Field, field_validator
from app.schemas.analysis import Verdict


class VectorClaimItem(BaseModel):
    """Representação de uma alegação indexada no banco de dados vetorial."""
    id: str = Field(..., description="Identificador único (hash determinístico ou UUID)")
    statement: str = Field(..., description="Texto canônico da alegação checada")
    verdict: Verdict = Field(..., description="Veredito: VERDADEIRO, FAKE, SUSPEITO, INCONCLUSIVO")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Grau de certeza atribuído à checagem")
    summary: str = Field(default="", description="Resumo explicativo do fact-checking")
    reasons: list[str] = Field(default_factory=list, description="Fundamentações e justificativas")
    sources: list[str] = Field(default_factory=list, description="Links e fontes oficiais que embasam a checagem")
    created_at: str | None = Field(default=None, description="Data/hora da indexação")
    similarity: float | None = Field(default=None, description="Grau de similaridade semântica (0.0 a 1.0)")
    distance: float | None = Field(default=None, description="Distância vetorial (cosseno)")
    category: str | None = Field(default=None, description="Categoria temática (ex: PUBLIC_HEALTH)")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Metadados adicionais brutos")

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


class VectorSearchRequest(BaseModel):
    """Parâmetros de busca por similaridade semântica de alegações."""
    query: str = Field(..., min_length=1, description="Texto ou alegação para busca vetorial")
    limit: int = Field(default=3, ge=1, le=20, description="Número máximo de resultados mais similares")
    min_similarity: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Limiar mínimo de similaridade cosseno (ex: 0.70)"
    )
    verdict_filter: Verdict | None = Field(
        default=None,
        description="Filtro opcional para retornar apenas alegações com este veredito"
    )


class VectorSearchResponse(BaseModel):
    """Resultados de busca semântica no banco vetorial."""
    query: str
    total_found: int
    results: list[VectorClaimItem]


class VectorIndexRequest(BaseModel):
    """Payload para indexar manualmente uma nova alegação com suas fundamentações."""
    statement: str = Field(..., min_length=3, description="Texto da alegação a ser indexada")
    verdict: Verdict = Field(..., description="Veredito: VERDADEIRO, FAKE, SUSPEITO, INCONCLUSIVO")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confiança da checagem")
    summary: str = Field(default="", description="Resumo da checagem")
    reasons: list[str] = Field(default_factory=list, description="Lista de fundamentações")
    sources: list[str] = Field(default_factory=list, description="Lista de fontes com links ou referências")
    category: str | None = Field(default=None, description="Categoria opcional")


class VectorIndexResponse(BaseModel):
    """Resposta após indexação de uma alegação."""
    id: str
    status: str = "indexed"
    statement: str
    verdict: Verdict


class VectorStatsResponse(BaseModel):
    """Estatísticas do banco de dados vetorial ChromaDB."""
    total_claims: int
    collection_name: str
    persist_directory: str
    enabled: bool
