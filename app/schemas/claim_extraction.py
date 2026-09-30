from enum import Enum
from pydantic import BaseModel, Field


class VerificationSourceType(str, Enum):
    """Categorias de fontes oficiais e públicas recomendadas para a checagem."""
    ORGAO_OFICIAL = "ORGAO_OFICIAL"          # Diário Oficial, Ministérios, Governo
    AGENCIA_REGULADORA = "AGENCIA_REGULADORA" # Anvisa, Anatel, Bacen, Aneel
    PODER_JUDICIARIO = "PODER_JUDICIARIO"    # STF, TSE, STJ, CNJ
    INSTITUTO_PESQUISA = "INSTITUTO_PESQUISA" # IBGE, Fiocruz, Ipea, INPE
    AGENCIA_CHECAGEM = "AGENCIA_CHECAGEM"    # Lupa, Aos Fatos, Fato ou Fake, Boatos.org
    DADOS_PUBLICOS = "DADOS_PUBLICOS"        # Portal da Transparência, Receita Federal, INSS


class KnowledgeTriple(BaseModel):
    """Tripla de Conhecimento (SPO) que formaliza a relação factual atômica."""
    subject: str = Field(..., description="Entidade ou sujeito principal que realiza/sofre a ação")
    predicate: str = Field(..., description="Ação, verbo ou relação assertiva (em ordem direta)")
    object: str = Field(..., description="Alvo, efeito, número ou objeto da asserção")


class AtomicAssertion(BaseModel):
    """Proposição atômica independente (fato único checável derivado do período)."""
    id: int = Field(..., description="Identificador sequencial da asserção")
    statement: str = Field(..., description="Fato normalizado em ordem direta e linguagem denotativa neutra")
    triple: KnowledgeTriple = Field(..., description="Tripla semântica sujeito-predicado-objeto")
    suggested_source_types: list[VerificationSourceType] = Field(
        default_factory=list,
        description="Tipos de fontes recomendadas para averiguar esta asserção"
    )
    is_check_worthy: bool = Field(
        default=True,
        description="Indica se a afirmação possui teor factual concreto e checável"
    )


class ClaimExtractionContract(BaseModel):
    """
    Contrato estrito de retorno da camada de extração e decomposição.
    """
    original_text: str = Field(..., description="Texto bruto de entrada recebido da rede social")
    cleaned_text: str = Field(..., description="Texto limpo e sanitizado sem sirenes, emojis ou apelos")
    entities: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Entidades detectadas agrupadas por categoria (PER, ORG, LOC, MISC)"
    )
    assertions: list[AtomicAssertion] = Field(
        default_factory=list,
        description="Lista de proposições atômicas desmembradas a partir do texto"
    )
    discarded_fragments: list[str] = Field(
        default_factory=list,
        description="Fragmentos descartados por serem apelos à ação, saudações ou pura opinião"
    )
    engine_used: str = Field(
        default="syntactic_fallback",
        description="Identificador do motor que gerou a decomposição (ex: 'llm:phi3.5' ou 'syntactic_fallback')"
    )

