import pytest
from unittest.mock import patch
from app.analyzers.claim_extractor import LLMClaimDecomposer, infer_source_types
from app.schemas.claim_extraction import (
    AtomicAssertion,
    ClaimCategory,
    KnowledgeTriple,
    VerificationSourceType,
)


async def _mock_decompose(self, cleaned_text: str, entities: dict[str, list[str]]) -> tuple[str | None, list[AtomicAssertion]]:
    """Mock offline e determinístico para testes unitários sem dependência do Ollama."""
    lower = cleaned_text.lower()
    if any(k in lower for k in ("bom dia", "eu acho", "vergonha", "absurdo")):
        return None, []

    # Períodos compostos (ex.: 'porque')
    if "porque" in lower:
        assertions = [
            AtomicAssertion(
                id=1,
                statement="O Ministério da Saúde cancelou a compra dos remédios.",
                category=ClaimCategory.FACTUAL_CLAIM,
                triple=KnowledgeTriple(
                    subject="Ministério da Saúde",
                    predicate="cancelou",
                    object="compra dos remédios",
                ),
                is_check_worthy=True,
                suggested_source_types=[VerificationSourceType.AGENCIA_REGULADORA],
            ),
            AtomicAssertion(
                id=2,
                statement="O laboratório farmacêutico fraudou os testes clínicos.",
                category=ClaimCategory.FACTUAL_CLAIM,
                triple=KnowledgeTriple(
                    subject="laboratório farmacêutico",
                    predicate="fraudou",
                    object="testes clínicos",
                ),
                is_check_worthy=True,
                suggested_source_types=[VerificationSourceType.AGENCIA_CHECAGEM],
            ),
        ]
        return assertions[0].statement, assertions

    sources = infer_source_types(cleaned_text, entities)
    subject = "Sujeito"
    predicate = "afirma"
    obj = cleaned_text
    stmt = cleaned_text

    if "médico" in lower or "vermes" in lower:
        subject = "Médico"
        predicate = "extrai"
        obj = "vermes do coração de uma pessoa que come carne de porco"
        stmt = "Médico extrai vermes do coração de uma pessoa que come carne de porco"
    elif "anvisa" in lower:
        subject = "Anvisa"
        predicate = "determinou" if "determinou" in lower else "proibiu"
        obj = "suspensão do lote de azeite adulterado"
        stmt = (
            "A Anvisa determinou a suspensão do lote de azeite adulterado em São Paulo."
            if "determinou" in lower
            else "A Anvisa proibiu a venda de lote de azeite adulterado no país."
        )
        if VerificationSourceType.AGENCIA_REGULADORA not in sources:
            sources.append(VerificationSourceType.AGENCIA_REGULADORA)
    elif "combustível" in lower or "governo federal" in lower:
        subject = "governo federal"
        predicate = "aprovou"
        obj = "aumento de 20% no combustível"
        stmt = "O governo federal aprovou aumento de 20% no combustível."
    elif "avião" in lower:
        subject = "avião monomotor"
        predicate = "colidiu"
        obj = "torre de transmissão"
    elif "empresa aérea" in lower:
        subject = "empresa aérea"
        predicate = "faliu"
        obj = "dívida bilionária no exterior"
    elif "bens" in lower:
        subject = "bens do empresário"
        predicate = "foram confiscados"
        obj = "Receita Federal"
    elif "petrobras" in lower:
        subject = "Petrobras"
        predicate = "aumentará"
        obj = "valor do diesel"
    elif "ministro" in lower:
        subject = "ministro"
        predicate = "acabou de suspender"
        obj = "pagamentos"

    assertions = [
        AtomicAssertion(
            id=1,
            statement=stmt,
            category=ClaimCategory.FACTUAL_CLAIM,
            triple=KnowledgeTriple(subject=subject, predicate=predicate, object=obj),
            is_check_worthy=True,
            suggested_source_types=sources,
        )
    ]
    return stmt, assertions


@pytest.fixture(autouse=True)
def mock_llm_decomposer():
    """Autouse fixture para isolar a suíte de testes de conexões de rede locais ou externas."""
    with patch.object(LLMClaimDecomposer, "decompose", new=_mock_decompose):
        yield
