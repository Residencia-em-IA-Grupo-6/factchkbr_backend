import pytest
from app.analyzers.fact_check_api import (
    SourceTier,
    get_source_tier,
    check_evidence_relevance,
    EvidenceItem,
    FactCheckApiAnalyzer,
)
from app.core.orchestrator import FactCheckOrchestrator
from app.schemas.analysis import AnalyzerResult, Verdict, SubClaimAnalysis


def test_source_tiering():
    """Valida a categorização correta por tiers de credibilidade institucional."""
    assert get_source_tier("Agência Lupa", "https://piaui.folha.uol.com.br/lupa") == SourceTier.TIER1_OFFICIAL_OR_IFCN
    assert get_source_tier("Aos Fatos", "https://www.aosfatos.org/noticias/...") == SourceTier.TIER1_OFFICIAL_OR_IFCN
    assert get_source_tier("Ministério da Saúde", "https://www.gov.br/saude") == SourceTier.TIER1_OFFICIAL_OR_IFCN
    assert get_source_tier("G1", "https://g1.globo.com") == SourceTier.TIER2_MAINSTREAM_MEDIA
    assert get_source_tier("Folha", "https://folha.uol.com.br") == SourceTier.TIER2_MAINSTREAM_MEDIA
    assert get_source_tier("Blog Desconhecido", "https://blogaleatorio123.com") == SourceTier.UNKNOWN


def test_evidence_relevance_filtering():
    """Valida o descarte de matérias sem correlação semântica substantiva com a alegação."""
    claim = "STF aprovou o marco temporal de terras indígenas"
    # Matéria relevante
    assert check_evidence_relevance(claim, "STF retoma julgamento do marco temporal de terras indígenas") is True
    # Matérias irrelevantes (esporte, culinária, etc.)
    assert check_evidence_relevance(claim, "Flamengo vence o Vasco com gol no final do clássico") is False
    assert check_evidence_relevance(claim, "Receita de bolo de cenoura com cobertura de chocolate") is False


def test_polarity_stance_detection():
    """Valida detecção de posicionamento sensível à polaridade e negações."""
    analyzer = FactCheckApiAnalyzer()

    # 1. Alegação afirmativa confrontada com desmentido oficial do boato
    ev_aff = EvidenceItem(
        title="É falso que vacinas causam autismo em crianças",
        source_name="Aos Fatos",
        url="https://aosfatos.org/noticias/vacina-autismo",
        snippet="Estudos comprovam que não existe relação entre vacina e autismo",
        rating="Falso",
        is_fact_check=True,
        source_tier=SourceTier.TIER1_OFFICIAL_OR_IFCN.value,
    )
    v_aff, conf_aff, _ = analyzer.evaluate_verdict([ev_aff], claim="Vacinas causam autismo")
    assert ev_aff.stance == "REFUTES"
    assert v_aff == Verdict.FAKE

    # 2. Alegação negativa confrontada com matéria de desmentido ao boato original
    ev_neg = EvidenceItem(
        title="É mentira que vacinas provocam autismo",
        source_name="Agência Lupa",
        url="https://lupa.uol.com.br/vacina-autismo",
        snippet="Agência esclarece que vacinas são seguras e não causam autismo",
        rating="Falso",
        is_fact_check=True,
        source_tier=SourceTier.TIER1_OFFICIAL_OR_IFCN.value,
    )
    v_neg, conf_neg, _ = analyzer.evaluate_verdict([ev_neg], claim="Vacinas não causam autismo")
    assert ev_neg.stance == "SUPPORTS"
    assert v_neg == Verdict.VERDADEIRO


def test_epistemological_safeguard_no_debunk_fallback():
    """Valida que ausência de prova não gera veredito FAKE indevido (deve ser INCONCLUSIVO)."""
    orchestrator = FactCheckOrchestrator()
    fc_result = AnalyzerResult(
        analyzer_name="fact_check_api",
        verdict=Verdict.INCONCLUSIVO,
        confidence=0.50,
        claim="Novo restaurante abriu na Paulista",
        summary="Nenhuma evidência localizada.",
        reasons=[],
        sources=[],
        raw_details={"evidences": [], "sub_claims": []},
    )
    judge_result = AnalyzerResult(
        analyzer_name="llm_judge",
        verdict=Verdict.FAKE,
        confidence=0.85,
        claim="Novo restaurante abriu na Paulista",
        summary="Não encontrei referências desse restaurante.",
        reasons=["Ausência de matérias sobre a inauguração."],
        sources=[],
        raw_details={"sub_claims": []},
    )

    response = orchestrator._consolidate("Novo restaurante abriu na Paulista", [fc_result, judge_result])
    assert response.verdict == Verdict.INCONCLUSIVO
    assert any("Ausência de referências comprobatórias de falsidade" in r for r in response.reasons)


def test_multi_claim_safeguard_sub_claim_fallback():
    """Valida que sub-alegações sem desmentido externo não herdam FAKE espúrio."""
    orchestrator = FactCheckOrchestrator()
    fc_result = AnalyzerResult(
        analyzer_name="fact_check_api",
        verdict=Verdict.INCONCLUSIVO,
        confidence=0.55,
        claim="Texto composto",
        summary="Varredura inconclusiva.",
        reasons=[],
        sources=[],
        raw_details={
            "sub_claims": [
                {
                    "statement": "Brigar não é benéfico",
                    "verdict": Verdict.INCONCLUSIVO,
                    "confidence": 0.50,
                    "justification": "Sem cobertura conclusiva.",
                    "sources": [],
                }
            ]
        },
    )
    judge_result = AnalyzerResult(
        analyzer_name="llm_judge",
        verdict=Verdict.FAKE,
        confidence=0.85,
        claim="Texto composto",
        summary="Julgamento de sub-alegação.",
        reasons=[],
        sources=[],
        raw_details={
            "sub_claims": [
                {
                    "statement": "Brigar não é benéfico",
                    "verdict": Verdict.FAKE,
                    "confidence": 0.85,
                    "justification": "Nenhuma fonte confirma isso.",
                }
            ]
        },
    )

    response = orchestrator._consolidate("Texto composto", [fc_result, judge_result])
    assert response.verdict == Verdict.INCONCLUSIVO
    assert len(response.sub_claims) == 1
    assert response.sub_claims[0].verdict == Verdict.INCONCLUSIVO
    assert "Ausência de referências comprobatórias de falsidade" in response.sub_claims[0].justification


@pytest.mark.asyncio
async def test_orchestrator_analyzer_cache_and_lifecycle():
    """Valida o reuso de instâncias de analisadores em cache e encerramento limpo."""
    orchestrator = FactCheckOrchestrator()
    analyzers1 = orchestrator.get_active_analyzers()
    analyzers2 = orchestrator.get_active_analyzers()
    assert len(analyzers1) == len(analyzers2)
    for a1, a2 in zip(analyzers1, analyzers2):
        assert a1 is a2

    await orchestrator.aclose()
    assert len(orchestrator._cached_analyzers) == 0
