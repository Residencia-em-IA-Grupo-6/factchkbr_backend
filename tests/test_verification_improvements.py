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


def test_adversarial_inss_negative_claim_debunk():
    """Valida que desmentido oficial de proposição negativa resulta estritamente em FAKE."""
    analyzer = FactCheckApiAnalyzer()
    claim = "O voto nas eleições não pode ser utilizado como prova de vida do INSS"

    # Caso 1: Aos Fatos desmentindo via Leitura Horizontal
    ev_aos_fatos = EvidenceItem(
        title="É falso que voto nas eleições não pode ser utilizado como prova de vida do INSS",
        source_name="Aos Fatos",
        url="https://aosfatos.org/noticias/inss-voto-prova-de-vida",
        rating=None,
        is_fact_check=False,
        source_tier=SourceTier.TIER1_OFFICIAL_OR_IFCN.value,
    )
    v1, conf1, _ = analyzer.evaluate_verdict([ev_aos_fatos], claim=claim)
    assert ev_aos_fatos.stance == "REFUTES"
    assert v1 == Verdict.FAKE
    assert conf1 >= 0.85

    # Caso 2: TSE confirmando que o comparecimento pode ser utilizado
    ev_tse = EvidenceItem(
        title="Comparecimento à votação pode ser utilizado como prova de vida do INSS",
        source_name="Tribunal Superior Eleitoral",
        url="https://www.tse.jus.br/comunicacao/noticias/prova-de-vida-inss",
        rating=None,
        is_fact_check=False,
        source_tier=SourceTier.TIER1_OFFICIAL_OR_IFCN.value,
    )
    v2, conf2, _ = analyzer.evaluate_verdict([ev_tse], claim=claim)
    assert ev_tse.stance == "REFUTES"
    assert v2 == Verdict.FAKE
    assert conf2 >= 0.85


def test_adversarial_golpe_context():
    """Garante que a palavra 'golpe' em contexto de iniciativa governamental não seja tratada como desmentido."""
    analyzer = FactCheckApiAnalyzer()
    claim = "Governo anunciou nova regra do Pix"
    ev = EvidenceItem(
        title="Governo anuncia golpe contra fraudes no Pix com nova regra",
        source_name="G1",
        url="https://g1.globo.com/economia/noticia/regras-pix",
        rating=None,
        is_fact_check=False,
        source_tier=SourceTier.TIER2_MAINSTREAM_MEDIA.value,
    )
    v, conf, _ = analyzer.evaluate_verdict([ev], claim=claim)
    assert ev.stance == "SUPPORTS"
    assert v == Verdict.VERDADEIRO


def test_adversarial_tiering_spoofing():
    """Garante que domínios não-oficiais não recebam Tier 1 por coincidência de substring."""
    assert get_source_tier("whoami news", "https://whoami.net") == SourceTier.UNKNOWN
    assert get_source_tier("Blog do STF Fake", "https://stfnoticias-urgente.com") == SourceTier.UNKNOWN
    assert get_source_tier("Afpnews", "https://afp-boatos-zap.com") == SourceTier.UNKNOWN
    assert get_source_tier("Desconhecido", "https://valoragora.blogspot.com") == SourceTier.UNKNOWN
    assert get_source_tier("Terra", "https://terra.com.br") == SourceTier.TIER2_MAINSTREAM_MEDIA


def test_adversarial_orchestrator_symmetrical_epistemology():
    """Garante simetria: ausência de comprovação impede VERDADEIRO; evidência factual prevalece sobre alucinação."""
    orchestrator = FactCheckOrchestrator()

    # 1. Judge alucina VERDADEIRO sem nenhuma evidência
    fc_empty = AnalyzerResult(
        analyzer_name="fact_check_api",
        verdict=Verdict.INCONCLUSIVO,
        confidence=0.50,
        claim="Fato sem evidência",
        summary="Sem evidências.",
        reasons=[],
        sources=[],
        raw_details={"evidences": [], "sub_claims": []},
    )
    judge_hallucinated = AnalyzerResult(
        analyzer_name="llm_judge",
        verdict=Verdict.VERDADEIRO,
        confidence=0.90,
        claim="Fato sem evidência",
        summary="Acredito que é verdade.",
        reasons=[],
        sources=[],
        raw_details={"sub_claims": []},
    )
    r1 = orchestrator._consolidate("Fato sem evidência", [fc_empty, judge_hallucinated])
    assert r1.verdict == Verdict.INCONCLUSIVO

    # 2. Fact-check tem confirmação documental e Judge alucina FAKE com alta confiança
    fc_proven = AnalyzerResult(
        analyzer_name="fact_check_api",
        verdict=Verdict.VERDADEIRO,
        confidence=0.90,
        claim="Fato comprovado",
        summary="Confirmado por portaria oficial.",
        reasons=["Publicação em Diário Oficial."],
        sources=["Diário Oficial"],
        raw_details={
            "evidences": [
                {
                    "title": "Portaria confirma nova diretriz",
                    "source_name": "Gov.br",
                    "rating": "Verdadeiro",
                    "stance": "SUPPORTS",
                    "source_tier": "tier1_official_or_ifcn",
                }
            ],
            "sub_claims": [],
        },
    )
    judge_fake_hallucination = AnalyzerResult(
        analyzer_name="llm_judge",
        verdict=Verdict.FAKE,
        confidence=0.95,
        claim="Fato comprovado",
        summary="Não achei evidências então é falso.",
        reasons=[],
        sources=[],
        raw_details={"sub_claims": []},
    )
    r2 = orchestrator._consolidate("Fato comprovado", [fc_proven, judge_fake_hallucination])
    assert r2.verdict == Verdict.VERDADEIRO
    assert r2.confidence == 0.90
