import pytest
from app.analyzers.fact_check_api import (
    SourceTier,
    get_source_tier,
    check_evidence_relevance,
    EvidenceItem,
    FactCheckApiAnalyzer,
    is_meta_debunk_claim,
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
    """Valida que alegações não encontradas em fontes confiáveis são tratadas como FAKE (provavelmente falso)."""
    orchestrator = FactCheckOrchestrator()
    fc_result = AnalyzerResult(
        analyzer_name="fact_check_api",
        verdict=Verdict.FAKE,
        confidence=0.75,
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
    assert response.verdict == Verdict.FAKE
    assert response.confidence >= 0.75
    assert any("não encontrada" in r.lower() or "ônus da prova" in r.lower() or "provavelmente fals" in r.lower() for r in response.reasons)


def test_multi_claim_safeguard_sub_claim_fallback():
    """Valida que sub-alegações sem respaldo factual são classificadas como FAKE (provavelmente falso)."""
    orchestrator = FactCheckOrchestrator()
    fc_result = AnalyzerResult(
        analyzer_name="fact_check_api",
        verdict=Verdict.FAKE,
        confidence=0.75,
        claim="Texto composto",
        summary="Varredura sem registros.",
        reasons=[],
        sources=[],
        raw_details={
            "sub_claims": [
                {
                    "statement": "Brigar não é benéfico",
                    "verdict": Verdict.FAKE,
                    "confidence": 0.75,
                    "justification": "Sem registros jornalísticos ou científicos.",
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
    assert response.verdict == Verdict.FAKE
    assert len(response.sub_claims) == 1
    assert response.sub_claims[0].verdict == Verdict.FAKE
    assert any(k in response.sub_claims[0].justification.lower() for k in ("provavelmente falsa", "sem registro", "nenhuma fonte", "não encontrad"))


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
    assert r1.verdict == Verdict.FAKE
    assert r1.confidence >= 0.70

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


def test_is_meta_debunk_claim():
    """Valida a detecção de meta-asserções de desmentido."""
    assert is_meta_debunk_claim("É falso que voto não pode ser usado no INSS") is True
    assert is_meta_debunk_claim("É mentira que vacina causa infarto") is True
    assert is_meta_debunk_claim("Não é verdade que o governo vai confiscar a poupança") is True
    assert is_meta_debunk_claim("Aos Fatos: É falso que Lula assinou decreto...") is True
    assert is_meta_debunk_claim("Não procede que haverá aumento de impostos") is True
    assert is_meta_debunk_claim("Voto nas eleições não pode ser utilizado como prova de vida") is False
    assert is_meta_debunk_claim("Governo anuncia novo programa social") is False


def test_meta_debunk_stance_and_verdict():
    """
    Valida que quando o usuário afirma que um boato é falso (ex: 'É falso que voto não pode ser usado no INSS'),
    e as evidências checam e desmentem o boato, a evidência dá suporte (SUPPORTS) e o veredito é VERDADEIRO.
    """
    analyzer = FactCheckApiAnalyzer()

    ev_aos_fatos = EvidenceItem(
        title="É falso que voto nas eleições não pode ser utilizado como prova de vida do INSS",
        source_name="Aos Fatos (Fact-Check)",
        url="https://aosfatos.org/noticias/voto-inss",
        snippet="Alegação revisada: Voto nas eleições não pode ser utilizado como prova de vida do INSS | Classificação: Falso",
        rating="Falso",
        is_fact_check=True,
        claim_reviewed="Voto nas eleições não pode ser utilizado como prova de vida do INSS",
        source_tier=SourceTier.TIER1_OFFICIAL_OR_IFCN.value,
    )

    ev_uol = EvidenceItem(
        title="Voto nas eleições de 2026 valerá como prova de vida do INSS",
        source_name="UOL Notícias (Fact-Check)",
        url="https://noticias.uol.com.br/confere/voto-inss",
        snippet="TSE e INSS confirmaram a integração",
        rating="Verdadeiro",
        is_fact_check=True,
        claim_reviewed="Voto nas eleições de 2026 valerá como prova de vida do INSS",
        source_tier=SourceTier.TIER1_OFFICIAL_OR_IFCN.value,
    )

    evidences = [ev_aos_fatos, ev_uol]
    claim = "É falso que voto não pode ser usado no INSS"

    verdict, confidence, reasons = analyzer.evaluate_verdict(evidences, claim=claim)

    assert ev_aos_fatos.stance == "SUPPORTS"
    assert ev_uol.stance == "SUPPORTS"
    assert verdict == Verdict.VERDADEIRO
    assert confidence >= 0.90
    assert any("confirma desmentido do boato apontado" in r or "comprovado por checador oficial" in r.lower() for r in reasons)


def test_meta_debunk_false_assertion():
    """
    Valida que se o usuário diz que um fato real é falso (ex: 'É falso que o Brasil foi pentacampeão'),
    e as evidências comprovam o fato, o veredito para o usuário é FAKE.
    """
    analyzer = FactCheckApiAnalyzer()

    ev_penta = EvidenceItem(
        title="Brasil conquista o pentacampeonato mundial de futebol",
        source_name="G1 Notícias",
        url="https://g1.globo.com/esporte/copa",
        snippet="Seleção brasileira vence a Alemanha e se consagra pentacampeã",
        rating="Verdadeiro",
        is_fact_check=True,
        claim_reviewed="Brasil conquista o pentacampeonato mundial",
        source_tier=SourceTier.TIER1_OFFICIAL_OR_IFCN.value,
    )

    claim = "É falso que o Brasil conquistou o pentacampeonato"
    verdict, confidence, reasons = analyzer.evaluate_verdict([ev_penta], claim=claim)

    assert ev_penta.stance == "REFUTES"
    assert verdict == Verdict.FAKE


def test_orchestrator_consolidation_meta_debunk():
    """Valida a consolidação final para uma meta-asserção verdadeira sem bloqueio por falso positivo de debunk."""
    orchestrator = FactCheckOrchestrator()

    fc_result = AnalyzerResult(
        analyzer_name="fact_check_api",
        verdict=Verdict.VERDADEIRO,
        confidence=0.95,
        claim="É falso que voto não pode ser usado no INSS",
        summary="Fontes de checagem confirmam que a tese de impedimento é um boato.",
        reasons=["Aos Fatos desmentiu o boato."],
        sources=["Aos Fatos", "UOL Confere"],
        raw_details={
            "evidences": [
                {
                    "title": "É falso que voto não pode ser utilizado como prova de vida",
                    "source_name": "Aos Fatos",
                    "rating": "Falso",
                    "stance": "SUPPORTS",
                    "source_tier": "tier1_official_or_ifcn",
                }
            ],
            "sub_claims": [
                {
                    "statement": "É falso que voto não pode ser usado no INSS",
                    "verdict": Verdict.VERDADEIRO,
                    "confidence": 0.95,
                    "justification": "Desmentido do boato confirmado.",
                    "sources": ["Aos Fatos"],
                    "evidences": [
                        {
                            "title": "É falso que voto não pode ser utilizado",
                            "source_name": "Aos Fatos",
                            "rating": "Falso",
                            "stance": "SUPPORTS",
                            "source_tier": "tier1_official_or_ifcn",
                        }
                    ],
                }
            ],
        },
    )

    judge_result = AnalyzerResult(
        analyzer_name="llm_judge",
        verdict=Verdict.VERDADEIRO,
        confidence=0.95,
        claim="É falso que voto não pode ser usado no INSS",
        summary="A afirmação do usuário está correta: as fontes comprovam que o voto pode sim ser utilizado.",
        reasons=["Aos Fatos desmentiu a tese contrária."],
        sources=["Aos Fatos"],
        raw_details={
            "sub_claims": [
                {
                    "statement": "É falso que voto não pode ser usado no INSS",
                    "verdict": Verdict.VERDADEIRO,
                    "confidence": 0.95,
                    "justification": "Afirmação verdadeira, boato desmentido.",
                }
            ]
        },
    )

    resp = orchestrator._consolidate("É falso que voto não pode ser usado no INSS", [fc_result, judge_result])
    assert resp.verdict == Verdict.VERDADEIRO
    assert resp.confidence >= 0.90
    assert len(resp.sub_claims) == 1
    assert resp.sub_claims[0].verdict == Verdict.VERDADEIRO


def test_claim_extractor_language_safeguard():
    """Valida o regex de detecção de contaminação linguística do claim extractor."""
    from app.analyzers.claim_extractor import ENGLISH_WORDS_PATTERN

    english_text = "voting can be used in INSS"
    portuguese_text = "É falso que voto não pode ser usado no INSS"

    assert bool(ENGLISH_WORDS_PATTERN.search(english_text)) is True
    assert bool(ENGLISH_WORDS_PATTERN.search(portuguese_text)) is False


def test_fact_check_api_stance_detection_on_declarative_reporting():
    """Valida que matérias jornalísticas reportando o fato (ex: nota de pesar, lamento oficial) são reconhecidas como confirmatórias."""
    analyzer = FactCheckApiAnalyzer()
    claim = "O Ministério da Saúde lamenta o falecimento de Paulo Roberto Teixeira"

    evidences = [
        EvidenceItem(
            title="Ministério da Saúde lamenta morte de Paulo Roberto Teixeira e destaca legado decisivo na resposta brasileira ao HIV e à aids - Agência Aids",
            source_name="Agência Aids",
            url="https://agenciaaids.com.br/noticia/ministerio-da-saude-lamenta-morte-de-paulo-roberto-teixeira/",
            snippet="O Ministério da Saúde manifestou profundo pesar pelo falecimento de Paulo Roberto Teixeira...",
            source_tier=SourceTier.TIER2_MAINSTREAM_MEDIA.value,
        ),
        EvidenceItem(
            title="Morre Paulo Roberto Teixeira, médico pioneiro no combate à Aids - VEJA SÃO PAULO",
            source_name="VEJA SÃO PAULO",
            url="https://vejasp.abril.com.br/cidades/morre-paulo-roberto-teixeira-medico-aids/",
            snippet="O Ministério da Saúde emitiu nota de pesar lamentando o falecimento...",
            source_tier=SourceTier.TIER2_MAINSTREAM_MEDIA.value,
        ),
    ]

    verdict, confidence, reasons = analyzer.evaluate_verdict(evidences, claim=claim)
    assert verdict == Verdict.VERDADEIRO
    assert confidence >= 0.85
    assert any(ev.stance == "SUPPORTS" for ev in evidences)


def test_consolidation_preserves_confirmed_verdict_when_evidences_support():
    """Valida que o orquestrador não rebaixa alegações confirmadas por fontes jornalísticas para inconclusivas."""
    orchestrator = FactCheckOrchestrator()
    claim = "O Ministério da Saúde lamenta o falecimento de Paulo Roberto Teixeira"

    fc_result = AnalyzerResult(
        analyzer_name="fact_check_api",
        verdict=Verdict.VERDADEIRO,
        confidence=0.89,
        claim=claim,
        reasons=["Leitura horizontal (Agência Aids): confirmação em notícia."],
        sources=["Agência Aids", "VEJA SÃO PAULO"],
        raw_details={
            "evidences": [
                {
                    "title": "Ministério da Saúde lamenta morte de Paulo Roberto Teixeira",
                    "source_name": "Agência Aids",
                    "url": "https://agenciaaids.com.br/...",
                    "source_tier": "tier2_mainstream_media",
                    "stance": "SUPPORTS",
                    "is_relevant": True,
                },
                {
                    "title": "Morre Paulo Roberto Teixeira, médico pioneiro - VEJA SÃO PAULO",
                    "source_name": "VEJA SÃO PAULO",
                    "url": "https://vejasp.abril.com.br/...",
                    "source_tier": "tier2_mainstream_media",
                    "stance": "SUPPORTS",
                    "is_relevant": True,
                },
            ],
            "sub_claims": [
                {
                    "statement": claim,
                    "verdict": Verdict.VERDADEIRO,
                    "confidence": 0.89,
                    "justification": "Confirmado por reportagens da imprensa.",
                }
            ],
        },
    )

    judge_result = AnalyzerResult(
        analyzer_name="llm_judge",
        verdict=Verdict.VERDADEIRO,
        confidence=0.95,
        claim=claim,
        summary="Alegação sobre o lamento do Ministério da Saúde pelo falecimento de Paulo Roberto Teixeira é comprovada como verdadeira com base em várias fontes noticiosas.",
        reasons=["O Ministério da Saúde expressou pesar pelo falecimento."],
        sources=["VEJA SÃO PAULO", "Agência Aids"],
        raw_details={
            "sub_claims": [
                {
                    "statement": claim,
                    "verdict": Verdict.VERDADEIRO,
                    "confidence": 0.95,
                    "justification": "A notícia do VEJA SÃO PAULO confirma diretamente o pesar oficial.",
                }
            ]
        },
    )

    resp = orchestrator._consolidate(claim, [fc_result, judge_result])
    assert resp.verdict == Verdict.VERDADEIRO
    assert resp.confidence >= 0.85
    assert "ausência de referências" not in resp.summary.lower()
    assert len(resp.sub_claims) == 1
    assert resp.sub_claims[0].verdict == Verdict.VERDADEIRO
    assert "ausência de referências" not in resp.sub_claims[0].justification.lower()


@pytest.mark.asyncio
async def test_heuristic_celebrity_scam_dialogue_detection():
    """Valida detecção heurística de entrevista/diálogo forjado e apelos de golpe comercial."""
    from app.analyzers.heuristic import HeuristicAnalyzer
    analyzer = HeuristicAnalyzer()

    scam_text = (
        "William Bonner: Boa noite! Hoje vamos explorar um tratamento de rejuvenescimento "
        "que tem conquistado muitas pessoas. Um dos grandes nomes que aderiu a esse tratamento "
        "é a icônica Vera Fischer. Recentemente, ela compartilhou sua experiência em uma entrevista. "
        "Vera Fischer: Em meus 72 anos, decidi me cuidar. Esse tratamento é super tranquilo, "
        "nada de botox ou cirurgias plásticas. O melhor de tudo: é acessível e cabe no bolso de qualquer mulher. "
        "Quer saber mais?"
    )

    result = await analyzer.analyze(scam_text, [])
    assert result.raw_details["urgency_lexicon_density"] > 0
    assert result.raw_details["composite_sensationalism_score"] >= 0.20
    assert any("diálogo/entrevista simulada" in r.lower() or "publicidade fraudulenta" in r.lower() for r in result.reasons)
    assert any("apelo comercial" in r.lower() or "cura milagrosa" in r.lower() for r in result.reasons)


def test_orchestrator_unverified_scam_consolidated_as_fake():
    """Valida a consolidação de alegação não encontrada em fontes como FAKE (provavelmente falso)."""
    orchestrator = FactCheckOrchestrator()
    claim = "Vera Fischer aderiu ao tratamento de rejuvenescimento facial sem botox"

    fc_result = AnalyzerResult(
        analyzer_name="fact_check_api",
        verdict=Verdict.FAKE,
        confidence=0.75,
        claim=claim,
        summary="Nenhuma evidência localizada em órgãos oficiais ou grandes veículos.",
        reasons=[
            "Nenhuma checagem prévia ou matéria em veículos de referência foi encontrada para este fato.",
            "Alegação não encontrada em fontes oficiais ou veículos confiáveis: sob o princípio de ônus da prova, afirmações públicas ou promessas de tratamento sem qualquer respaldo factual são tratadas como provavelmente falsas.",
        ],
        sources=[],
        raw_details={"evidences": [], "sub_claims": []},
    )

    judge_result = AnalyzerResult(
        analyzer_name="llm_judge",
        verdict=Verdict.FAKE,
        confidence=0.85,
        claim=claim,
        summary="A alegação de endosso por Vera Fischer não possui nenhum registro público em veículos jornalísticos ou notas oficiais, configurando formato clássico de golpe comercial.",
        reasons=[
            "Não foram encontrados registros oficiais ou jornalísticos confirmando a alegação.",
            "Formato típico de publicidade enganosa utilizando figura pública sem consentimento.",
        ],
        sources=["LLM Judge (phi3.5)"],
        raw_details={"sub_claims": []},
    )

    response = orchestrator._consolidate(claim, [fc_result, judge_result])
    assert response.verdict == Verdict.FAKE
    assert response.confidence >= 0.75
    assert any("ônus da prova" in r.lower() or "não encontrada" in r.lower() or "provavelmente fals" in r.lower() for r in response.reasons)


def test_orchestrator_multi_subclaims_unverified_all_fake():
    """Valida que múltiplas sub-alegações sem lastro resultem em veredito consolidado FAKE (provavelmente falso)."""
    orchestrator = FactCheckOrchestrator()
    full_text = "William Bonner e Vera Fischer promovem tratamento de rejuvenescimento"

    fc_result = AnalyzerResult(
        analyzer_name="fact_check_api",
        verdict=Verdict.FAKE,
        confidence=0.75,
        claim=full_text,
        summary="Nenhum registro encontrado.",
        reasons=[],
        sources=[],
        raw_details={
            "sub_claims": [
                {
                    "statement": "William Bonner vai explorar tratamento de rejuvenescimento",
                    "verdict": Verdict.FAKE,
                    "confidence": 0.75,
                    "justification": "Sem cobertura em veículos jornalísticos.",
                    "sources": [],
                    "evidences": [],
                },
                {
                    "statement": "Vera Fischer aderiu a esse tratamento",
                    "verdict": Verdict.FAKE,
                    "confidence": 0.75,
                    "justification": "Sem registros públicos comprovando a adesão.",
                    "sources": [],
                    "evidences": [],
                },
            ]
        },
    )

    judge_result = AnalyzerResult(
        analyzer_name="llm_judge",
        verdict=Verdict.FAKE,
        confidence=0.85,
        claim=full_text,
        summary="Alegações sem respaldo factual.",
        reasons=[],
        sources=[],
        raw_details={
            "sub_claims": [
                {
                    "statement": "William Bonner vai explorar tratamento de rejuvenescimento",
                    "verdict": Verdict.FAKE,
                    "confidence": 0.85,
                    "justification": "Declaração inexistente atribuída a William Bonner.",
                },
                {
                    "statement": "Vera Fischer aderiu a esse tratamento",
                    "verdict": Verdict.FAKE,
                    "confidence": 0.85,
                    "justification": "Depoimento forjado atribuído a Vera Fischer.",
                },
            ]
        },
    )

    response = orchestrator._consolidate(full_text, [fc_result, judge_result])
    assert response.verdict == Verdict.FAKE
    assert response.confidence >= 0.75
    assert len(response.sub_claims) == 2
    for sc in response.sub_claims:
        assert sc.verdict == Verdict.FAKE



