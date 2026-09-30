import json
import pytest
from app.analyzers.heuristic import HeuristicAnalyzer
from app.core.base import BaseAnalyzer
from app.core.orchestrator import FactCheckOrchestrator
from app.core.registry import registry
from app.schemas.analysis import AnalyzerResult, Verdict


@pytest.mark.asyncio
async def test_base_analyzer_inheritance():
    analyzer = HeuristicAnalyzer()
    assert isinstance(analyzer, BaseAnalyzer)
    result = await analyzer.analyze("Texto de teste", [])
    assert isinstance(result, AnalyzerResult)

    # HeuristicAnalyzer NÃO emite veredito (retorna verdict=None)
    assert result.verdict is None
    assert result.confidence == 0.0
    assert result.raw_details is not None
    assert "composite_sensationalism_score" in result.raw_details


def test_registry_discovery():
    registry.auto_discover("app.analyzers")
    available = registry.list_available()
    assert "claim_extractor" in available
    assert "heuristic" in available
    assert "fact_check_api" in available
    assert "llm_judge" in available



@pytest.mark.asyncio
async def test_orchestrator_execution():
    orchestrator = FactCheckOrchestrator()
    # Executa orquestrador contendo heuristic + outros modelos
    response = await orchestrator.analyze("Mensagem para verificação", [])
    # A resposta consolidada da API final sempre tem veredito válido
    assert response.verdict in [Verdict.VERDADEIRO, Verdict.FAKE, Verdict.SUSPEITO, Verdict.INCONCLUSIVO]
    assert 0.0 <= response.confidence <= 1.0


def test_heuristic_feature_extraction_contract():
    analyzer = HeuristicAnalyzer()
    text = "URGENTE!! A MÍDIA ESCONDE QUE VACINA MATA E ALTERA O DNA HUMANO???"
    features = analyzer.extract_features(text)

    # 1. Verifica presença estrita das 7 chaves especificadas
    expected_keys = {
        "uppercase_ratio",
        "allcaps_words_ratio",
        "excessive_punctuation_count",
        "exclamation_density",
        "question_density",
        "urgency_lexicon_density",
        "composite_sensationalism_score",
    }
    assert set(features.keys()) == expected_keys

    # 2. Tipagem e faixas de valores
    assert isinstance(features["uppercase_ratio"], float)
    assert isinstance(features["allcaps_words_ratio"], float)
    assert isinstance(features["excessive_punctuation_count"], float)
    assert isinstance(features["exclamation_density"], float)
    assert isinstance(features["question_density"], float)
    assert isinstance(features["urgency_lexicon_density"], float)
    assert isinstance(features["composite_sensationalism_score"], float)

    assert 0.0 <= features["uppercase_ratio"] <= 1.0
    assert 0.0 <= features["allcaps_words_ratio"] <= 1.0
    assert features["excessive_punctuation_count"] >= 2.0
    assert features["exclamation_density"] > 0.0
    assert features["question_density"] > 0.0
    assert features["urgency_lexicon_density"] > 0.0
    assert 0.0 <= features["composite_sensationalism_score"] <= 1.0
    assert features["composite_sensationalism_score"] > 0.60


def test_heuristic_acronym_filtering():
    analyzer = HeuristicAnalyzer()
    text = "O STF e o SUS anunciaram novas diretrizes em Brasília, no DF."
    features = analyzer.extract_features(text)

    assert features["allcaps_words_ratio"] == 0.0
    assert features["excessive_punctuation_count"] == 0.0
    assert features["composite_sensationalism_score"] < 0.20


def test_lexicon_repository_decoupling():
    from app.analyzers.lexicon_repository import (
        JsonFileLexiconRepository,
        BaseLexiconRepository,
        get_lexicon_repository
    )

    repo = get_lexicon_repository()
    assert isinstance(repo, BaseLexiconRepository)

    acronyms = repo.get_acronyms()
    assert "STF" in acronyms
    assert "SUS" in acronyms
    assert "DF" in acronyms
    assert "OMS" in acronyms

    patterns = repo.get_urgency_patterns()
    assert len(patterns) >= 15
    for label, compiled_regex, severity in patterns:
        assert isinstance(label, str)
        assert hasattr(compiled_regex, "findall")
        assert 0.0 <= severity <= 1.0


def test_heuristic_allcaps_short_words_and_articles():
    analyzer = HeuristicAnalyzer()

    # 1. Artigos no início de frase não devem ser contados como ALL CAPS (evita falsos positivos)
    neutral_text = "A vacina foi testada. O médico aprovou o medicamento."
    neutral_features = analyzer.extract_features(neutral_text)
    assert neutral_features["allcaps_words_ratio"] == 0.0

    # 2. Artigos e conjunções curtas no meio da frase em maiúsculas DEVEM ser contados
    # 'E' e 'O' são contados, enquanto 'DNA' é excluído por ser sigla legítima
    shouting_text = "VACINA MATA E ALTERA O DNA"
    shouting_features = analyzer.extract_features(shouting_text)
    # Palavras: VACINA(caps), MATA(caps), E(caps), ALTERA(caps), O(caps), DNA(sigla) = 5/6 = 0.8333
    assert shouting_features["allcaps_words_ratio"] == 0.8333

    # 3. Artigo 'A' no meio da frase
    mid_article_text = "MÍDIA ESCONDE A VERDADE"
    mid_article_features = analyzer.extract_features(mid_article_text)
    assert mid_article_features["allcaps_words_ratio"] == 1.0

    # 4. Caso composto com pontuação terminal e palavras curtas
    full_sample = "🚨 URGENTE!! A MÍDIA ESCONDE QUE VACINA MATA E ALTERA O DNA HUMANO???"
    full_features = analyzer.extract_features(full_sample)
    # 10 tokens ALL CAPS de 12 palavras totais (A pós-ponto descartado, E e O no meio incluídos, DNA sigla excluído)
    assert full_features["allcaps_words_ratio"] == 0.8333


@pytest.mark.asyncio
async def test_claim_extractor_adaptive_morphology():
    """Verifica se o extrator detecta verbos dinâmicos sem listas engessadas (morfologia verbal)."""
    from app.analyzers.claim_extractor import ClaimExtractorAnalyzer
    extractor = ClaimExtractorAnalyzer()

    # Verbos não listados estaticamente: colidiu, faliu, foram confiscados, aumentará
    texts = [
        "O avião monomotor colidiu com uma torre de transmissão.",
        "A empresa aérea faliu após dívida bilionária no exterior.",
        "Os bens do empresário foram confiscados pela Receita Federal.",
        "A Petrobras aumentará o valor do diesel na próxima segunda-feira.",
        "O ministro acabou de suspender todos os pagamentos.",
    ]

    for t in texts:
        result = await extractor.analyze(t, [])
        assert result.claim is not None, f"Falha ao isolar alegação com verbo morfológico: {t}"
        assert result.raw_details["extracted_claim"] != ""
        assert result.raw_details["claims_found"] >= 1


@pytest.mark.asyncio
async def test_claim_extractor_factual_isolation():
    """Verifica isolamento da alegação removendo lixo sensacionalista, alertas e apelos."""
    from app.analyzers.claim_extractor import ClaimExtractorAnalyzer
    extractor = ClaimExtractorAnalyzer()

    sample = (
        "🚨🚨 ATENÇÃO BRASIL! COMPARTILHEM ANTES QUE APAGUEM!! "
        "O governo federal aprovou aumento de 20% no combustível. "
        "Não deixe a mídia esconder! Repassem já!!"
    )

    result = await extractor.analyze(sample, [])
    assert result.claim is not None
    assert "aumento de 20% no combustível" in result.claim
    assert "COMPARTILHEM" not in result.claim
    assert "🚨" not in result.claim
    assert "Repassem" not in result.claim


@pytest.mark.asyncio
async def test_claim_extractor_opinion_and_noise_filtering():
    """Garante que frases que são puramente opinativas ou saudações não sejam marcadas como fatos."""
    from app.analyzers.claim_extractor import ClaimExtractorAnalyzer
    extractor = ClaimExtractorAnalyzer()

    opinions = [
        "Bom dia a todos, que Deus abençoe nossa nação maravilhosa!",
        "Eu acho esse político muito incompetente e antipático.",
        "Que absurdo inacreditável, que vergonha esse país!",
    ]

    for op in opinions:
        result = await extractor.analyze(op, [])
        assert result.claim is None
        assert result.raw_details["claims_found"] == 0


@pytest.mark.asyncio
async def test_orchestrator_claim_propagation():
    """Verifica se o Orchestrator utiliza a claim extraída pelo ClaimExtractor."""
    from app.config import Settings
    custom_settings = Settings(ACTIVE_ANALYZERS="claim_extractor,heuristic,fact_check_api,llm_judge")
    orchestrator = FactCheckOrchestrator(settings=custom_settings)

    raw_message = (
        "URGENTE!! BOMBA!! VEJA ANTES QUE APAGUEM! "
        "A Anvisa proibiu a venda de lote de azeite adulterado no país."
    )

    response = await orchestrator.analyze(raw_message, [])
    assert "proibiu a venda" in response.claim
    assert "URGENTE" not in response.claim
    assert "BOMBA" not in response.claim


@pytest.mark.asyncio
async def test_claim_extraction_contract_schema():
    """Valida o cumprimento estrito do contrato Pydantic (Camada 1)."""
    from app.analyzers.claim_extractor import ClaimExtractorAnalyzer
    from app.schemas.claim_extraction import ClaimExtractionContract, VerificationSourceType
    extractor = ClaimExtractorAnalyzer()

    text = "A Anvisa determinou a suspensão do lote de azeite adulterado em São Paulo."
    contract = await extractor.extract_contract(text)

    assert isinstance(contract, ClaimExtractionContract)
    assert len(contract.assertions) >= 1
    assertion = contract.assertions[0]
    assert assertion.triple.subject != ""
    assert assertion.triple.predicate != ""
    assert assertion.triple.object != ""
    assert VerificationSourceType.AGENCIA_REGULADORA in assertion.suggested_source_types


@pytest.mark.asyncio
async def test_claim_decomposer_atomic_and_triples():
    """Valida a decomposição de períodos compostos em proposições atômicas e triplas (Camada 3)."""
    from app.analyzers.claim_extractor import ClaimExtractorAnalyzer
    extractor = ClaimExtractorAnalyzer()

    compound_text = (
        "O Ministério da Saúde cancelou a compra dos remédios "
        "porque o laboratório farmacêutico fraudou os testes clínicos."
    )

    contract = await extractor.extract_contract(compound_text)
    assert len(contract.assertions) == 2
    # Proposição 1
    assert "Ministério da Saúde" in contract.assertions[0].statement
    # Proposição 2
    assert "laboratório farmacêutico fraudou" in contract.assertions[1].statement


def test_spacy_cleaner_and_ner():
    """Valida a camada de limpeza e NER (Camada 2)."""
    from app.analyzers.claim_extractor import SpacyNERCleaner
    cleaner = SpacyNERCleaner()

    raw = "🚨🚨 BOMBA!! URGENTE: O ministro Alexandre de Moraes e o STF suspenderam a rede social no Brasil???"
    cleaned = cleaner.clean_text(raw)

    assert "🚨" not in cleaned
    assert "BOMBA" not in cleaned
    assert "???" not in cleaned

    entities = cleaner.extract_entities(cleaned)
    assert "ORG" in entities
    assert "STF" in entities["ORG"]
    assert "LOC" in entities
    assert "Brasil" in entities["LOC"]


@pytest.mark.asyncio
async def test_claim_extractor_medical_action_rumor():
    """Valida extração de alegações médicas e biológicas no presente do indicativo."""
    from app.analyzers.claim_extractor import ClaimExtractorAnalyzer
    extractor = ClaimExtractorAnalyzer()

    text = "Médico extrai vermes do coração de uma pessoa que come carne de porco"
    result = await extractor.analyze(text, [])

    assert result.claim is not None
    assert "extrai vermes" in result.claim
    assert "PER" in result.raw_details["entities"]
    assert "Médico" in result.raw_details["entities"]["PER"]
    assert len(result.raw_details["assertions"]) >= 1
    triple = result.raw_details["assertions"][0]["triple"]
    assert triple["subject"] == "Médico"
    assert triple["predicate"] == "extrai"


def test_fact_check_api_evaluate_verdict_ifcn():
    """Valida avaliação de veredito a partir de checagens IFCN do Google Fact Check."""
    from app.analyzers.fact_check_api import FactCheckApiAnalyzer, EvidenceItem

    analyzer = FactCheckApiAnalyzer()

    # 1. Caso Falso / Mentira
    ev_fake = [
        EvidenceItem(
            title="Não é verdade que vacina altera o DNA",
            source_name="Agência Lupa",
            url="https://lupa.news/exemplo",
            rating="Falso",
            is_fact_check=True,
        )
    ]
    v_fake, conf_fake, reasons_fake = analyzer.evaluate_verdict(ev_fake)
    assert v_fake == Verdict.FAKE
    assert conf_fake >= 0.90
    assert any("Agência Lupa" in r for r in reasons_fake)

    # 2. Caso Verdadeiro / Comprovado
    ev_true = [
        EvidenceItem(
            title="Anvisa aprova novo tratamento contra a dengue",
            source_name="Fato ou Fake",
            url="https://g1.globo.com/fato-ou-fake",
            rating="Fato",
            is_fact_check=True,
        )
    ]
    v_true, conf_true, reasons_true = analyzer.evaluate_verdict(ev_true)
    assert v_true == Verdict.VERDADEIRO
    assert conf_true >= 0.90

    # 3. Caso Enganoso / Fora de contexto
    ev_misleading = [
        EvidenceItem(
            title="Vídeo com fala cortada distorce declaração",
            source_name="Aos Fatos",
            url="https://aosfatos.org",
            rating="Enganoso",
            is_fact_check=True,
        )
    ]
    v_susp, conf_susp, _ = analyzer.evaluate_verdict(ev_misleading)
    assert v_susp == Verdict.SUSPEITO
    assert conf_susp >= 0.80


def test_fact_check_api_evaluate_verdict_lateral_reading():
    """Valida avaliação de veredito via Leitura Horizontal (notícias e órgãos de saúde)."""
    from app.analyzers.fact_check_api import FactCheckApiAnalyzer, EvidenceItem

    analyzer = FactCheckApiAnalyzer()

    # 1. Desmentido detectado no título
    ev_debunk = [
        EvidenceItem(
            title="É mentira que médico extraiu vermes do coração de consumidor de carne",
            source_name="G1",
            url="https://g1.globo.com/fato-ou-fake/noticia/123",
            rating=None,
            is_fact_check=False,
        )
    ]
    v_debunk, conf_debunk, reasons_debunk = analyzer.evaluate_verdict(ev_debunk)
    assert v_debunk == Verdict.FAKE
    assert conf_debunk >= 0.80
    assert any("desmentido ou contestação" in r for r in reasons_debunk)

    # 2. Confirmação detectada no título
    ev_confirm = [
        EvidenceItem(
            title="Anvisa determina suspensão imediata de lote de azeite adulterado",
            source_name="Agência Brasil",
            url="https://agenciabrasil.ebc.com.br/saude",
            rating=None,
            is_fact_check=False,
        )
    ]
    v_conf, conf_conf, reasons_conf = analyzer.evaluate_verdict(ev_confirm)
    assert v_conf == Verdict.VERDADEIRO
    assert conf_conf >= 0.78


def test_fact_check_api_empty_evidences():
    """Garante retorno neutro e inconclusivo na ausência de matérias ou checagens."""
    from app.analyzers.fact_check_api import FactCheckApiAnalyzer

    analyzer = FactCheckApiAnalyzer()
    verdict, conf, reasons = analyzer.evaluate_verdict([])
    assert verdict == Verdict.INCONCLUSIVO
    assert conf == 0.50
    assert len(reasons) >= 1


@pytest.mark.asyncio
async def test_llm_judge_offline_fallback():
    """Valida comportamento defensivo do LLM Judge quando o provedor estiver offline."""
    from app.analyzers.llm_judge import LlmJudgeAnalyzer

    judge = LlmJudgeAnalyzer()
    res = await judge.analyze("Alegação de teste para fallback", [])

    assert isinstance(res, AnalyzerResult)
    assert res.analyzer_name == "llm_judge"
    assert res.verdict == Verdict.INCONCLUSIVO
    assert res.confidence == 0.50
    assert "offline" in res.summary.lower() or "ollama" in res.summary.lower()


def test_select_primary_assertion_by_category():
    """Valida a priorização da asserção substantiva baseada na taxonomia semântica (sem listas hardcoded)."""
    from app.schemas.claim_extraction import AtomicAssertion, ClaimCategory, KnowledgeTriple, VerificationSourceType
    from app.analyzers.claim_extractor import select_primary_assertion

    noise_assertion = AtomicAssertion(
        id=1,
        statement="Não adianta brigar comigo.",
        category=ClaimCategory.CONVERSATIONAL_NOISE,
        triple=KnowledgeTriple(subject="sujeito", predicate="brigar", object="comigo"),
        suggested_source_types=[],
        is_check_worthy=False,
    )
    substantive_assertion = AtomicAssertion(
        id=2,
        statement="Vacinas causam autismo.",
        category=ClaimCategory.FACTUAL_CLAIM,
        triple=KnowledgeTriple(subject="Vacinas", predicate="causam", object="autismo"),
        suggested_source_types=[
            VerificationSourceType.AGENCIA_REGULADORA,
            VerificationSourceType.INSTITUTO_PESQUISA,
        ],
        is_check_worthy=True,
    )
    attribution_assertion = AtomicAssertion(
        id=3,
        statement="A afirmação está na bula.",
        category=ClaimCategory.ATTRIBUTION,
        triple=KnowledgeTriple(subject="Afirmação", predicate="está", object="bula"),
        suggested_source_types=[VerificationSourceType.DADOS_PUBLICOS],
        is_check_worthy=True,
    )

    chosen = select_primary_assertion([noise_assertion, substantive_assertion, attribution_assertion])
    assert chosen is not None
    assert chosen.statement == "Vacinas causam autismo."
    assert chosen.category == ClaimCategory.FACTUAL_CLAIM


def test_select_primary_assertion_with_suggested_primary():
    """Valida a correspondência direta com o primary_claim identificado pelo modelo semântico."""
    from app.schemas.claim_extraction import AtomicAssertion, ClaimCategory, KnowledgeTriple, VerificationSourceType
    from app.analyzers.claim_extractor import select_primary_assertion

    a1 = AtomicAssertion(
        id=1,
        statement="A taxa de juros subiu 0.5%",
        category=ClaimCategory.FACTUAL_CLAIM,
        triple=KnowledgeTriple(subject="taxa de juros", predicate="subiu", object="0.5%"),
        suggested_source_types=[VerificationSourceType.ORGAO_OFICIAL],
        is_check_worthy=True,
    )
    a2 = AtomicAssertion(
        id=2,
        statement="O ministro declarou que a inflação está controlada",
        category=ClaimCategory.ATTRIBUTION,
        triple=KnowledgeTriple(subject="ministro", predicate="declarou", object="inflação controlada"),
        suggested_source_types=[VerificationSourceType.ORGAO_OFICIAL],
        is_check_worthy=True,
    )

    chosen = select_primary_assertion([a1, a2], suggested_primary="taxa de juros subiu 0.5%")
    assert chosen is not None
    assert chosen.id == 1


def test_orchestrator_inconclusive_safeguard_without_debunk():
    """Garante que a ausência de referências de desmentido resulte em INCONCLUSIVO (sem falsos FAKE)."""
    orchestrator = FactCheckOrchestrator()

    # Simula resultado de fact_check_api sem evidências de desmentido (INCONCLUSIVO)
    fc_result = AnalyzerResult(
        analyzer_name="fact_check_api",
        verdict=Verdict.INCONCLUSIVO,
        confidence=0.55,
        claim="Fatoide recente sem cobertura",
        summary="Nenhuma checagem prévia foi encontrada.",
        reasons=["Ausência de registros para este fato."],
        sources=[],
        raw_details={"evidences": []},
    )

    # Simula LLM judge que emitiu FAKE por inferência sem fonte
    judge_result = AnalyzerResult(
        analyzer_name="llm_judge",
        verdict=Verdict.FAKE,
        confidence=0.85,
        claim="Fatoide recente sem cobertura",
        summary="A alegação não possui dados confirmatórios.",
        reasons=["Ausência de provas"],
        sources=[],
    )

    response = orchestrator._consolidate("Fatoide recente sem cobertura", [fc_result, judge_result])
    assert response.verdict == Verdict.INCONCLUSIVO
    assert any("impedem a classificação como fake" in r.lower() or "ausência de referências" in r.lower() for r in response.reasons)
    assert any(k in response.summary.lower() for k in ("recente", "insuficiente", "ausência", "dados", "imprecis"))


def test_fact_check_api_desmentido_rating():
    """Garante que a classificação 'Desmentido' de agências IFCN resulte em FAKE com alta confiança."""
    from app.analyzers.fact_check_api import EvidenceItem, FactCheckApiAnalyzer
    analyzer = FactCheckApiAnalyzer()

    evidences = [
        EvidenceItem(
            title="É falso que voto nas eleições não pode ser utilizado como prova de vida do INSS",
            source_name="Aos Fatos",
            url="https://aosfatos.org/noticias/...",
            snippet="Checagem oficial",
            rating="Desmentido",
            is_fact_check=True,
        )
    ]

    verdict, confidence, reasons = analyzer.evaluate_verdict(evidences, claim="voto nas eleições não pode ser utilizado como prova de vida")
    assert verdict == Verdict.FAKE
    assert confidence >= 0.90
    assert any("desmentido por checador oficial" in r.lower() for r in reasons)


@pytest.mark.asyncio
async def test_llm_judge_polarity_inversion_safeguard():
    """Garante que inversões de polaridade do LLM (diz 'desmentida' mas marca VERDADEIRO) sejam corrigidas."""
    from unittest.mock import AsyncMock, MagicMock, patch
    from app.analyzers.llm_judge import LlmJudgeAnalyzer

    judge = LlmJudgeAnalyzer()

    # Simula resposta do LLM com inversão de polaridade (VERDADEIRO em veredito, mas resumo aponta desmentido)
    mock_payload = {
        "choices": [
            {
                "message": {
                    "content": json.dumps({
                        "verdict": "VERDADEIRO",
                        "confidence": 0.95,
                        "summary": "A alegação de que o voto não pode ser utilizado foi desmentida por várias fontes confiáveis e pelo TSE.",
                        "reasons": [
                            "O Aos Fatos explicitamente desmentiu a afirmação",
                            "TSE confirma que comparecimento vale como prova de vida"
                        ],
                        "sources": ["Aos Fatos", "TSE"]
                    })
                }
            }
        ]
    }

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = mock_payload

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_resp
        res = await judge.analyze_with_context(
            text="voto nas eleições não pode ser utilizado como prova de vida do INSS",
            urls=[],
            evidences=[{"source_name": "Aos Fatos", "title": "É falso que voto não pode ser utilizado", "rating": "Desmentido"}]
        )

        assert res.verdict == Verdict.FAKE
        assert res.raw_details.get("polarity_corrected") is True


def test_orchestrator_debunk_precedence_over_hallucinated_verdadeiro():
    """Garante que checagens oficiais com desmentido prevaleçam sobre alucinação de VERDADEIRO."""
    orchestrator = FactCheckOrchestrator()

    fc_result = AnalyzerResult(
        analyzer_name="fact_check_api",
        verdict=Verdict.FAKE,
        confidence=0.85,
        claim="voto nas eleições não pode ser utilizado como prova de vida",
        summary="A alegação foi desmentida por agências de checagem.",
        reasons=["Desmentido por checador oficial"],
        sources=["Aos Fatos"],
        raw_details={
            "evidences": [
                {"source_name": "Aos Fatos", "title": "É falso que voto...", "rating": "Desmentido"}
            ]
        },
    )

    judge_result = AnalyzerResult(
        analyzer_name="llm_judge",
        verdict=Verdict.VERDADEIRO,
        confidence=0.95,
        claim="voto nas eleições não pode ser utilizado como prova de vida",
        summary="Alegação afirmando que o voto não pode ser utilizado foi desmentida.",
        reasons=["Aos Fatos desmentiu a afirmação"],
        sources=["Aos Fatos"],
        raw_details={"model": "phi3.5"},
    )

    response = orchestrator._consolidate("voto nas eleições não pode ser utilizado como prova de vida", [fc_result, judge_result])
    assert response.verdict == Verdict.FAKE
    assert any("prevalência de checagem oficial" in r.lower() for r in response.reasons)


def test_fact_check_api_aggregate_sub_verdicts_mixed():
    """Valida o impacto no score e veredito quando há mistura de alegação falsa e verdadeira."""
    from app.analyzers.fact_check_api import FactCheckApiAnalyzer
    analyzer = FactCheckApiAnalyzer()

    sub_results = [
        {
            "statement": "vacinas causam autismo",
            "verdict": Verdict.FAKE,
            "confidence": 0.95,
            "justification": "Desmentido pela comunidade médica e científica.",
            "sources": ["Fiocruz"],
            "evidences": [],
        },
        {
            "statement": "vacinas são distribuídas gratuitamente no SUS",
            "verdict": Verdict.VERDADEIRO,
            "confidence": 0.90,
            "justification": "Confirmado pelo Ministério da Saúde e Programa Nacional de Imunizações.",
            "sources": ["Ministério da Saúde"],
            "evidences": [],
        },
    ]

    verdict, confidence, reasons = analyzer.aggregate_sub_verdicts(sub_results)
    # Conteúdo misto resulta em SUSPEITO
    assert verdict == Verdict.SUSPEITO
    assert confidence >= 0.85
    assert any("conteúdo misto" in r.lower() for r in reasons)
    assert any("vacinas causam autismo" in r for r in reasons)
    assert any("vacinas são distribuídas gratuitamente" in r for r in reasons)


@pytest.mark.asyncio
async def test_fact_check_api_multi_claim_isolated_checking():
    """Verifica se fact_check_api executa checagem de cada alegação isoladamente quando passado 'assertions'."""
    from unittest.mock import AsyncMock, patch
    from app.analyzers.fact_check_api import FactCheckApiAnalyzer

    analyzer = FactCheckApiAnalyzer()

    # Mock das funções de busca
    with patch.object(analyzer, "check_single_claim", new_callable=AsyncMock) as mock_single:
        mock_single.side_effect = [
            {
                "statement": "alegação 1",
                "verdict": Verdict.FAKE,
                "confidence": 0.9,
                "justification": "Falsidade comprovada 1",
                "sources": ["Fonte A"],
                "evidences": [],
            },
            {
                "statement": "alegação 2",
                "verdict": Verdict.VERDADEIRO,
                "confidence": 0.88,
                "justification": "Verdade comprovada 2",
                "sources": ["Fonte B"],
                "evidences": [],
            },
        ]

        res = await analyzer.analyze("Texto completo", [], assertions=["alegação 1", "alegação 2"])

        assert mock_single.call_count == 2
        assert res.verdict == Verdict.SUSPEITO
        assert res.raw_details["claims_checked"] == 2
        assert len(res.raw_details["sub_claims"]) == 2


def test_orchestrator_multi_claim_score_and_justification():
    """Valida a consolidação do orquestrador com discriminação e justificativa por alegação."""
    from app.schemas.analysis import SubClaimAnalysis

    orchestrator = FactCheckOrchestrator()

    fc_result = AnalyzerResult(
        analyzer_name="fact_check_api",
        verdict=Verdict.SUSPEITO,
        confidence=0.88,
        claim="Alegação composta",
        summary="Varredura avaliou 2 alegações isoladamente.",
        reasons=["Conteúdo misto detectado."],
        sources=["Fonte 1", "Fonte 2"],
        raw_details={
            "sub_claims": [
                {
                    "statement": "O hospital foi inaugurado ontem",
                    "verdict": Verdict.VERDADEIRO,
                    "confidence": 0.90,
                    "justification": "Confirmado pela assessoria oficial do governo.",
                    "sources": ["Gov"],
                },
                {
                    "statement": "Cinquenta pacientes morreram por erro médico",
                    "verdict": Verdict.FAKE,
                    "confidence": 0.95,
                    "justification": "Desmentido pela secretaria de saúde e perícia.",
                    "sources": ["Aos Fatos"],
                },
            ]
        },
    )

    judge_result = AnalyzerResult(
        analyzer_name="llm_judge",
        verdict=Verdict.SUSPEITO,
        confidence=0.90,
        claim="Alegação composta",
        summary="O texto mistura um fato real com um boato infundado.",
        reasons=["Uma alegação é verdadeira e outra é falsa."],
        sources=["Gov", "Aos Fatos"],
        raw_details={
            "sub_claims": [
                {
                    "statement": "O hospital foi inaugurado ontem",
                    "verdict": Verdict.VERDADEIRO,
                    "confidence": 0.90,
                    "justification": "Inauguração oficial ocorreu de fato conforme cronograma público.",
                },
                {
                    "statement": "Cinquenta pacientes morreram por erro médico",
                    "verdict": Verdict.FAKE,
                    "confidence": 0.95,
                    "justification": "Boato sem qualquer evidência fática, desmentido por órgãos oficiais.",
                },
            ]
        },
    )

    response = orchestrator._consolidate("Texto composto de teste", [fc_result, judge_result])

    # Verifica veredito composto e pontuação no score
    assert response.verdict == Verdict.SUSPEITO
    assert 0.85 <= response.confidence <= 1.0

    # Verifica presença da lista estruturada sub_claims
    assert len(response.sub_claims) == 2
    assert isinstance(response.sub_claims[0], SubClaimAnalysis)
    assert response.sub_claims[0].statement == "O hospital foi inaugurado ontem"
    assert response.sub_claims[0].verdict == Verdict.VERDADEIRO
    assert "inauguração" in response.sub_claims[0].justification.lower()

    assert response.sub_claims[1].statement == "Cinquenta pacientes morreram por erro médico"
    assert response.sub_claims[1].verdict == Verdict.FAKE
    assert "boato" in response.sub_claims[1].justification.lower() or "desmentido" in response.sub_claims[1].justification.lower()

    # Verifica razões detalhadas com pontuação de cada alegação
    assert any("[alegação 1 - verdadeiro]" in r.lower() for r in response.reasons)
    assert any("[alegação 2 - fake]" in r.lower() for r in response.reasons)








