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


