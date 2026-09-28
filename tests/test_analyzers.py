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
