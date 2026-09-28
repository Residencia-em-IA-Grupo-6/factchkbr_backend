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
    assert result.verdict in [Verdict.VERDADEIRO, Verdict.FAKE, Verdict.SUSPEITO, Verdict.INCONCLUSIVO]


def test_registry_discovery():
    registry.auto_discover("app.analyzers")
    available = registry.list_available()
    assert "heuristic" in available
    assert "fact_check_api" in available
    assert "llm_judge" in available


@pytest.mark.asyncio
async def test_orchestrator_execution():
    orchestrator = FactCheckOrchestrator()
    response = await orchestrator.analyze("Mensagem para verificação", [])
    assert response.verdict in [Verdict.VERDADEIRO, Verdict.FAKE, Verdict.SUSPEITO, Verdict.INCONCLUSIVO]
    assert 0.0 <= response.confidence <= 1.0
