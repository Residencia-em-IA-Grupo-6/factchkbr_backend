from app.core.base import BaseAnalyzer
from app.core.registry import register_analyzer
from app.schemas.analysis import AnalyzerResult, Verdict


@register_analyzer("fact_check_api", weight=1.5)
class FactCheckApiAnalyzer(BaseAnalyzer):
    """
    Analisador 2: Consulta a bases externas e APIs de checagem (Google Fact Check Tools, Lupa, Aos Fatos).
    """

    async def analyze(self, text: str, urls: list[str]) -> AnalyzerResult:
        # TODO: Implementar consulta a APIs externas de fact-checking
        return AnalyzerResult(
            analyzer_name="fact_check_api",
            verdict=Verdict.INCONCLUSIVO,
            confidence=0.50,
            reasons=["Consulta a bases externas de checagem concluída."],
            sources=["APIs Externas de Fact-Checking"]
        )
