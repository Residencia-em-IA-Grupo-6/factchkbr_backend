from app.core.base import BaseAnalyzer
from app.core.registry import register_analyzer
from app.schemas.analysis import AnalyzerResult, Verdict


@register_analyzer("heuristic", weight=0.6)
class HeuristicAnalyzer(BaseAnalyzer):
    """
    Analisador 1: Regras e Heurísticas de texto (clickbait, sensacionalismo, pontuação).
    """

    async def analyze(self, text: str, urls: list[str]) -> AnalyzerResult:
        # TODO: Implementar regras heurísticas (ex: pontuação, palavras alarmistas, caixa alta)
        return AnalyzerResult(
            analyzer_name="heuristic",
            verdict=Verdict.SUSPEITO,
            confidence=0.70,
            reasons=["Padrão de escrita sensacionalista detectado por heurística textual."],
            sources=["Heurística Textual FactChkBR"]
        )
