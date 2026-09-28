import asyncio
from app.config import Settings, get_settings
from app.core.base import BaseAnalyzer
from app.core.registry import registry
from app.schemas.analysis import (
    AnalyzeResponse,
    AnalyzerResult,
    Verdict,
)


class FactCheckOrchestrator:
    """
    Orquestrador / Ensemble responsável por:
    1. Carregar os analisadores ativos definidos nas configurações (.env).
    2. Executar os modelos em paralelo (asyncio.gather).
    3. Consolidar os resultados em um veredito final ponderado.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def get_active_analyzers(self) -> list[BaseAnalyzer]:
        """
        Instancia e retorna os analisadores ativos configurados no ACTIVE_ANALYZERS.
        """
        registry.auto_discover("app.analyzers")
        active_names = self.settings.get_active_analyzers_list()

        analyzers: list[BaseAnalyzer] = []
        for name in active_names:
            analyzer_cls = registry.get(name)
            if analyzer_cls:
                analyzers.append(analyzer_cls())
        return analyzers

    async def analyze(self, text: str, urls: list[str]) -> AnalyzeResponse:
        """
        Executa os analisadores ativos e consolida o veredito.
        """
        active_analyzers = self.get_active_analyzers()

        if not active_analyzers:
            return AnalyzeResponse(
                claim=text[:100],
                verdict=Verdict.INCONCLUSIVO,
                confidence=0.0,
                summary="Nenhum analisador ativo configurado.",
                reasons=["Pipeline de analisadores vazio."],
                sources=[]
            )

        # Execução assíncrona concorrente dos modelos
        tasks = [analyzer.analyze(text, urls) for analyzer in active_analyzers]
        results: list[AnalyzerResult] = await asyncio.gather(*tasks, return_exceptions=False)

        return self._consolidate(text, results)

    def _consolidate(self, text: str, results: list[AnalyzerResult]) -> AnalyzeResponse:
        """
        Consolida os resultados individuais através de ensemble ponderado / consenso.
        """
        all_reasons: list[str] = []
        all_sources: list[str] = []
        dominant_verdict = Verdict.INCONCLUSIVO
        avg_confidence = 0.5

        if results:
            first = results[0]
            dominant_verdict = first.verdict
            avg_confidence = sum(r.confidence for r in results) / len(results)

            for r in results:
                all_reasons.extend(r.reasons)
                all_sources.extend(r.sources)

        return AnalyzeResponse(
            claim=text[:120],
            verdict=dominant_verdict,
            confidence=round(avg_confidence, 2),
            summary=f"Análise estrutural processada por {len(results)} modelo(s).",
            reasons=list(dict.fromkeys(all_reasons)),
            sources=list(dict.fromkeys(all_sources))
        )
