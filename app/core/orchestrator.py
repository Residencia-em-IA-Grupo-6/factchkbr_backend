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
    3. Consolidar os resultados em um veredito final ponderado, ignorando
       analisadores que atuam puramente como extratores de features (sem veredito).
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
        Se claim_extractor estiver presente, ele extrai a alegação factual central primeiro
        e repassa para os analisadores de checagem/LLM.
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

        # Se claim_extractor estiver ativo, isola a alegação factual primeiro
        extractor = next((a for a in active_analyzers if a.name == "claim_extractor"), None)
        other_analyzers = [a for a in active_analyzers if a.name != "claim_extractor"]

        results: list[AnalyzerResult] = []
        target_claim = text

        if extractor:
            extractor_result = await extractor.analyze(text, urls)
            results.append(extractor_result)
            if extractor_result.claim:
                target_claim = extractor_result.claim

        if other_analyzers:
            # Analisadores estilísticos/heurísticos analisam o texto bruto com formatação;
            # APIs de checagem e LLMs analisam a alegação factual isolada.
            tasks = []
            for analyzer in other_analyzers:
                if analyzer.name == "heuristic":
                    tasks.append(analyzer.analyze(text, urls))
                else:
                    tasks.append(analyzer.analyze(target_claim, urls))

            other_results: list[AnalyzerResult] = await asyncio.gather(*tasks, return_exceptions=False)
            results.extend(other_results)

        return self._consolidate(text, results)

    def _consolidate(self, text: str, results: list[AnalyzerResult]) -> AnalyzeResponse:
        """
        Consolida os resultados individuais através de ensemble ponderado / consenso.
        Filtra apenas analisadores que emitiram veredito concreto.
        """
        all_reasons: list[str] = []
        all_sources: list[str] = []
        primary_claim: str | None = None

        # Coleta todas as razões, fontes e alegação isolada
        for r in results:
            all_reasons.extend(r.reasons)
            all_sources.extend(r.sources)
            if r.claim and not primary_claim:
                primary_claim = r.claim

        # Filtra apenas os analisadores que emitem veredito
        verdict_bearing_results = [r for r in results if r.verdict is not None]

        if verdict_bearing_results:
            # Seleciona o veredito dos modelos decisores
            first = verdict_bearing_results[0]
            dominant_verdict = first.verdict
            avg_confidence = sum(r.confidence for r in verdict_bearing_results) / len(verdict_bearing_results)
            summary = f"Análise consolidada por {len(verdict_bearing_results)} modelo(s) decisor(es) com apoio de {len(results) - len(verdict_bearing_results)} módulo(s) de features."
        else:
            # Caso nenhum modelo tenha emitido veredito (ex: apenas HeuristicAnalyzer ativo)
            dominant_verdict = Verdict.INCONCLUSIVO
            avg_confidence = 0.5
            summary = "Extração de features concluída; nenhum modelo decisor emitiu veredito final."

        return AnalyzeResponse(
            claim=primary_claim or text[:120],
            verdict=dominant_verdict,
            confidence=round(avg_confidence, 2),
            summary=summary,
            reasons=list(dict.fromkeys(all_reasons)),
            sources=list(dict.fromkeys(all_sources))
        )

