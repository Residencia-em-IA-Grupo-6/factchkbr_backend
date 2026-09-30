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
        Executa o pipeline linear contextual:
        1. Heurística (texto bruto, captura estilo e sensacionalismo)
        2. Extrator de Alegações (spaCy + LLM, isola fatos atômicos e faz gatekeeping)
        3. Fact-Check API & Leitura Horizontal (pesquisa evidências com a claim limpa)
        4. LLM Judge (julga o fato embasado nas evidências e nas métricas)
        """
        active_map = {a.name: a for a in self.get_active_analyzers()}

        if not active_map:
            return AnalyzeResponse(
                claim=text[:100],
                verdict=Verdict.INCONCLUSIVO,
                confidence=0.0,
                summary="Nenhum analisador ativo configurado.",
                reasons=["Pipeline de analisadores vazio."],
                sources=[]
            )

        results: list[AnalyzerResult] = []

        # 1. Heuristic Analyzer (Texto Bruto)
        heuristic_features: dict[str, Any] = {}
        if "heuristic" in active_map:
            h_res = await active_map["heuristic"].analyze(text, urls)
            results.append(h_res)
            heuristic_features = h_res.raw_details or {}

        # 2. Claim Extractor (spaCy Gatekeeper + LLM Decomposer)
        target_claim = text
        if "claim_extractor" in active_map:
            c_res = await active_map["claim_extractor"].analyze(text, urls)
            results.append(c_res)
            if c_res.claim:
                target_claim = c_res.claim
            else:
                # Se gatekeeper descartou por ser ruído ou opinião pura, encerra o pipeline
                assertions = (c_res.raw_details or {}).get("assertions", [])
                if not assertions:
                    return AnalyzeResponse(
                        claim=text[:120],
                        verdict=Verdict.INCONCLUSIVO,
                        confidence=0.0,
                        summary="Nenhuma alegação factual identificada (texto classificado como ruído, saudação ou mera opinião).",
                        reasons=c_res.reasons or ["Sem predicado factual falseável."],
                        sources=c_res.sources
                    )

        # 3. Fact-Check API & Leitura Horizontal (com a claim isolada)
        evidences: list[dict[str, Any]] = []
        if "fact_check_api" in active_map:
            fc_res = await active_map["fact_check_api"].analyze(target_claim, urls)
            results.append(fc_res)
            evidences = (fc_res.raw_details or {}).get("evidences", [])

        # 4. LLM Judge (com contexto de evidências e métricas)
        if "llm_judge" in active_map:
            judge = active_map["llm_judge"]
            if hasattr(judge, "analyze_with_context"):
                j_res = await judge.analyze_with_context(
                    target_claim, urls, evidences=evidences, heuristic_features=heuristic_features
                )
            else:
                j_res = await judge.analyze(target_claim, urls)
            results.append(j_res)

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

