import asyncio
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable

# Permite execução direta via `python app/core/orchestrator.py`
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from app.config import Settings, get_settings
from app.core.base import BaseAnalyzer
from app.core.registry import registry
from app.schemas.analysis import (
    AnalyzeResponse,
    AnalyzerResult,
    SubClaimAnalysis,
    Verdict,
)

logger = logging.getLogger("factchkbr.core.orchestrator")


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
        self._cached_analyzers: dict[str, BaseAnalyzer] = {}

    def get_active_analyzers(self) -> list[BaseAnalyzer]:
        """
        Instancia e retorna os analisadores ativos configurados no ACTIVE_ANALYZERS,
        mantendo instâncias em cache para evitar reinicializações e vazamento de conexões.
        """
        registry.auto_discover("app.analyzers")
        active_names = self.settings.get_active_analyzers_list()

        analyzers: list[BaseAnalyzer] = []
        for name in active_names:
            if name not in self._cached_analyzers:
                analyzer_cls = registry.get(name)
                if analyzer_cls:
                    self._cached_analyzers[name] = analyzer_cls()
            if name in self._cached_analyzers:
                analyzers.append(self._cached_analyzers[name])
        return analyzers

    async def aclose(self) -> None:
        """Encerra conexões HTTP e recursos assíncronos dos analisadores em cache."""
        for name, analyzer in list(self._cached_analyzers.items()):
            if hasattr(analyzer, "aclose") and callable(analyzer.aclose):
                try:
                    await analyzer.aclose()
                except Exception as e:
                    logger.warning("Erro ao fechar analisador %s: %s", name, e)
            elif hasattr(analyzer, "http_client") and hasattr(analyzer.http_client, "aclose"):
                try:
                    await analyzer.http_client.aclose()
                except Exception as e:
                    logger.warning("Erro ao fechar cliente HTTP de %s: %s", name, e)
        self._cached_analyzers.clear()

    async def __aenter__(self) -> "FactCheckOrchestrator":
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.aclose()

    async def analyze(
        self,
        text: str,
        urls: list[str],
        on_step: Callable[[str, AnalyzerResult, float], Any] | None = None,
    ) -> AnalyzeResponse:
        """
        Executa o pipeline linear contextual com rastreabilidade de etapas:
        1. Heurística (texto bruto, captura estilo e sensacionalismo) [Paralelo]
        2. Extrator de Alegações (spaCy + LLM, isola fatos atômicos e faz gatekeeping) [Paralelo]
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

        async def _run_stage(name: str, coro: Any) -> tuple[str, AnalyzerResult, float]:
            t0 = time.perf_counter()
            res = await coro
            dur = time.perf_counter() - t0
            return name, res, dur

        # Dispara heurística e extrator em paralelo (ambos processam o texto original)
        h_task = None
        c_task = None
        if "heuristic" in active_map:
            h_task = asyncio.create_task(_run_stage("heuristic", active_map["heuristic"].analyze(text, urls)))
        if "claim_extractor" in active_map:
            c_task = asyncio.create_task(_run_stage("claim_extractor", active_map["claim_extractor"].analyze(text, urls)))

        # 1. Heuristic Analyzer (Texto Bruto)
        heuristic_features: dict[str, Any] = {}
        if h_task:
            _, h_res, dur = await h_task
            results.append(h_res)
            heuristic_features = h_res.raw_details or {}
            logger.info("Etapa 'heuristic' concluída em %.3fs", dur)
            if on_step:
                cb = on_step("heuristic", h_res, dur)
                if asyncio.iscoroutine(cb):
                    await cb

        # 2. Claim Extractor (spaCy Gatekeeper + LLM Decomposer)
        target_claim = text
        check_worthy_stmts: list[str] = []
        if c_task:
            _, c_res, dur = await c_task
            results.append(c_res)
            logger.info("Etapa 'claim_extractor' concluída em %.3fs. Claim: '%s'", dur, c_res.claim)
            if on_step:
                cb = on_step("claim_extractor", c_res, dur)
                if asyncio.iscoroutine(cb):
                    await cb

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

            # Extrai proposições checáveis para checagem isolada
            raw_assertions = (c_res.raw_details or {}).get("assertions", [])
            for a in raw_assertions:
                if isinstance(a, dict):
                    if a.get("is_check_worthy", True) and a.get("statement"):
                        check_worthy_stmts.append(a["statement"])

        # 3. Fact-Check API & Leitura Horizontal (com a claim isolada e proposições atômicas)
        evidences: list[dict[str, Any]] = []
        fc_sub_claims: list[dict[str, Any]] = []
        if "fact_check_api" in active_map:
            t0 = time.perf_counter()
            fc_res = await active_map["fact_check_api"].analyze(target_claim, urls, assertions=check_worthy_stmts)
            dur = time.perf_counter() - t0
            results.append(fc_res)
            evidences = (fc_res.raw_details or {}).get("evidences", [])
            fc_sub_claims = (fc_res.raw_details or {}).get("sub_claims", [])
            logger.info("Etapa 'fact_check_api' concluída em %.3fs. %d evidência(s)", dur, len(evidences))
            if on_step:
                cb = on_step("fact_check_api", fc_res, dur)
                if asyncio.iscoroutine(cb):
                    await cb

        # 4. LLM Judge (com contexto de evidências, métricas e proposições isoladas)
        if "llm_judge" in active_map:
            t0 = time.perf_counter()
            judge = active_map["llm_judge"]
            if hasattr(judge, "analyze_with_context"):
                j_res = await judge.analyze_with_context(
                    target_claim,
                    urls,
                    evidences=evidences,
                    heuristic_features=heuristic_features,
                    sub_claims=fc_sub_claims,
                )
            else:
                j_res = await judge.analyze(target_claim, urls)
            dur = time.perf_counter() - t0
            results.append(j_res)
            logger.info("Etapa 'llm_judge' concluída em %.3fs. Veredito: %s", dur, j_res.verdict)
            if on_step:
                cb = on_step("llm_judge", j_res, dur)
                if asyncio.iscoroutine(cb):
                    await cb

        return self._consolidate(text, results, target_claim=target_claim)

    def _consolidate(
        self,
        text: str,
        results: list[AnalyzerResult],
        target_claim: str | None = None,
    ) -> AnalyzeResponse:
        """
        Consolida os resultados individuais através de ensemble ponderado / consenso.
        Filtra apenas analisadores que emitiram veredito concreto.
        """
        all_reasons: list[str] = []
        all_sources: list[str] = []
        primary_claim: str | None = target_claim

        # Coleta todas as razões, fontes e alegação isolada
        for r in results:
            all_reasons.extend(r.reasons)
            all_sources.extend(r.sources)
            if not primary_claim and r.claim:
                primary_claim = r.claim

        # Filtra previamente itens repetidos de sub-alegações para reconstrução limpa e consolidada
        filtered_reasons = [
            r for r in all_reasons
            if not re.match(r"^\[Alegação \d+ - [^\]]+\]", r, re.IGNORECASE)
        ]

        # Filtra apenas os analisadores que emitem veredito
        verdict_bearing_results = [r for r in results if r.verdict is not None]

        # Consolidar sub_claims discriminadas por alegação
        fc_res = next((r for r in verdict_bearing_results if r.analyzer_name == "fact_check_api"), None)
        judge_res = next((r for r in verdict_bearing_results if r.analyzer_name == "llm_judge"), None)

        fc_sub_claims = (fc_res.raw_details or {}).get("sub_claims", []) if fc_res else []
        judge_sub_claims = (judge_res.raw_details or {}).get("sub_claims", []) if judge_res else []

        # Salvaguarda epistemológica: Ausência de evidência não é evidência de falsidade.
        # Verifica se há desmentido factual comprovado nas evidências globais ou sub-alegações
        has_debunk = False
        if fc_res:
            evidences = (fc_res.raw_details or {}).get("evidences", [])
            has_debunk = (
                fc_res.verdict == Verdict.FAKE
                or any(
                    e.get("rating") in ("Falso", "Fake", "Mentira", "Desmentido")
                    or e.get("stance") == "REFUTES"
                    for e in evidences
                )
                or any(
                    sc.get("verdict") in (Verdict.FAKE, "FAKE")
                    for sc in fc_sub_claims
                )
            )

        consolidated_sub_claims: list[SubClaimAnalysis] = []
        base_sub_claims = judge_sub_claims if len(judge_sub_claims) >= len(fc_sub_claims) and judge_sub_claims else fc_sub_claims

        if base_sub_claims:
            for idx, item in enumerate(base_sub_claims):
                stmt = item.get("statement", f"Alegação {idx+1}")
                # Encontra entrada correspondente em judge e fc
                j_match = next((j for j in judge_sub_claims if j.get("statement") == stmt), None)
                if not j_match and idx < len(judge_sub_claims):
                    j_match = judge_sub_claims[idx]

                f_match = next((f for f in fc_sub_claims if f.get("statement") == stmt), None)
                if not f_match and idx < len(fc_sub_claims):
                    f_match = fc_sub_claims[idx]

                # Determina veredito da sub-alegação
                sub_v = item.get("verdict", Verdict.INCONCLUSIVO)
                if j_match and j_match.get("verdict"):
                    sub_v = j_match["verdict"]
                elif f_match and f_match.get("verdict"):
                    sub_v = f_match["verdict"]

                if isinstance(sub_v, str):
                    sub_v = Verdict[sub_v.upper()] if sub_v.upper() in Verdict.__members__ else Verdict.INCONCLUSIVO

                # Verifica se a sub-alegação tem comprovação de desmentido em fontes factuais
                f_debunk = False
                if f_match:
                    f_v = f_match.get("verdict")
                    if isinstance(f_v, str):
                        f_v = Verdict[f_v.upper()] if f_v.upper() in Verdict.__members__ else Verdict.INCONCLUSIVO
                    f_debunk = (f_v == Verdict.FAKE)
                    if not f_debunk and f_match.get("evidences"):
                        f_debunk = any(
                            e.get("rating") in ("Falso", "Fake", "Mentira", "Desmentido")
                            or e.get("stance") == "REFUTES"
                            for e in f_match.get("evidences", [])
                        )

                if len(base_sub_claims) == 1:
                    f_debunk = f_debunk or has_debunk

                # Determina confiança
                sub_c = 0.80
                if j_match and "confidence" in j_match:
                    sub_c = j_match["confidence"]
                elif f_match and "confidence" in f_match:
                    sub_c = f_match["confidence"]
                elif "confidence" in item:
                    sub_c = item["confidence"]

                # Determina justificativa pontual
                sub_just = ""
                if j_match and j_match.get("justification"):
                    sub_just = j_match["justification"]
                elif f_match and f_match.get("justification"):
                    sub_just = f_match["justification"]
                else:
                    sub_just = item.get("justification", "Avaliação individual.")

                # Salvaguarda epistemológica no nível de cada proposição:
                # Se o avaliador apontou FAKE mas não há evidência/desmentido factual, reverte para INCONCLUSIVO
                if sub_v == Verdict.FAKE and not f_debunk:
                    sub_v = Verdict.INCONCLUSIVO
                    sub_c = 0.55
                    sub_just = (
                        f"{sub_just.rstrip('.')} — Ausência de referências comprobatórias de falsidade; "
                        f"classificado como inconclusivo por carência de dados ou fato recente."
                    )

                # Determina fontes
                sub_sources = []
                if f_match and f_match.get("sources"):
                    sub_sources = f_match["sources"]
                elif item.get("sources"):
                    sub_sources = item["sources"]

                consolidated_sub_claims.append(
                    SubClaimAnalysis(
                        statement=stmt,
                        verdict=sub_v,
                        confidence=round(sub_c, 2),
                        justification=sub_just,
                        sources=sub_sources,
                    )
                )

                sub_v_name = sub_v.value if hasattr(sub_v, "value") else str(sub_v)
                filtered_reasons.append(f"[Alegação {idx+1} - {sub_v_name}]: \"{stmt}\" ➔ {sub_just}")

        if verdict_bearing_results:
            conclusive = [r for r in verdict_bearing_results if r.verdict != Verdict.INCONCLUSIVO]
            if conclusive:
                best_result = max(conclusive, key=lambda r: r.confidence)
                if best_result.verdict == Verdict.FAKE and not has_debunk:
                    dominant_verdict = Verdict.INCONCLUSIVO
                    final_confidence = 0.60
                    filtered_reasons.append(
                        "Ausência de referências comprobatórias de falsidade: a carência de dados ou matérias recentes impede a classificação como fake."
                    )
                else:
                    dominant_verdict = best_result.verdict
                    final_confidence = best_result.confidence

                # Salvaguarda contra falso-verdadeiro: checagens oficiais de desmentido prevalecem sobre alucinações de VERDADEIRO
                if has_debunk and fc_res and fc_res.verdict == Verdict.FAKE and dominant_verdict == Verdict.VERDADEIRO:
                    logger.warning("Conflito detectado: fact_check_api possui desmentido comprovado mas decisor apontou VERDADEIRO. Prevalecendo FAKE.")
                    dominant_verdict = Verdict.FAKE
                    final_confidence = max(fc_res.confidence, 0.85)
                    filtered_reasons.append(
                        "Prevalência de checagem oficial: fontes jornalísticas/IFCN de desmentido têm precedência probatória sobre confirmação divergente."
                    )
            else:
                dominant_verdict = Verdict.INCONCLUSIVO
                final_confidence = sum(r.confidence for r in verdict_bearing_results) / len(verdict_bearing_results)

            # Impacto composto no veredito se houver múltiplas sub-alegações consolidadas
            if len(consolidated_sub_claims) > 1:
                has_fake = any(sc.verdict == Verdict.FAKE for sc in consolidated_sub_claims)
                has_true = any(sc.verdict == Verdict.VERDADEIRO for sc in consolidated_sub_claims)
                has_suspect = any(sc.verdict == Verdict.SUSPEITO for sc in consolidated_sub_claims)
                has_inconclusive = any(sc.verdict == Verdict.INCONCLUSIVO for sc in consolidated_sub_claims)

                if has_fake and has_true:
                    dominant_verdict = Verdict.SUSPEITO
                    final_confidence = sum(sc.confidence for sc in consolidated_sub_claims) / len(consolidated_sub_claims)
                elif has_suspect:
                    dominant_verdict = Verdict.SUSPEITO
                    final_confidence = sum(sc.confidence for sc in consolidated_sub_claims) / len(consolidated_sub_claims)
                elif has_fake and not has_true:
                    if has_debunk:
                        dominant_verdict = Verdict.FAKE
                        final_confidence = max(sc.confidence for sc in consolidated_sub_claims if sc.verdict == Verdict.FAKE)
                    else:
                        dominant_verdict = Verdict.INCONCLUSIVO
                        final_confidence = 0.55
                        filtered_reasons.append(
                            "Ausência de referências comprobatórias de falsidade: a carência de dados ou matérias recentes impede a classificação como fake."
                        )
                elif has_true and not has_fake and not has_inconclusive:
                    dominant_verdict = Verdict.VERDADEIRO
                    final_confidence = sum(sc.confidence for sc in consolidated_sub_claims) / len(consolidated_sub_claims)
                elif has_true and has_inconclusive and not has_fake:
                    dominant_verdict = Verdict.SUSPEITO
                    final_confidence = sum(sc.confidence for sc in consolidated_sub_claims) / len(consolidated_sub_claims)
                    filtered_reasons.append(
                        "Texto misto: contém alegações confirmadas associadas a alegações sem confirmação factual disponível."
                    )
                else:
                    dominant_verdict = Verdict.INCONCLUSIVO
                    final_confidence = sum(sc.confidence for sc in consolidated_sub_claims) / len(consolidated_sub_claims)

            # Prioriza o resumo explicativo do LLM Judge se coerente com o veredito dominante, senão do fact_check_api
            if judge_res and judge_res.summary and not judge_res.summary.startswith("Avaliador LLM offline"):
                j_summary_lower = judge_res.summary.lower()
                if dominant_verdict == Verdict.FAKE and any(c in j_summary_lower for c in ("é verdadeira", "alegação é verdadeira", "fato verdadeiro")):
                    summary = fc_res.summary if (fc_res and fc_res.summary) else judge_res.summary
                elif dominant_verdict == Verdict.VERDADEIRO and any(d in j_summary_lower for d in ("é falsa", "foi desmentida", "foi refutada")):
                    summary = fc_res.summary if (fc_res and fc_res.summary) else judge_res.summary
                else:
                    summary = judge_res.summary
            elif fc_res and fc_res.summary:
                summary = fc_res.summary
            else:
                summary = f"Análise consolidada por {len(verdict_bearing_results)} modelo(s) decisor(es) com apoio de {len(results) - len(verdict_bearing_results)} módulo(s) de features."

            if dominant_verdict == Verdict.INCONCLUSIVO and not any(k in summary.lower() for k in ("recente", "insuficiente", "ausência", "falta de", "imprecis", "escassez")):
                summary += " Não há referências suficientes para confirmar nem refutar a afirmação (imprecisão por escassez de dados ou acontecimento recente)."
        else:
            dominant_verdict = Verdict.INCONCLUSIVO
            final_confidence = 0.50
            summary = "Extração de features concluída; nenhum modelo decisor emitiu veredito final."

        return AnalyzeResponse(
            claim=primary_claim or text[:120],
            verdict=dominant_verdict,
            confidence=round(final_confidence, 2),
            summary=summary,
            reasons=list(dict.fromkeys(filtered_reasons)),
            sources=list(dict.fromkeys(all_sources)),
            sub_claims=consolidated_sub_claims,
        )


def format_cli_result(res: AnalyzeResponse) -> None:
    """Imprime relatório estruturado e legível do pipeline no terminal."""
    v_icons = {
        Verdict.VERDADEIRO: "✅ VERDADEIRO",
        Verdict.FAKE: "❌ FAKE",
        Verdict.SUSPEITO: "⚠️ SUSPEITO",
        Verdict.INCONCLUSIVO: "❓ INCONCLUSIVO",
    }
    verdict_display = v_icons.get(res.verdict, res.verdict.value)

    print("\n" + "=" * 78)
    print("⚖️  FACTCHKBR - RESULTADO CONSOLIDADO DO PIPELINE")
    print("=" * 78)
    print(f"🎯 ALEGAÇÃO ISOLADA (CLAIM):\n   👉 \"{res.claim}\"\n")
    print(f"⚖️  VEREDITO CONSOLIDADO: {verdict_display}")
    print(f"📊 GRAU DE CONFIANÇA:    {res.confidence * 100:.1f}%\n")
    print(f"📝 RESUMO EXPLICATIVO:\n   {res.summary}\n")

    if res.reasons:
        print(f"🔍 RAZÕES APONTADAS ({len(res.reasons)}):")
        for r in res.reasons:
            print(f"   • {r}")
        print()

    if res.sources:
        print(f"🌐 FONTES & REFERÊNCIAS ({len(res.sources)}):")
        for s in res.sources:
            print(f"   • {s}")
        print()
    print("=" * 78 + "\n")


if __name__ == "__main__":
    import select
    import warnings

    warnings.filterwarnings("ignore", category=RuntimeWarning)

    async def _run_cli() -> None:
        orchestrator = FactCheckOrchestrator()
        active = [a.name for a in orchestrator.get_active_analyzers()]

        print("\n" + "=" * 78)
        print("🔎 FactChkBR - Pipeline Completo de Checagem de Fatos")
        print(f"⚙️  Módulos ativos: {' -> '.join(active)}")
        print("=" * 78)

        # 1. Se passou o texto diretamente como argumento via terminal
        if len(sys.argv) > 1:
            raw_text = " ".join(sys.argv[1:])
            print(f"\n📥 Processando: \"{raw_text}\"")
            print("⏳ Executando analisadores (Heurística -> Extrator -> Fact-Check API -> LLM Judge)...")
            res = await orchestrator.analyze(raw_text, [])
            format_cli_result(res)
            return

        # 2. Modo interativo contínuo
        print("\nCole qualquer notícia ou mensagem abaixo e pressione ENTER para analisar.")
        print("Digite 'sair' ou pressione Ctrl+C para encerrar.\n")

        while True:
            try:
                print("📥 Cole o texto a ser analisado:")
                first_line = input("> ").strip()
                if not first_line:
                    continue
                if first_line.lower() in ("sair", "exit", "quit", "q"):
                    print("Encerrando testador do FactChkBR.")
                    break

                lines = [first_line]
                try:
                    while select.select([sys.stdin], [], [], 0.05)[0]:
                        extra = sys.stdin.readline()
                        if not extra:
                            break
                        lines.append(extra.strip())
                except Exception:
                    pass

                full_text = " ".join(line for line in lines if line)
                print("\n⏳ Executando pipeline completo...")
                res = await orchestrator.analyze(full_text, [])
                format_cli_result(res)

            except (KeyboardInterrupt, EOFError):
                print("\nSessão encerrada.")
                break

    asyncio.run(_run_cli())



