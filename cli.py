#!/usr/bin/env python3
"""
FactChkBR - CLI de Checagem de Fatos com Rastreabilidade em Tempo Real
Executa e monitora o pipeline:
1. Heurística Estilística e Sensacionalismo (CPU/Regex)
2. Extrator Semântico e Gatekeeper (spaCy + LLM Structured Outputs)
3. Fact-Check API & Leitura Horizontal (Google Fact Check Tools + RSS + IBGE/Ipea)
4. LLM Judge Contextual (LLM local Ollama / OpenAI)
"""

import asyncio
import datetime
import json
import logging
import select
import sys
import time
from pathlib import Path

# Garante inclusão da raiz do repositório no path de importação
_root = str(Path(__file__).resolve().parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

from app.core.orchestrator import FactCheckOrchestrator
from app.schemas.analysis import AnalyzeResponse, AnalyzerResult, Verdict

# Configuração de logging base
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)


def print_step_trace(step_name: str, result: AnalyzerResult, duration: float, raw_mode: bool = False) -> None:
    """Imprime em tempo real o log visual e dados técnicos da etapa finalizada."""
    now_str = datetime.datetime.now().strftime("%H:%M:%S")
    raw = result.raw_details or {}

    if step_name == "health_gatekeeper":
        is_health = raw.get("is_health_topic", False)
        is_pol = raw.get("is_political_polemic", False)
        allows = raw.get("allows_verification", False)
        cat = raw.get("category", "OUT_OF_SCOPE")
        signals = raw.get("matched_signals", [])
        detected_topic = raw.get("detected_topic")
        topic_conf = raw.get("topic_confidence")

        status_icon = "🩺" if allows else "🛑"
        print(f"\n[{now_str}] ── [0/4] {status_icon} FILTRO TEMÁTICO DE SAÚDE ({duration:.3f}s) " + "─" * 29)
        status_str = "APROVADO (Tema de Saúde Válido)" if allows else "BLOQUEADO (Fora do Escopo)"
        print(f"      • Status do Escopo:      {status_str} [Categoria: {cat}]")
        if detected_topic:
            conf_str = f" ({topic_conf * 100:.1f}%)" if topic_conf is not None else ""
            print(f"      • Tema Classificado:     {detected_topic}{conf_str}")
        if signals:
            print(f"      • Sinais Detectados:     {', '.join(signals)}")
        print(f"      • Justificativa:         {result.reasons[0] if result.reasons else result.summary}")

    elif step_name == "heuristic":
        score = raw.get("composite_sensationalism_score", 0.0)
        risk = raw.get("risk_level", "DESCONHECIDO")
        raw_feats = raw.get("raw_features", {})
        flagged = raw.get("flagged_tokens", [])

        print(f"\n[{now_str}] ── [1/4] 🧠 HEURÍSTICA ESTILÍSTICA ({duration:.3f}s) " + "─" * 38)
        print(f"      • Score Sensacionalismo: {score:.2f} (Risco: {risk})")
        print(
            f"      • Métricas Formais:      CAPS: {raw_feats.get('all_caps_count', 0)} | "
            f"Pontuação repetida: {raw_feats.get('repeated_punctuation_count', 0)} | "
            f"Emojis de alerta: {raw_feats.get('alarm_emoji_count', 0)}"
        )
        if flagged:
            print(f"      • Termos Alarmistas:     {', '.join(flagged[:6])}")
        print(f"      • Diagnóstico:           {result.summary}")

    elif step_name == "claim_extractor":
        engine = raw.get("engine_used", "desconhecido")
        assertions = raw.get("assertions", [])
        entities = raw.get("entities", {})
        contract = raw.get("contract", {})
        discarded = contract.get("discarded_fragments", [])

        print(f"\n[{now_str}] ── [2/4] 🔬 EXTRATOR SEMÂNTICO DE FATOS ({duration:.3f}s | {engine}) " + "─" * 20)
        
        ent_strs = []
        for cat, items in entities.items():
            if items:
                ent_strs.append(f"{cat}: {items}")
        if ent_strs:
            print(f"      • Entidades (NER):       {' | '.join(ent_strs)}")

        if assertions:
            print(f"      • Proposições Atômicas ({len(assertions)}):")
            for a in assertions:
                idx = a.get("id", "-")
                stmt = a.get("statement", "")
                cat = a.get("category", "FACTUAL_CLAIM")
                cat_str = cat.value if hasattr(cat, "value") else str(cat).split(".")[-1]
                triple = a.get("triple", {})
                s = triple.get("subject", "?")
                p = triple.get("predicate", "?")
                o = triple.get("object", "?")
                worthy = "checável" if a.get("is_check_worthy", True) else "não-checável"
                print(f"        [{idx}] \"{stmt}\" [{cat_str} | {worthy}]")
                print(f"            └─ Tripla (SPO): ({s} ➔ {p} ➔ {o})")

        if discarded:
            print(f"      • Ruídos Descartados:    {', '.join(repr(d) for d in discarded[:3])}")

        if result.claim:
            print(f"      🎯 Alegação Principal:   \"{result.claim}\"")
        else:
            print("      ⚠️ Nenhuma alegação factual identificada (ruído/opinião)")

    elif step_name == "fact_check_api":
        evidences = raw.get("evidences", [])
        fc_count = raw.get("google_fact_check_count", 0)
        lat_count = raw.get("lateral_reading_count", 0)
        is_quant = raw.get("is_quantitative", False)
        v_str = result.verdict.value if result.verdict else "INCONCLUSIVO"
        is_cached = raw.get("vector_cache_hit", False)

        if is_cached:
            sim_val = raw.get("vector_similarity", 1.0) * 100
            print(f"\n[{now_str}] ── [3/4] ⚡ BASE VETORIAL (MEMÓRIA PERSISTENTE) ({duration:.3f}s) " + "─" * 15)
            print("      • Status da Alegação:    🎯 JÁ ANALISADA PREVIAMENTE (Buscas externas dispensadas)")
            print(f"      • Similaridade Vetorial: {sim_val:.1f}% com checagem histórica consolidada")
        else:
            print(f"\n[{now_str}] ── [3/4] 🌐 BUSCA DE EVIDÊNCIAS EXTERNAS ({duration:.3f}s) " + "─" * 26)
            print(f"      • Escopos Consultados:   Google Fact Check Tools ({fc_count}) + Leitura Horizontal ({lat_count})")
            if is_quant:
                print(f"      • Filtro Quantitativo:   Ativo (sobreposição temática com repositórios oficiais)")

        if evidences:
            print(f"      • Evidências Recuperadas ({len(evidences)}):")
            for i, ev in enumerate(evidences[:4], 1):
                src = ev.get("source_name", "Fonte")
                title = ev.get("title", "")
                rating = ev.get("rating")
                rating_str = f" [Classificação: {rating}]" if rating else ""
                print(f"        {i}. [{src}] \"{title}\"{rating_str}")
            if len(evidences) > 4:
                print(f"        ... e mais {len(evidences) - 4} registro(s) adicional(is)")
        else:
            print("      • Evidências:            Nenhum registro encontrado nas fontes externas.")

        sub_claims = raw.get("sub_claims", [])
        if len(sub_claims) > 1:
            print(f"      • Checagem Isolada de Proposições ({len(sub_claims)}):")
            for idx, sc in enumerate(sub_claims, 1):
                sc_v = sc.get("verdict")
                sc_v_str = sc_v.value if hasattr(sc_v, "value") else str(sc_v)
                sc_conf = sc.get("confidence", 0.0) * 100
                print(f"        [{idx}] \"{sc.get('statement')}\" ➔ {sc_v_str} ({sc_conf:.1f}%)")

        print(f"      • Veredito Preliminar:   {v_str} (Confiança: {result.confidence * 100:.1f}%)")

    elif step_name == "llm_judge":
        model = raw.get("model", "desconhecido")
        provider = raw.get("provider", "desconhecido")
        v_str = result.verdict.value if result.verdict else "INCONCLUSIVO"

        print(f"\n[{now_str}] ── [4/4] ⚖️ JULGAMENTO CONTEXTUAL (LLM JUDGE) ({duration:.3f}s | {model} via {provider}) " + "─" * 12)
        print(f"      • Decisão do Juiz:       {v_str} (Confiança: {result.confidence * 100:.1f}%)")
        if raw.get("polarity_corrected"):
            print("      • Salvaguarda Semântica: 🔄 Polaridade corrigida para alinhamento com a fundamentação analítica")
        print(f"      • Síntese Analítica:     {result.summary}")
        if result.reasons:
            print(f"      • Fundamentos:")
            for r in result.reasons[:3]:
                print(f"        - {r}")

    if raw_mode:
        print(f"\n      [RAW DUMP - {step_name}]:")
        print(json.dumps(raw, indent=6, ensure_ascii=False))


def format_cli_result(res: AnalyzeResponse, total_duration: float | None = None) -> None:
    """Exibe no terminal o relatório consolidado final do pipeline."""
    v_icons = {
        Verdict.VERDADEIRO: "✅ VERDADEIRO",
        Verdict.FAKE: "❌ FAKE",
        Verdict.SUSPEITO: "⚠️ SUSPEITO",
        Verdict.INCONCLUSIVO: "❓ INCONCLUSIVO",
    }
    verdict_display = v_icons.get(res.verdict, res.verdict.value)

    print("\n" + "=" * 78)
    print("⚖️  FACTCHKBR - RESULTADO CONSOLIDADO DO PIPELINE")
    if total_duration:
        print(f"⏱️  Tempo total de execução: {total_duration:.2f}s")
    print("=" * 78)
    print(f"🎯 ALEGAÇÃO ISOLADA (CLAIM):\n   👉 \"{res.claim}\"\n")
    print(f"⚖️  VEREDITO CONSOLIDADO: {verdict_display}")
    print(f"📊 GRAU DE CONFIANÇA:    {res.confidence * 100:.1f}%\n")
    print(f"📝 RESUMO EXPLICATIVO:\n   {res.summary}\n")

    if res.sub_claims:
        print(f"🧬 AVALIAÇÃO DISCRIMINADA POR ALEGAÇÃO ({len(res.sub_claims)}):")
        for i, sc in enumerate(res.sub_claims, 1):
            badge = v_icons.get(sc.verdict, sc.verdict.value)
            print(f"   [{i}] {badge} ({sc.confidence * 100:.1f}%): \"{sc.statement}\"")
            print(f"       └─ Justificativa: {sc.justification}")
            if sc.sources:
                print(f"       └─ Fontes:        {', '.join(sc.sources[:2])}")
        print()

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


async def run_pipeline(text: str, orchestrator: FactCheckOrchestrator, raw_mode: bool = False) -> None:
    """Executa a verificação completa com callback de rastreabilidade passo a passo."""
    print(f"\n📥 Entrada recebida: \"{text}\"")
    print("⏳ Iniciando execução das etapas do pipeline em sequência...")

    def on_step(step_name: str, result: AnalyzerResult, duration: float) -> None:
        print_step_trace(step_name, result, duration, raw_mode=raw_mode)

    t_start = time.perf_counter()
    response = await orchestrator.analyze(text, [], on_step=on_step)
    t_total = time.perf_counter() - t_start

    format_cli_result(response, total_duration=t_total)


async def main() -> None:
    raw_mode = "--raw" in sys.argv
    debug_mode = "--debug" in sys.argv or "-d" in sys.argv

    if debug_mode:
        logging.getLogger("factchkbr").setLevel(logging.DEBUG)
        logging.getLogger("httpx").setLevel(logging.INFO)

    orchestrator = FactCheckOrchestrator()
    active = [a.name for a in orchestrator.get_active_analyzers()]
    modules_display = (["health_gatekeeper"] + active) if orchestrator.settings.HEALTH_ONLY_MODE else active

    print("\n" + "=" * 78)
    print("🔎 FactChkBR - Pipeline de Checagem Factual com Rastreabilidade")
    print(f"⚙️  Módulos ativos: {' ➔ '.join(modules_display)}")
    if raw_mode:
        print("🔧 Modo RAW JSON ativo: detalhes internos serão exibidos.")
    if debug_mode:
        print("🐞 Modo DEBUG ativo: logs de rede e subsistemas habilitados.")
    print("=" * 78)

    # Filtra flags da linha de comando para obter o texto de entrada
    args = [arg for arg in sys.argv[1:] if not arg.startswith("-")]

    # 1. Se passou o texto diretamente como argumento via terminal:
    if args:
        raw_text = " ".join(args)
        await run_pipeline(raw_text, orchestrator, raw_mode=raw_mode)
        return

    # 2. Modo interativo contínuo
    print("\nCole qualquer notícia ou mensagem abaixo e pressione ENTER para verificar.")
    print("Opções: digite 'sair' para encerrar | adicione --raw nos argumentos para dumps completos.\n")

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
            await run_pipeline(full_text, orchestrator, raw_mode=raw_mode)

        except (KeyboardInterrupt, EOFError):
            print("\nSessão encerrada.")
            break


if __name__ == "__main__":
    asyncio.run(main())
