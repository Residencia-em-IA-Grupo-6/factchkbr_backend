#!/usr/bin/env python3
"""
FactChkBR - CLI de Checagem de Fatos
Executa o pipeline completo: Heurística -> Extrator (spaCy + LLM) -> Fact-Check API & Leitura Horizontal -> LLM Judge
"""

import asyncio
import select
import sys
from pathlib import Path

# Garante inclusão da raiz do repositório no path de importação
_root = str(Path(__file__).resolve().parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

from app.core.orchestrator import FactCheckOrchestrator
from app.schemas.analysis import AnalyzeResponse, Verdict


def format_cli_result(res: AnalyzeResponse) -> None:
    """Exibe no terminal um relatório legível e estruturado do veredito."""
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


async def main() -> None:
    orchestrator = FactCheckOrchestrator()
    active = [a.name for a in orchestrator.get_active_analyzers()]

    print("\n" + "=" * 78)
    print("🔎 FactChkBR - Pipeline de Checagem Factual")
    print(f"⚙️  Módulos ativos: {' -> '.join(active)}")
    print("=" * 78)

    # 1. Se passou o texto diretamente como argumento via terminal:
    if len(sys.argv) > 1:
        raw_text = " ".join(sys.argv[1:])
        print(f"\n📥 Analisando: \"{raw_text}\"")
        print("⏳ Executando analisadores (Heurística -> Extrator -> Fact-Check API -> LLM Judge)...")
        res = await orchestrator.analyze(raw_text, [])
        format_cli_result(res)
        return

    # 2. Modo interativo contínuo
    print("\nCole qualquer notícia ou mensagem abaixo e pressione ENTER para verificar.")
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


if __name__ == "__main__":
    asyncio.run(main())
