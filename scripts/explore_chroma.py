#!/usr/bin/env python3
"""
Utilitário CLI interativo para explorar e inspecionar a base vetorial ChromaDB do FactChkBR.

Uso:
    python scripts/explore_chroma.py stats
    python scripts/explore_chroma.py list [--limit 10]
    python scripts/explore_chroma.py search "água com limão" [--limit 3]
    python scripts/explore_chroma.py show <claim_id>
    python scripts/explore_chroma.py (modo interativo)
"""

import argparse
import json
import sys
from pathlib import Path

# Adiciona o diretório raiz ao sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from app.config import get_settings
from app.services.vector_kb import get_vector_kb
from app.schemas.analysis import Verdict


def print_banner():
    print("=" * 76)
    print(" 🧭 EXPLORADOR DE DADOS - CHROMADB FACTCHKBR")
    print("=" * 76)


def cmd_stats(args, kb):
    stats = kb.get_stats()
    print_banner()
    print(f"📁 Diretório persistente: {stats['persist_directory']}")
    print(f"📦 Coleção ativa:        {stats['collection_name']}")
    print(f"📊 Total de alegações:   {stats['total_claims']}")
    print(f"🎯 Limiar de similaridade:{stats['similarity_threshold']}")
    print(f"⚙️  Busca ativada:        {stats['enabled']}")
    print("=" * 76)


def cmd_list(args, kb):
    limit = getattr(args, "limit", 20) or 20
    total = kb.count()
    print_banner()
    print(f"📋 Listando até {limit} alegações gravadas (Total no banco: {total}):\n")

    if total == 0:
        print("⚠️  A base vetorial está vazia no momento.")
        print("    Dica: execute uma checagem no pipeline ou use 'POST /api/v1/vector/index' para adicionar dados.")
        print("=" * 76)
        return

    # Recupera todos os registros da coleção
    res = kb.collection.get(limit=limit)
    ids = res.get("ids", [])
    documents = res.get("documents", [])
    metadatas = res.get("metadatas", [])

    v_icons = {
        "VERDADEIRO": "✅ VERDADEIRO",
        "FAKE": "❌ FAKE",
        "SUSPEITO": "⚠️ SUSPEITO",
        "INCONCLUSIVO": "❓ INCONCLUSIVO",
    }

    for idx, (doc_id, doc, meta) in enumerate(zip(ids, documents, metadatas), 1):
        v = meta.get("verdict", "INCONCLUSIVO")
        v_display = v_icons.get(v, v)
        conf = float(meta.get("confidence", 0.0)) * 100
        claim_type = meta.get("claim_type", "primary")
        created = meta.get("created_at", "")[:19].replace("T", " ")

        print(f"[{idx}] ID: {doc_id} | {v_display} ({conf:.0f}% confiança) | Tipo: {claim_type}")
        print(f"    📝 Alegação: \"{doc}\"")
        if meta.get("summary"):
            print(f"    💡 Resumo:   {meta['summary']}")

        sources = []
        if "sources_json" in meta:
            try:
                sources = json.loads(meta["sources_json"])
            except Exception:
                sources = []
        if sources:
            print(f"    🌐 Fontes ({len(sources)}): {sources[0]}")
            if len(sources) > 1:
                print(f"                (+ {len(sources)-1} outra(s))")
        print(f"    🕒 Data:     {created}")
        print("-" * 76)


def cmd_search(args, kb):
    query = args.query
    limit = getattr(args, "limit", 3) or 3
    min_sim = getattr(args, "min_sim", None)

    print_banner()
    print(f"🔍 Buscando por similaridade vetorial para: \"{query}\"\n")

    matches = kb.search_claims(query=query, limit=limit, min_similarity=min_sim)

    if not matches:
        print("❌ Nenhuma alegação com similaridade semântica encontrada.")
        print(f"   (Limiar aplicado: {min_sim if min_sim is not None else kb.min_similarity})")
        print("=" * 76)
        return

    v_icons = {
        Verdict.VERDADEIRO: "✅ VERDADEIRO",
        Verdict.FAKE: "❌ FAKE",
        Verdict.SUSPEITO: "⚠️ SUSPEITO",
        Verdict.INCONCLUSIVO: "❓ INCONCLUSIVO",
    }

    print(f"🎯 {len(matches)} resultado(s) recuperado(s):\n")
    for idx, m in enumerate(matches, 1):
        v_display = v_icons.get(m.verdict, str(m.verdict))
        sim_pct = (m.similarity or 0.0) * 100
        dist = m.distance or 0.0

        print(f"[{idx}] Similaridade: {sim_pct:.1f}% | Distância Cosseno: {dist:.4f}")
        print(f"    ID:       {m.id}")
        print(f"    Veredito: {v_display} ({m.confidence * 100:.0f}% certeza)")
        print(f"    Alegação: \"{m.statement}\"")
        if m.summary:
            print(f"    Resumo:   {m.summary}")
        if m.reasons:
            print(f"    Motivos:  • " + "\n              • ".join(m.reasons[:2]))
        if m.sources:
            print(f"    Fontes:   • " + "\n              • ".join(m.sources[:2]))
        print("-" * 76)


def cmd_show(args, kb):
    claim_id = args.claim_id
    item = kb.get_claim(claim_id)

    print_banner()
    if not item:
        print(f"❌ Alegação com ID '{claim_id}' não foi encontrada na base vetorial.")
        print("=" * 76)
        return

    print(f"🆔 ID:          {item.id}")
    print(f"📝 Alegação:    \"{item.statement}\"")
    print(f"⚖️  Veredito:    {item.verdict.value}")
    print(f"📊 Confiança:   {item.confidence * 100:.1f}%")
    print(f"🏷️  Categoria:   {item.category or 'Geral'}")
    print(f"🕒 Indexado em: {item.created_at}")
    print(f"\n💡 Resumo da Checagem:\n   {item.summary or 'Sem resumo.'}\n")

    if item.reasons:
        print(f"🔍 Justificativas e Fundamentações ({len(item.reasons)}):")
        for r in item.reasons:
            print(f"   • {r}")
        print()

    if item.sources:
        print(f"🌐 Fontes & Links ({len(item.sources)}):")
        for s in item.sources:
            print(f"   • {s}")
        print()

    print("=" * 76)


def cmd_dedup(args, kb):
    print_banner()
    print("🧹 Iniciando processo de unificação e deduplicação da base vetorial...\n")
    res = kb.deduplicate_collection()
    print(f"📊 Registros antes:      {res['total_before']}")
    print(f"✅ Registros atuais:     {res['total_after']}")
    print(f"🔄 Unificados/Mesclados: {res['merged']}")
    if res['removed_ids']:
        print(f"🗑️  IDs obsoletos removidos: {', '.join(res['removed_ids'])}")
    print("=" * 76)


def interactive_mode(kb):
    print_banner()
    print("Modo Interativo iniciado. Digite comandos ou termos para buscar:")
    print("  • Digite qualquer texto para fazer busca semântica instantânea")
    print("  • 'list'  -> lista os registros gravados")
    print("  • 'stats' -> estatísticas gerais do ChromaDB")
    print("  • 'dedup' -> unifica registros duplicados")
    print("  • 'show <id>' -> detalha um item por ID")
    print("  • 'exit' ou 'sair' -> encerra")
    print("=" * 76)

    while True:
        try:
            line = input("\n🧭 ChromaDB > ").strip()
            if not line:
                continue
            if line.lower() in ("exit", "sair", "quit", "q"):
                print("Encerrando explorador.")
                break
            elif line.lower() == "stats":
                cmd_stats(None, kb)
            elif line.lower() == "list":
                cmd_list(argparse.Namespace(limit=15), kb)
            elif line.lower() == "dedup":
                cmd_dedup(None, kb)
            elif line.lower().startswith("show "):
                cid = line.split(" ", 1)[1].strip()
                cmd_show(argparse.Namespace(claim_id=cid), kb)
            else:
                cmd_search(argparse.Namespace(query=line, limit=3, min_sim=0.0), kb)
        except (KeyboardInterrupt, EOFError):
            print("\nEncerrando explorador.")
            break


def main():
    parser = argparse.ArgumentParser(description="Explorador de Dados ChromaDB - FactChkBR")
    subparsers = parser.add_subparsers(dest="command")

    # Subcomando stats
    subparsers.add_parser("stats", help="Exibe estatísticas da coleção vetorial")

    # Subcomando list
    p_list = subparsers.add_parser("list", help="Lista alegações armazenadas")
    p_list.add_argument("--limit", type=int, default=15, help="Quantidade máxima de itens a listar")

    # Subcomando dedup
    subparsers.add_parser("dedup", help="Varre a base e unifica alegações duplicadas")

    # Subcomando search
    p_search = subparsers.add_parser("search", help="Busca alegações por proximidade semântica")
    p_search.add_argument("query", type=str, help="Texto da alegação a buscar")
    p_search.add_argument("--limit", type=int, default=3, help="Número de resultados")
    p_search.add_argument("--min-sim", type=float, default=None, help="Limiar mínimo de similaridade")

    # Subcomando show
    p_show = subparsers.add_parser("show", help="Mostra detalhes de uma alegação por ID")
    p_show.add_argument("claim_id", type=str, help="Hash/ID da alegação")

    args = parser.parse_args()
    kb = get_vector_kb()

    if args.command == "stats":
        cmd_stats(args, kb)
    elif args.command == "list":
        cmd_list(args, kb)
    elif args.command == "dedup":
        cmd_dedup(args, kb)
    elif args.command == "search":
        cmd_search(args, kb)
    elif args.command == "show":
        cmd_show(args, kb)
    else:
        interactive_mode(kb)


if __name__ == "__main__":
    main()
