#!/usr/bin/env python3
"""
CLI de Alta Performance para Geração de Dataset de Treinamento via Plumb-4B.

Processa arquivos TSV ou CSV contendo notícias/matérias jornalísticas,
transforma cada texto em alegações atômicas curtas e denotativas,
e realiza a classificação de Tópico e Fact Checking binário (V ou F)
utilizando exclusivamente o motor neural Plumb-4B via JevK5.

Uso:
    python scripts/generate_plumb_dataset.py -i dados/noticias.tsv -o dados/dataset_treino.tsv
    python scripts/generate_plumb_dataset.py -i noticias.csv --limit 50 --verbose
    python scripts/generate_plumb_dataset.py -i noticias.tsv --resume
"""

import argparse
import csv
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# Adiciona o diretório raiz ao sys.path
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from app.services.plumb_classifier import get_plumb_classifier
from app.services.plumb_dataset_generator import (
    ClaimCondenser,
    PlumbDatasetGenerator,
    PlumbDatasetRecord,
)

# Tenta importar rich para formatação visual premium
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn
    HAS_RICH = True
    console = Console()
except ImportError:
    HAS_RICH = False
    console = None


def setup_logger(log_file: Path | None, verbose: bool = False) -> logging.Logger:
    """Configura logging detalhado para arquivo e console."""
    logger = logging.getLogger("plumb_dataset_batch")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    # Formatter estruturado para o arquivo de log detalhado
    file_fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(str(log_file), encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(file_fmt)
        logger.addHandler(fh)

    # Console Handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG if verbose else logging.INFO)
    console_fmt = logging.Formatter("%(message)s")
    ch.setFormatter(console_fmt)
    logger.addHandler(ch)

    return logger


def print_banner(input_path: Path, output_path: Path, args: argparse.Namespace) -> None:
    """Exibe banner informativo no terminal."""
    if HAS_RICH and console:
        grid = Table.grid(expand=True)
        grid.add_column(style="bold cyan", justify="left")
        grid.add_column(style="white", justify="left")
        grid.add_row("📁 Arquivo de Entrada: ", str(input_path))
        grid.add_row("💾 Arquivo de Saída:   ", str(output_path))
        grid.add_row("🧠 Motor Neural:       ", "Plumb-4B (crh225/plumb-4b via JevK5)")
        grid.add_row("⚙️  Estratégia:         ", f"{args.strategy} (máx {args.max_claims_per_news} claim/notícia)")
        grid.add_row("🔄 Retomada (--resume):", "Ativada" if args.resume else "Desativada")
        if args.limit:
            grid.add_row("🔢 Limite de Notícias: ", str(args.limit))

        console.print(
            Panel(
                grid,
                title="[bold yellow]🚀 PLUMB-4B DATASET GENERATOR - FACTCHKBR[/bold yellow]",
                border_style="cyan",
            )
        )
    else:
        print("=" * 76)
        print(" 🚀 PLUMB-4B DATASET GENERATOR - FACTCHKBR")
        print("=" * 76)
        print(f"📁 Arquivo de Entrada: {input_path}")
        print(f"💾 Arquivo de Saída:   {output_path}")
        print("🧠 Motor Neural:       Plumb-4B (crh225/plumb-4b via JevK5)")
        print(f"⚙️  Estratégia:         {args.strategy} (máx {args.max_claims_per_news} claim/notícia)")
        print(f"🔄 Retomada (--resume): {'Ativada' if args.resume else 'Desativada'}")
        if args.limit:
            print(f"🔢 Limite:             {args.limit}")
        print("=" * 76)


def load_processed_ids(output_path: Path, delimiter: str = "\t") -> set[str]:
    """Lê os IDs já gravados no arquivo de saída para suportar retomada (--resume)."""
    if not output_path.exists():
        return set()

    processed: set[str] = set()
    try:
        with open(output_path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f, delimiter=delimiter)
            header = next(reader, None)
            if not header:
                return set()
            id_idx = 0
            if "id" in [h.lower() for h in header]:
                id_idx = [h.lower() for h in header].index("id")

            for row in reader:
                if row and len(row) > id_idx:
                    processed.add(str(row[id_idx]).strip())
    except Exception as exc:
        print(f"Aviso ao ler checkpoint anterior: {exc}")

    return processed


def main():
    parser = argparse.ArgumentParser(
        description="Gera dataset de treino com alegações curtas, tópicos e vereditos V/F via Plumb-4B.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Caminho do arquivo TSV ou CSV de entrada com notícias.",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Caminho do arquivo de saída (TSV, CSV ou JSONL). Se omitido, gera <input>_plumb_dataset.tsv.",
    )
    parser.add_argument(
        "--delimiter", "-d",
        type=str,
        default=None,
        help="Delimitador do arquivo de entrada (auto-detectado se omitido).",
    )
    parser.add_argument(
        "--out-delimiter",
        type=str,
        default="\t",
        help="Delimitador do arquivo de saída (padrão: TAB para evitar colisões com vírgulas em textos).",
    )
    parser.add_argument(
        "--text-col",
        type=str,
        default=None,
        help="Nome ou índice da coluna com o corpo da notícia (auto-detectado se omitido).",
    )
    parser.add_argument(
        "--title-col",
        type=str,
        default=None,
        help="Nome da coluna de título da notícia (opcional).",
    )
    parser.add_argument(
        "--id-col",
        type=str,
        default=None,
        help="Nome da coluna de identificador único (se omitido, usa índice sequencial).",
    )
    parser.add_argument(
        "--strategy", "-s",
        type=str,
        choices=["lead", "spacy", "sentences", "full"],
        default="lead",
        help="Estratégia para transformar a notícia em alegação curta: 'lead' (lide informativo conciso), 'spacy' (sentenças com predicado factual), 'sentences', 'full'.",
    )
    parser.add_argument(
        "--max-claims-per-news",
        type=int,
        default=1,
        help="Quantidade máxima de alegações extraídas por notícia.",
    )
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=None,
        help="Número máximo de notícias a processar.",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Número de notícias iniciais a pular.",
    )
    parser.add_argument(
        "--resume", "-r",
        action="store_true",
        help="Retoma de onde parou, pulando notícias cujos IDs já constam no arquivo de saída.",
    )
    parser.add_argument(
        "--batch-flush",
        type=int,
        default=1,
        help="Intervalo de registros para flush forçado em disco (1 = segurança máxima contra falhas).",
    )
    parser.add_argument(
        "--jsonl",
        action="store_true",
        help="Gera também um arquivo paralelo no formato JSONL (ideal para Hugging Face Datasets).",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Caminho do arquivo de log detalhado (padrão: logs/plumb_dataset_<data_hora>.log).",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Exibe detalhes estendidos no console durante a execução.",
    )

    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    if not input_path.exists():
        sys.exit(f"❌ Erro: Arquivo de entrada não encontrado: {input_path}")

    # Define arquivo de saída
    if args.output:
        output_path = Path(args.output).resolve()
    else:
        out_ext = ".tsv" if args.out_delimiter == "\t" else ".csv"
        output_path = input_path.with_name(f"{input_path.stem}_plumb_dataset{out_ext}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Define arquivo de log
    if args.log_file:
        log_file = Path(args.log_file).resolve()
    else:
        now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = _repo_root / "logs" / f"plumb_dataset_{now_str}.log"

    logger = setup_logger(log_file, verbose=args.verbose)
    print_banner(input_path, output_path, args)
    logger.info("Log detalhado iniciado em: %s", log_file)

    # 1. Carrega modelo Plumb-4B (Lazy loaded singleton)
    logger.info("Carregando e aquecendo modelo Plumb-4B...")
    classifier = get_plumb_classifier()
    agent = classifier._ensure_loaded()
    if agent is None:
        sys.exit("❌ Falha crítica: Não foi possível carregar o modelo Plumb-4B via JevK5.")
    logger.info("Modelo Plumb-4B pronto para inferência.")

    # 2. Inicializa o condensador de alegações
    condenser = ClaimCondenser()
    generator = PlumbDatasetGenerator(classifier=classifier, condenser=condenser)

    # 3. Detecta delimitador e cabeçalhos
    delimiter = args.delimiter or PlumbDatasetGenerator.detect_delimiter(input_path)
    logger.info("Delimitador detectado: %r", delimiter)

    # 4. Verifica IDs já processados se --resume
    processed_ids: set[str] = set()
    if args.resume and output_path.exists():
        processed_ids = load_processed_ids(output_path, delimiter=args.out_delimiter)
        logger.info("Modo --resume ativo: %d registros já encontrados no arquivo de saída.", len(processed_ids))

    # 5. Prepara arquivo de saída
    fieldnames = [
        "id",
        "noticia_original",
        "alegacao",
        "tema",
        "tema_confianca",
        "veredito",
        "veredito_confianca",
        "prob_v",
        "prob_f",
        "probabilidades_tema_json",
        "tempo_ms",
    ]

    file_exists = output_path.exists() and output_path.stat().st_size > 0
    out_file = open(output_path, "a" if args.resume else "w", encoding="utf-8", newline="")
    writer = csv.DictWriter(out_file, fieldnames=fieldnames, delimiter=args.out_delimiter)

    if not file_exists or not args.resume:
        writer.writeheader()
        out_file.flush()

    jsonl_file = None
    if args.jsonl:
        jsonl_path = output_path.with_suffix(".jsonl")
        jsonl_file = open(jsonl_path, "a" if args.resume else "w", encoding="utf-8")

    # 6. Leitura do arquivo de entrada e execução do pipeline
    stats = {
        "total_news_read": 0,
        "total_skipped_resume": 0,
        "total_claims_generated": 0,
        "topics": {},
        "verdicts": {"V": 0, "F": 0},
        "times": [],
    }

    t_start_global = time.perf_counter()

    try:
        with open(input_path, "r", encoding="utf-8", errors="replace") as in_f:
            reader = csv.DictReader(in_f, delimiter=delimiter)
            if not reader.fieldnames:
                sys.exit(f"❌ Erro: Arquivo {input_path} está vazio ou sem cabeçalhos.")

            text_col = PlumbDatasetGenerator.find_text_column(reader.fieldnames, args.text_col)
            title_col = args.title_col if (args.title_col and args.title_col in reader.fieldnames) else None
            id_col = args.id_col if (args.id_col and args.id_col in reader.fieldnames) else None

            logger.info("Coluna de texto selecionada: '%s'", text_col)
            if title_col:
                logger.info("Coluna de título selecionada: '%s'", title_col)
            if id_col:
                logger.info("Coluna de ID selecionada: '%s'", id_col)

            records_since_flush = 0

            for idx, row in enumerate(reader, start=1):
                if idx <= args.offset:
                    continue

                if args.limit and stats["total_news_read"] >= args.limit:
                    logger.info("Limite de %d notícias atingido.", args.limit)
                    break

                raw_id = str(row.get(id_col, idx)).strip() if id_col else str(idx)
                if args.resume and raw_id in processed_ids:
                    stats["total_skipped_resume"] += 1
                    continue

                raw_text = str(row.get(text_col, "")).strip()
                raw_title = str(row.get(title_col, "")).strip() if title_col else None

                if not raw_text and not raw_title:
                    logger.debug("Linha #%d ignorada: texto e título vazios.", idx)
                    continue

                stats["total_news_read"] += 1
                t0_item = time.perf_counter()

                # Processa item (extração de alegações + classificação com Plumb-4B)
                records = generator.process_item(
                    item_id=raw_id,
                    raw_text=raw_text,
                    title=raw_title,
                    strategy=args.strategy,
                    max_claims=args.max_claims_per_news,
                    extra_meta={k: v for k, v in row.items() if k not in (text_col, title_col, id_col)},
                )

                item_dur = time.perf_counter() - t0_item
                stats["times"].append(item_dur)

                for r in records:
                    stats["total_claims_generated"] += 1
                    stats["topics"][r.tema] = stats["topics"].get(r.tema, 0) + 1
                    stats["verdicts"][r.veredito] = stats["verdicts"].get(r.veredito, 0) + 1

                    row_dict = {
                        "id": r.id,
                        "noticia_original": r.noticia_original.replace("\n", " ").replace("\t", " "),
                        "alegacao": r.alegacao.replace("\n", " ").replace("\t", " "),
                        "tema": r.tema,
                        "tema_confianca": f"{r.tema_confianca:.4f}",
                        "veredito": r.veredito,
                        "veredito_confianca": f"{r.veredito_confianca:.4f}",
                        "prob_v": f"{r.prob_v:.4f}",
                        "prob_f": f"{r.prob_f:.4f}",
                        "probabilidades_tema_json": r.probabilidades_tema_json,
                        "tempo_ms": f"{r.tempo_processamento_ms:.1f}",
                    }
                    writer.writerow(row_dict)

                    if jsonl_file:
                        jsonl_file.write(json.dumps(r.model_dump(), ensure_ascii=False) + "\n")

                    records_since_flush += 1

                    # Log detalhado por item
                    v_icon = "✅ V" if r.veredito == "V" else "❌ F"
                    log_msg = (
                        f"[{datetime.now().strftime('%H:%M:%S')}] #{stats['total_news_read']} ID:{r.id} | "
                        f"🎯 Alegação: \"{r.alegacao[:70]}{'...' if len(r.alegacao)>70 else ''}\" | "
                        f"🏷️  {r.tema} ({r.tema_confianca*100:.1f}%) | "
                        f"⚖️  {v_icon} ({r.veredito_confianca*100:.1f}%) | ⏱️ {r.tempo_processamento_ms/1000:.2f}s"
                    )
                    logger.info(log_msg)

                    logger.debug(
                        "Detalhe ID %s: Probs V/F: [V=%.3f, F=%.3f] | Probs Tema: %s",
                        r.id, r.prob_v, r.prob_f, r.probabilidades_tema_json
                    )

                if records_since_flush >= args.batch_flush:
                    out_file.flush()
                    if jsonl_file:
                        jsonl_file.flush()
                    records_since_flush = 0

    except KeyboardInterrupt:
        logger.warning("\n⚠️ Processamento interrompido pelo usuário (Ctrl+C). Salvando checkpoint...")
    finally:
        out_file.flush()
        out_file.close()
        if jsonl_file:
            jsonl_file.flush()
            jsonl_file.close()

    # 7. Resumo Estatístico Final
    total_time = time.perf_counter() - t_start_global
    avg_time = (sum(stats["times"]) / len(stats["times"])) if stats["times"] else 0.0
    throughput = (stats["total_claims_generated"] / total_time) if total_time > 0 else 0.0

    print("\n" + "=" * 76)
    print(" 📊 RESUMO FINAL DA CLASSIFICAÇÃO COM PLUMB-4B")
    print("=" * 76)
    print(f"📰 Notícias processadas:      {stats['total_news_read']}")
    if stats['total_skipped_resume'] > 0:
        print(f"⏩ Notícias puladas (resume): {stats['total_skipped_resume']}")
    print(f"🎯 Alegações geradas:         {stats['total_claims_generated']}")
    print(f"⏱️  Tempo total de execução:   {total_time:.2f}s ({total_time/60:.2f} min)")
    print(f"⚡ Throughput médio:          {throughput:.2f} alegações/segundo ({avg_time:.2f}s/notícia)")
    print("-" * 76)
    print("⚖️  DISTRIBUIÇÃO DE VEREDITOS:")
    v_count = stats["verdicts"].get("V", 0)
    f_count = stats["verdicts"].get("F", 0)
    total_v = stats["total_claims_generated"] or 1
    print(f"   • Verdadeiro (V): {v_count:5d} ({v_count/total_v*100:5.1f}%)")
    print(f"   • Falso/Fake (F): {f_count:5d} ({f_count/total_v*100:5.1f}%)")
    print("-" * 76)
    print("🏷️  DISTRIBUIÇÃO POR TEMA:")
    for top, cnt in sorted(stats["topics"].items(), key=lambda x: x[1], reverse=True):
        print(f"   • {top:15s}: {cnt:5d} ({cnt/total_v*100:5.1f}%)")
    print("-" * 76)
    print(f"💾 Arquivo final gerado: {output_path}")
    if args.jsonl:
        print(f"📄 Arquivo JSONL:        {output_path.with_suffix('.jsonl')}")
    print(f"📝 Log completo em:      {log_file}")
    print("=" * 76)

    logger.info("Processamento concluído com sucesso. %d alegações gravadas em %s", stats["total_claims_generated"], output_path)


if __name__ == "__main__":
    main()

