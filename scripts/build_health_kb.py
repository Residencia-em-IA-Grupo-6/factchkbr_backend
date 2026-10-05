#!/usr/bin/env python3
"""
Script utilitário para montagem, atualização e verificação da Base de Conhecimento Local da ANVISA.

Opera sobre os Dados Abertos Oficiais de Medicamentos da ANVISA (DADOS_ABERTOS_MEDICAMENTOS.csv).
Não utiliza dados estáticos ou hardcoded.

Uso:
    python scripts/build_health_kb.py
    python scripts/build_health_kb.py --rebuild
    python scripts/build_health_kb.py --stats
    python scripts/build_health_kb.py --query "ozempic"
    python scripts/build_health_kb.py --csv-file /caminho/para/DADOS_ABERTOS_MEDICAMENTOS.csv
"""

import argparse
import sys
import time
from pathlib import Path

# Garante inclusão da raiz do projeto no path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.health_kb import HealthLocalKB, DEFAULT_DB_PATH, ANVISA_CSV_URL


def main() -> None:
    parser = argparse.ArgumentParser(description="Construtor e Verificador da Base Local da ANVISA (Dados Abertos)")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Força novo download dos dados abertos da ANVISA e reconstrói o banco SQLite do zero.",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=str(DEFAULT_DB_PATH),
        help=f"Caminho customizado para o arquivo SQLite (padrão: {DEFAULT_DB_PATH})",
    )
    parser.add_argument(
        "--url",
        type=str,
        default=ANVISA_CSV_URL,
        help=f"URL do arquivo CSV de dados abertos da ANVISA (padrão: {ANVISA_CSV_URL})",
    )
    parser.add_argument(
        "--csv-file",
        type=str,
        help="Caminho para arquivo CSV local para montagem offline (evita download).",
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Testa busca por termo de fármaco ou princípio ativo na base montada.",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Exibe estatísticas de registros da tabela oficial de medicamentos da ANVISA.",
    )

    args = parser.parse_args()

    db_path = Path(args.db_path)
    print(f"📦 Base Local da ANVISA configurada em: {db_path}")

    kb = HealthLocalKB(db_path=db_path)

    if args.csv_file:
        cpath = Path(args.csv_file)
        print(f"🔄 Importando base a partir do arquivo CSV local: {cpath} ...")
        t0 = time.time()
        success = kb.bootstrap_from_local_csv(cpath)
        t1 = time.time()
        if success:
            print(f"✅ Base local construída com sucesso em {t1 - t0:.2f}s!")
        else:
            print("❌ Falha ao construir base a partir do CSV local.")
            sys.exit(1)
    elif args.rebuild or not kb._check_db_ready():
        if args.rebuild and db_path.exists():
            db_path.unlink()
        print(f"🔄 Baixando dados abertos oficiais da ANVISA de:\n   {args.url} ...")
        t0 = time.time()
        success = kb.bootstrap_from_anvisa(url=args.url)
        t1 = time.time()
        if success:
            print(f"✅ Base oficial ANVISA resgatada e construída com sucesso em {t1 - t0:.2f}s!")
        else:
            print("❌ Falha ao resgatar base da ANVISA via rede. O pipeline usará exclusivamente Camada 2.")
            if not kb._check_db_ready():
                sys.exit(1)
    else:
        print("ℹ️ Base local da ANVISA já se encontra pronta e indexada.")

    # Exibe estatísticas se o banco estiver pronto
    if kb._check_db_ready():
        with kb._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM anvisa_meds")
            total_meds = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM anvisa_meds WHERE LOWER(registration_status) IN ('ativo', 'valido', 'válido')")
            active_meds = cur.fetchone()[0]
            inactive_meds = total_meds - active_meds

        print("\n📊 Estatísticas da Base Oficial ANVISA:")
        print(f"   • Total de Medicamentos/Fármacos: {total_meds:,} registros")
        print(f"   • Registros Ativos / Válidos:    {active_meds:,}")
        print(f"   • Registros Cancelados/Inativos: {inactive_meds:,}")

    if args.query:
        print(f"\n🔍 Executando busca de teste para: '{args.query}'")
        meds = kb.search_medication(args.query, limit=5)
        inspection = kb.inspect_health_claim(args.query)

        print(f"\n   [Medicamentos Encontrados ({len(meds)})]:")
        for m in meds:
            print(f"     - {m['trade_name']} | Princípio Ativo: {m['active_principle']}")
            print(f"       Classe: {m['therapeutic_class']} | Empresa: {m['company']}")
            print(f"       Status: {m['registration_status']} | Registro: {m['registration_number']}")
        if not meds:
            print("     (Nenhum medicamento diretamente correlacionado na base local)")

        if inspection.get("regulatory_alerts"):
            print("\n   ⚠️ [Alertas Regulatórios Detectados]:")
            for alert in inspection["regulatory_alerts"]:
                print(f"     • {alert}")

    print("\n✨ Operação concluída com sucesso.\n")


if __name__ == "__main__":
    main()
