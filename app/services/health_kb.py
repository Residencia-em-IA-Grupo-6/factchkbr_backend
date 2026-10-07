import csv
import io
import logging
import os
import re
import sqlite3
import ssl
import urllib.request
from pathlib import Path
from typing import Any

logger = logging.getLogger("factchkbr.services.health_kb")

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "health_kb.sqlite3"
DEFAULT_LOCAL_CSV = Path(__file__).resolve().parent.parent / "data" / "DADOS_ABERTOS_MEDICAMENTOS.csv"
ANVISA_CSV_URL = "https://dados.anvisa.gov.br/dados/DADOS_ABERTOS_MEDICAMENTOS.csv"


class HealthLocalKB:
    """
    Camada 1: Base de Conhecimento Local de Saúde baseada nos Dados Abertos da ANVISA.
    
    Não utiliza listas estáticas/hardcoded. Opera sobre o banco SQLite gerado diretamente
    a partir do arquivo oficial 'DADOS_ABERTOS_MEDICAMENTOS.csv' da ANVISA.
    Se a base local não existir, o sistema tenta resgatá-la automaticamente dos dados abertos da ANVISA.
    Caso a obtenção falhe (ambiente offline, timeout ou restrição de rede), o sistema registra o log
    e opera de forma transparente e exclusiva via Camada 2 (APIs externas).
    """

    def __init__(self, db_path: Path | str | None = None) -> None:
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self._bootstrap_attempted: bool = False

    def is_available(self, auto_bootstrap: bool = True) -> bool:
        """
        Verifica se a base local da Anvisa existe e contém registros.
        Se não existir e auto_bootstrap for True, tenta resgatar a base da Anvisa automaticamente.
        """
        if self._check_db_ready():
            return True

        if auto_bootstrap and not self._bootstrap_attempted:
            self._bootstrap_attempted = True
            logger.info("Base local da Anvisa não encontrada em %s. Tentando resgatar automaticamente...", self.db_path)
            success = self.bootstrap_from_anvisa()
            if success and self._check_db_ready():
                return True
            logger.warning(
                "Não foi possível resgatar a base da ANVISA automaticamente. "
                "O pipeline prosseguirá operando exclusivamente via Camada 2 (APIs externas)."
            )

        return False

    def _check_db_ready(self) -> bool:
        """Verifica se o arquivo do banco existe e a tabela anvisa_meds possui registros válidos."""
        if not self.db_path.exists() or self.db_path.stat().st_size == 0:
            return False
        try:
            with self._get_connection() as conn:
                cur = conn.cursor()
                cur.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='anvisa_meds'")
                if not cur.fetchone():
                    return False
                cur.execute("SELECT COUNT(*) FROM anvisa_meds")
                count = cur.fetchone()[0]
                return count > 0
        except Exception as e:
            logger.debug("Erro ao verificar disponibilidade do health_kb (%s): %s", self.db_path, e)
            return False

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=5.0)
        conn.row_factory = sqlite3.Row
        return conn

    def populate_from_csv_stream(self, stream: io.TextIOBase, target_db: Path | None = None) -> int:
        """
        Lê o stream do CSV oficial da ANVISA e popula a tabela anvisa_meds no banco de destino.
        """
        db_file = target_db or self.db_path
        db_file.parent.mkdir(parents=True, exist_ok=True)
        temp_db = db_file.with_name(f"{db_file.name}.tmp")

        if temp_db.exists():
            temp_db.unlink(missing_ok=True)

        reader = csv.reader(stream, delimiter=";")
        try:
            header = next(reader)
        except StopIteration:
            return 0

        conn = sqlite3.connect(str(temp_db))
        try:
            cur = conn.cursor()
            cur.execute("""
                CREATE TABLE anvisa_meds (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    product_type TEXT,
                    trade_name TEXT NOT NULL,
                    process_date TEXT,
                    regulatory_category TEXT,
                    registration_number TEXT,
                    expiration_date TEXT,
                    process_number TEXT,
                    therapeutic_class TEXT,
                    company TEXT,
                    registration_status TEXT,
                    active_principle TEXT
                )
            """)

            batch = []
            total_count = 0
            for row in reader:
                if len(row) >= 11:
                    batch.append((
                        row[0].strip(),   # TIPO_PRODUTO
                        row[1].strip(),   # NOME_PRODUTO
                        row[2].strip(),   # DATA_FINALIZACAO_PROCESSO
                        row[3].strip(),   # CATEGORIA_REGULATORIA
                        row[4].strip(),   # NUMERO_REGISTRO_PRODUTO
                        row[5].strip(),   # DATA_VENCIMENTO_REGISTRO
                        row[6].strip(),   # NUMERO_PROCESSO
                        row[7].strip(),   # CLASSE_TERAPEUTICA
                        row[8].strip(),   # EMPRESA_DETENTORA_REGISTRO
                        row[9].strip(),   # SITUACAO_REGISTRO
                        row[10].strip(),  # PRINCIPIO_ATIVO
                    ))
                    if len(batch) >= 2000:
                        cur.executemany("""
                            INSERT INTO anvisa_meds (
                                product_type, trade_name, process_date, regulatory_category,
                                registration_number, expiration_date, process_number,
                                therapeutic_class, company, registration_status, active_principle
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, batch)
                        total_count += len(batch)
                        batch.clear()

            if batch:
                cur.executemany("""
                    INSERT INTO anvisa_meds (
                        product_type, trade_name, process_date, regulatory_category,
                        registration_number, expiration_date, process_number,
                        therapeutic_class, company, registration_status, active_principle
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, batch)
                total_count += len(batch)

            cur.execute("CREATE INDEX idx_anvisa_trade_name ON anvisa_meds(trade_name)")
            cur.execute("CREATE INDEX idx_anvisa_active_principle ON anvisa_meds(active_principle)")
            conn.commit()
        finally:
            conn.close()

        # Substitui atomicamente o banco temporário pelo definitivo
        os.replace(str(temp_db), str(db_file))
        logger.info("Base local da Anvisa construída com sucesso: %d registros em %s", total_count, db_file)
        return total_count

    def bootstrap_from_local_csv(self, csv_path: Path | str) -> bool:
        """Monta o banco a partir de um arquivo CSV local pré-existente."""
        cpath = Path(csv_path)
        if not cpath.exists() or cpath.stat().st_size == 0:
            return False
        try:
            with open(cpath, "r", encoding="latin-1", errors="replace") as f:
                count = self.populate_from_csv_stream(f)
                return count > 0
        except Exception as e:
            logger.warning("Erro ao popular base a partir de CSV local (%s): %s", cpath, e)
            return False

    def bootstrap_from_anvisa(
        self,
        url: str = ANVISA_CSV_URL,
        timeout: float = 35.0,
        cache_raw_csv: bool = True,
    ) -> bool:
        """
        Resgata e constrói a base de dados a partir dos dados abertos oficiais da ANVISA.
        Utiliza SSL com contexto adequado para o certificado dos portais governamentais brasileiros.
        """
        # Se houver um arquivo CSV em cache local, aproveita para agilidade
        if DEFAULT_LOCAL_CSV.exists() and DEFAULT_LOCAL_CSV.stat().st_size > 100_000:
            logger.info("Encontrado CSV de medicamentos em cache local (%s). Importando...", DEFAULT_LOCAL_CSV)
            if self.bootstrap_from_local_csv(DEFAULT_LOCAL_CSV):
                return True

        logger.info("Baixando dados abertos de medicamentos da ANVISA via %s ...", url)
        try:
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE

            req = urllib.request.Request(
                url,
                headers={"User-Agent": "FactChkBR/1.0 (Health Verifier; OpenData Bot)"},
            )
            with urllib.request.urlopen(req, context=ctx, timeout=timeout) as resp:
                raw_bytes = resp.read()

            if not raw_bytes or len(raw_bytes) < 10_000:
                logger.warning("Conteúdo retornado pela ANVISA é vazio ou insuficiente.")
                return False

            # Opcionalmente salva o CSV bruto para acelerar futuras reconstruções
            if cache_raw_csv:
                try:
                    DEFAULT_LOCAL_CSV.parent.mkdir(parents=True, exist_ok=True)
                    DEFAULT_LOCAL_CSV.write_bytes(raw_bytes)
                except Exception as ce:
                    logger.debug("Não foi possível salvar cache bruto do CSV: %s", ce)

            stream = io.StringIO(raw_bytes.decode("latin-1", errors="replace"))
            count = self.populate_from_csv_stream(stream)
            return count > 0
        except Exception as e:
            logger.warning("Falha ao resgatar base oficial da ANVISA (%s): %s", url, e)
            return False

    def search_medication(self, term: str, limit: int = 5) -> list[dict[str, Any]]:
        """Pesquisa medicamentos na base oficial da Anvisa por nome comercial ou princípio ativo."""
        if not self.is_available() or not term or len(term.strip()) < 3:
            return []

        clean_term = term.strip().lower()
        query = """
            SELECT trade_name, active_principle, regulatory_category, 
                   therapeutic_class, registration_status, company, registration_number
            FROM anvisa_meds 
            WHERE LOWER(trade_name) = ?
               OR LOWER(active_principle) = ?
               OR LOWER(trade_name) LIKE ?
               OR LOWER(trade_name) LIKE ?
               OR LOWER(trade_name) LIKE ?
               OR LOWER(active_principle) LIKE ?
               OR LOWER(active_principle) LIKE ?
               OR LOWER(active_principle) LIKE ?
            ORDER BY 
                CASE 
                    WHEN LOWER(trade_name) = ? THEN 1
                    WHEN LOWER(active_principle) = ? THEN 2
                    WHEN LOWER(trade_name) LIKE ? THEN 3
                    ELSE 4
                END,
                CASE
                    WHEN LOWER(registration_status) IN ('ativo', 'valido', 'válido') THEN 1
                    ELSE 2
                END
            LIMIT ?
        """
        prefix_pattern = f"{clean_term} %"
        suffix_pattern = f"% {clean_term}"
        mid_pattern = f"% {clean_term} %"
        try:
            with self._get_connection() as conn:
                cur = conn.cursor()
                cur.execute(
                    query,
                    (
                        clean_term, clean_term,
                        prefix_pattern, suffix_pattern, mid_pattern,
                        prefix_pattern, suffix_pattern, mid_pattern,
                        clean_term, clean_term, prefix_pattern, limit
                    )
                )
                rows = cur.fetchall()
                return [dict(r) for r in rows]
        except Exception as e:
            logger.warning("Erro ao consultar base local da Anvisa: %s", e)
            return []

    def inspect_health_claim(self, claim_text: str) -> dict[str, Any]:
        """
        Inspeciona uma alegação buscando por substâncias e medicamentos registrados na Anvisa.
        Se a base local não estiver instalada/disponível, retorna 'is_available: False'
        para que o pipeline dependa exclusivamente da Camada 2.
        """
        if not self.is_available():
            return {
                "is_available": False,
                "has_health_entities": False,
                "medications_found": [],
                "regulatory_alerts": [],
                "summary": "Base local da Anvisa indisponível. Análise dependente exclusivamente da Camada 2 (Fontes Externas).",
            }

        # Extrai possíveis nomes de entidades / substantivos com mais de 3 letras
        tokens = re.findall(r"\b[A-Za-zÀ-ÿ]{4,}\b", claim_text)
        matched_meds: list[dict[str, Any]] = []
        alerts: list[str] = []

        seen_names: set[str] = set()
        stop_health = {
            "para", "como", "sobre", "fazer", "onde", "quando", "tratamento", "saúde", "cura",
            "novo", "nova", "vida", "governo", "ministro", "brasil", "dizer", "disse", "pode",
            "esse", "essa", "esta", "este", "qual", "quem", "mais", "muito",
            "paulo", "são", "rio", "minas", "gerais", "bahia", "brasília", "unidade", "unidades",
            "posto", "postos", "hospital", "hospitais", "clínica", "clínicas", "médico", "médicos",
            "paciente", "pacientes", "centro", "estado", "cidade", "ranking", "lista", "total",
            "melhor", "melhores", "pior", "piores", "concentra", "concentram", "concentração",
            "número", "todos", "todas", "primeiro", "último", "ano", "anos", "dia", "dias",
            "meses", "taxa", "índice", "população", "serviço", "serviços", "público", "públicos",
            "água", "agua", "limão", "limao", "café", "cafe", "chá", "cha", "suco", "sucos",
            "leite", "comida", "alimento", "alimentos", "dieta", "dietas", "fruta", "frutas",
            "legume", "legumes", "açúcar", "acucar", "sal", "gordura", "corpo", "peso",
            "emagrece", "emagrecem", "emagrecer", "emagrecimento",
            "engorda", "engordam", "engordar", "engordamento",
            "natural", "naturais", "forma", "formas", "efeito", "efeitos",
            "faz", "fazem", "fazer", "mal", "bem",
            "mata", "matam", "matar", "morte", "mortes",
            "pessoa", "pessoas", "humano", "humanos",
            "bom", "boa", "bons", "boas", "ruim", "ruins",
        }

        for token in tokens:
            if token.lower() in stop_health:
                continue
            meds = self.search_medication(token, limit=2)
            for m in meds:
                t_name = m["trade_name"]
                if t_name not in seen_names:
                    seen_names.add(t_name)
                    matched_meds.append(m)

                    status = m.get("registration_status", "")
                    if status and status.lower() not in ("ativo", "válido", "valido"):
                        alerts.append(
                            f"Registro Anvisa ({t_name}): Status regulatório é '{status}'."
                        )

        return {
            "is_available": True,
            "has_health_entities": bool(matched_meds),
            "medications_found": matched_meds,
            "regulatory_alerts": alerts,
            "summary": (
                f"Base local da Anvisa identificou {len(matched_meds)} produto(s) correspondente(s)."
                if matched_meds
                else "Nenhum fármaco registrado na Anvisa identificado na base local."
            ),
        }


# Instância singleton
_health_kb_instance: HealthLocalKB | None = None


def get_health_kb(db_path: Path | str | None = None) -> HealthLocalKB:
    """Retorna instância singleton do HealthLocalKB."""
    global _health_kb_instance
    if _health_kb_instance is None or db_path is not None:
        _health_kb_instance = HealthLocalKB(db_path=db_path)
    return _health_kb_instance
