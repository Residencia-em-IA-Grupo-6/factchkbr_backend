from abc import ABC, abstractmethod
import json
import logging
import os
import re
from typing import Any

logger = logging.getLogger("factchkbr.analyzers.lexicon")


class BaseLexiconRepository(ABC):
    """
    Interface abstrata para armazenamento e recuperação de padrões léxicos,
    expressões regulares de urgência e siglas.

    Permite desacoplar a heurística de arquivos estáticos, viabilizando
    a transição transparente para um Banco de Dados escalável (PostgreSQL, Redis, SQLite).
    """

    @abstractmethod
    def get_acronyms(self) -> set[str]:
        """Retorna conjunto de siglas legítimas para exclusão de ALL CAPS."""
        pass

    @abstractmethod
    def get_urgency_patterns(self) -> list[tuple[str, re.Pattern, float]]:
        """
        Retorna lista de tuplas contendo:
        (rótulo, regex_compilada, severidade/peso)
        """
        pass

    @abstractmethod
    def get_raw_patterns(self) -> list[dict[str, Any]]:
        """Retorna os metadados brutos dos padrões cadastrados."""
        pass

    @abstractmethod
    def reload(self) -> None:
        """Recarrega os padrões em memória a partir da fonte de dados."""
        pass


class JsonFileLexiconRepository(BaseLexiconRepository):
    """
    Implementação baseada em arquivos JSON desacoplados.
    Carrega os dados e mantém cache compilado em memória para máxima performance.
    """

    def __init__(
        self,
        acronyms_path: str | None = None,
        patterns_path: str | None = None
    ) -> None:
        base_dir = os.path.join(os.path.dirname(__file__), "data")
        self.acronyms_path = acronyms_path or os.path.join(base_dir, "acronyms.json")
        self.patterns_path = patterns_path or os.path.join(base_dir, "urgency_patterns.json")

        self._acronyms: set[str] = set()
        self._compiled_patterns: list[tuple[str, re.Pattern, float]] = []
        self._raw_patterns: list[dict[str, Any]] = []
        self._acronyms_by_cat: dict[str, list[str]] = {}

        self.reload()

    def reload(self) -> None:
        """Carrega e compila as expressões a partir dos arquivos JSON."""
        # 1. Carrega siglas
        self._acronyms = set()
        if os.path.exists(self.acronyms_path):
            try:
                with open(self.acronyms_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    categories = data.get("categories", {})
                    self._acronyms_by_cat = categories
                    for cat_items in categories.values():
                        for item in cat_items:
                            self._acronyms.add(item.strip().upper())
                logger.info("Carregadas %d siglas de %s", len(self._acronyms), self.acronyms_path)
            except Exception as e:
                logger.error("Erro ao carregar siglas de %s: %s", self.acronyms_path, e)

        # 2. Carrega padrões de urgência
        self._compiled_patterns = []
        self._raw_patterns = []
        if os.path.exists(self.patterns_path):
            try:
                with open(self.patterns_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    patterns = data.get("patterns", [])
                    self._raw_patterns = patterns
                    for p in patterns:
                        label = p.get("label", "")
                        regex_str = p.get("regex", "")
                        severity = float(p.get("severity", 1.0))
                        if regex_str:
                            compiled = re.compile(regex_str, re.IGNORECASE)
                            self._compiled_patterns.append((label, compiled, severity))
                logger.info("Carregados %d padrões de urgência de %s", len(self._compiled_patterns), self.patterns_path)
            except Exception as e:
                logger.error("Erro ao carregar padrões de urgência de %s: %s", self.patterns_path, e)

    def get_acronyms(self) -> set[str]:
        return self._acronyms

    def get_urgency_patterns(self) -> list[tuple[str, re.Pattern, float]]:
        return self._compiled_patterns

    def get_raw_patterns(self) -> list[dict[str, Any]]:
        return self._raw_patterns


class DatabaseLexiconRepository(BaseLexiconRepository):
    """
    Estrutura pronta para conexão com Banco de Dados Relacional Escalável (PostgreSQL / SQLite).
    
    Esquema de Tabelas Recomendado:
    
    CREATE TABLE acronyms (
        id SERIAL PRIMARY KEY,
        acronym VARCHAR(20) UNIQUE NOT NULL,
        category VARCHAR(50) NOT NULL,
        is_active BOOLEAN DEFAULT TRUE,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    CREATE INDEX idx_acronyms_upper ON acronyms (acronym);

    CREATE TABLE urgency_patterns (
        id VARCHAR(50) PRIMARY KEY,
        label VARCHAR(100) NOT NULL,
        regex_pattern TEXT NOT NULL,
        category VARCHAR(50) NOT NULL,
        severity FLOAT DEFAULT 1.0,
        source VARCHAR(100),
        is_active BOOLEAN DEFAULT TRUE,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """

    def __init__(self, db_connection: Any | None = None) -> None:
        self.db_connection = db_connection
        self._fallback = JsonFileLexiconRepository()

    def get_acronyms(self) -> set[str]:
        if not self.db_connection:
            return self._fallback.get_acronyms()
        # TODO: Executar consulta SELECT acronym FROM acronyms WHERE is_active = TRUE;
        return self._fallback.get_acronyms()

    def get_urgency_patterns(self) -> list[tuple[str, re.Pattern, float]]:
        if not self.db_connection:
            return self._fallback.get_urgency_patterns()
        # TODO: Executar consulta SELECT label, regex_pattern, severity FROM urgency_patterns WHERE is_active = TRUE;
        return self._fallback.get_urgency_patterns()

    def get_raw_patterns(self) -> list[dict[str, Any]]:
        return self._fallback.get_raw_patterns()

    def reload(self) -> None:
        if self._fallback:
            self._fallback.reload()


# Singleton padrão do repositório de léxicos
_global_repository: BaseLexiconRepository | None = None


def get_lexicon_repository() -> BaseLexiconRepository:
    """Retorna a instância singleton do repositório de léxicos e acrônimos."""
    global _global_repository
    if _global_repository is None:
        _global_repository = JsonFileLexiconRepository()
    return _global_repository
