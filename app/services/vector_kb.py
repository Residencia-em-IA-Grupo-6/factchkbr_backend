import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import chromadb
from chromadb.api.models.Collection import Collection

from app.config import Settings, get_settings
from app.schemas.analysis import AnalyzeResponse, SubClaimAnalysis, Verdict
from app.schemas.vector import VectorClaimItem, VectorSearchResponse

logger = logging.getLogger("factchkbr.services.vector_kb")


class VectorClaimKB:
    """
    Serviço de Banco de Dados Vetorial Persistente baseado no ChromaDB.
    
    Responsável por:
    1. Armazenar alegações fáticas com veredito (verdadeiro/falso/suspeito/inconclusivo).
    2. Armazenar as fundamentações explicativas e fontes que embasam cada checagem.
    3. Realizar busca vetorial por similaridade semântica para acelerar e embasar o pipeline.
    4. Garantir persistência dos vetores em disco em diretório configurável.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        persist_directory: str | None = None,
        collection_name: str | None = None,
        embedding_function: Any = None,
    ) -> None:
        self.settings = settings or get_settings()
        
        # Diretório de persistência do ChromaDB
        raw_persist_dir = persist_directory or self.settings.CHROMA_PERSIST_DIR
        persist_path = Path(raw_persist_dir)
        if not persist_path.is_absolute():
            # Relativo à raiz do projeto
            repo_root = Path(__file__).resolve().parent.parent.parent
            persist_path = repo_root / raw_persist_dir
            
        self.persist_directory = str(persist_path)
        os.makedirs(self.persist_directory, exist_ok=True)
        
        self.collection_name = collection_name or self.settings.CHROMA_COLLECTION_NAME
        self.min_similarity = self.settings.VECTOR_SIMILARITY_THRESHOLD
        self.embedding_function = embedding_function

        logger.info(
            "Inicializando ChromaDB persistente em '%s' (coleção: '%s')",
            self.persist_directory,
            self.collection_name,
        )

        self._client: chromadb.PersistentClient | None = None
        self._collection: Collection | None = None

    @property
    def client(self) -> chromadb.PersistentClient:
        """Instancia o cliente persistente sob demanda (lazy loading)."""
        if self._client is None:
            self._client = chromadb.PersistentClient(path=self.persist_directory)
        return self._client

    @property
    def collection(self) -> Collection:
        """Obtém ou cria a coleção vetorial no ChromaDB com métrica de cosseno."""
        if self._collection is None:
            kwargs: dict[str, Any] = {
                "name": self.collection_name,
                "metadata": {"hnsw:space": "cosine"},
            }
            if self.embedding_function is not None:
                kwargs["embedding_function"] = self.embedding_function

            self._collection = self.client.get_or_create_collection(**kwargs)
        return self._collection

    @staticmethod
    def generate_claim_id(statement: str) -> str:
        """Gera hash determinístico SHA256 único para a alegação fática normalizada."""
        clean_text = statement.strip().lower()
        return hashlib.sha256(clean_text.encode("utf-8")).hexdigest()[:24]

    def add_claim(
        self,
        statement: str,
        verdict: Verdict | str,
        summary: str = "",
        reasons: list[str] | None = None,
        sources: list[str] | None = None,
        confidence: float = 1.0,
        category: str | None = None,
        claim_id: str | None = None,
        claim_type: str = "primary",
        extra_metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Indexa uma alegação com veredito, fundamentações e fontes no ChromaDB.
        Se a alegação já existir, atualiza suas informações (upsert determinístico).
        """
        stmt_clean = statement.strip()
        if not stmt_clean:
            raise ValueError("O texto da alegação não pode ser vazio.")

        doc_id = claim_id or self.generate_claim_id(stmt_clean)
        
        # Converte veredito para enum canônico
        if isinstance(verdict, str):
            v_upper = verdict.upper()
            v_val = Verdict[v_upper].value if v_upper in Verdict.__members__ else verdict
        else:
            v_val = verdict.value

        reasons_list = reasons or []
        sources_list = sources or []
        
        # Metadados no ChromaDB exigem tipos primitivos (str, int, float, bool)
        metadata: dict[str, Any] = {
            "verdict": str(v_val),
            "confidence": float(confidence),
            "summary": str(summary or ""),
            "reasons_json": json.dumps(reasons_list, ensure_ascii=False),
            "sources_json": json.dumps(sources_list, ensure_ascii=False),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "claim_type": str(claim_type),
            "category": str(category or ""),
        }

        if extra_metadata:
            for k, val in extra_metadata.items():
                if isinstance(val, (str, int, float, bool)):
                    metadata[k] = val
                else:
                    metadata[f"{k}_json"] = json.dumps(val, ensure_ascii=False)

        self.collection.upsert(
            ids=[doc_id],
            documents=[stmt_clean],
            metadatas=[metadata],
        )

        logger.info(
            "Alegação indexada no ChromaDB (id=%s, veredito=%s, fontes=%d): '%s'",
            doc_id, v_val, len(sources_list), stmt_clean[:60]
        )
        return doc_id

    def add_from_analysis(
        self,
        analysis: AnalyzeResponse,
        category: str | None = None,
    ) -> list[str]:
        """
        Indexa automaticamente os resultados de uma verificação completa:
        - A alegação principal consolidada.
        - Todas as sub-alegações atômicas avaliadas individualmente.
        """
        indexed_ids: list[str] = []

        # 1. Indexa a alegação principal
        if analysis.claim and analysis.verdict:
            main_id = self.add_claim(
                statement=analysis.claim,
                verdict=analysis.verdict,
                summary=analysis.summary,
                reasons=analysis.reasons,
                sources=analysis.sources,
                confidence=analysis.confidence,
                category=category,
                claim_type="primary",
            )
            indexed_ids.append(main_id)

        # 2. Indexa as sub-alegações específicas
        for idx, sub in enumerate(analysis.sub_claims):
            sub_id = self.generate_claim_id(sub.statement)
            if sub_id in indexed_ids:
                continue
            
            sub_reasons = [sub.justification] if sub.justification else []
            self.add_claim(
                statement=sub.statement,
                verdict=sub.verdict,
                summary=sub.justification,
                reasons=sub_reasons,
                sources=sub.sources or analysis.sources,
                confidence=sub.confidence,
                category=category,
                claim_id=sub_id,
                claim_type="sub_claim",
                extra_metadata={"parent_claim_id": indexed_ids[0] if indexed_ids else ""},
            )
            indexed_ids.append(sub_id)

        return indexed_ids

    def search_claims(
        self,
        query: str,
        limit: int = 3,
        min_similarity: float | None = None,
        verdict_filter: Verdict | str | None = None,
    ) -> list[VectorClaimItem]:
        """
        Realiza busca semântica por similaridade vetorial (cosseno) no ChromaDB.
        
        Retorna itens que atendem ao limiar mínimo de similaridade (padrão configurado).
        """
        q_clean = query.strip()
        if not q_clean or self.count() == 0:
            return []

        threshold = min_similarity if min_similarity is not None else self.min_similarity

        where_clause: dict[str, Any] | None = None
        if verdict_filter:
            v_val = verdict_filter.value if hasattr(verdict_filter, "value") else str(verdict_filter).upper()
            where_clause = {"verdict": v_val}

        # Query no ChromaDB
        query_kwargs: dict[str, Any] = {
            "query_texts": [q_clean],
            "n_results": min(limit, self.count()),
        }
        if where_clause:
            query_kwargs["where"] = where_clause

        try:
            raw_res = self.collection.query(**query_kwargs)
        except Exception as e:
            logger.error("Erro na busca vetorial do ChromaDB: %s", e)
            return []

        ids = raw_res.get("ids", [[]])[0]
        documents = raw_res.get("documents", [[]])[0]
        metadatas = raw_res.get("metadatas", [[]])[0]
        distances = raw_res.get("distances", [[]])[0] if raw_res.get("distances") else [0.0] * len(ids)

        items: list[VectorClaimItem] = []
        for doc_id, doc_text, meta, dist in zip(ids, documents, metadatas, distances):
            # Métrica de cosseno no ChromaDB varia de 0.0 (idêntico) a 2.0 (oposto)
            # Similaridade cosseno: 1.0 - distância
            sim = max(0.0, min(1.0, 1.0 - float(dist)))
            
            if sim < threshold:
                continue

            # Deserializa listas de razões e fontes
            reasons = []
            if "reasons_json" in meta:
                try:
                    reasons = json.loads(meta["reasons_json"])
                except Exception:
                    reasons = []

            sources = []
            if "sources_json" in meta:
                try:
                    sources = json.loads(meta["sources_json"])
                except Exception:
                    sources = []

            v_str = meta.get("verdict", "INCONCLUSIVO")
            verdict_enum = Verdict[v_str] if v_str in Verdict.__members__ else Verdict.INCONCLUSIVO

            items.append(
                VectorClaimItem(
                    id=doc_id,
                    statement=doc_text,
                    verdict=verdict_enum,
                    confidence=float(meta.get("confidence", 0.5)),
                    summary=str(meta.get("summary", "")),
                    reasons=reasons,
                    sources=sources,
                    created_at=meta.get("created_at"),
                    similarity=round(sim, 4),
                    distance=round(float(dist), 4),
                    category=meta.get("category"),
                    metadata=meta,
                )
            )

        # Ordena por maior similaridade
        items.sort(key=lambda x: (x.similarity or 0.0), reverse=True)
        return items

    def get_claim(self, claim_id: str) -> VectorClaimItem | None:
        """Recupera uma alegação específica pelo seu ID."""
        try:
            res = self.collection.get(ids=[claim_id])
            ids = res.get("ids", [])
            if not ids:
                return None
            
            doc_id = ids[0]
            doc_text = res["documents"][0]
            meta = res["metadatas"][0]

            reasons = []
            if "reasons_json" in meta:
                try:
                    reasons = json.loads(meta["reasons_json"])
                except Exception:
                    reasons = []

            sources = []
            if "sources_json" in meta:
                try:
                    sources = json.loads(meta["sources_json"])
                except Exception:
                    sources = []

            v_str = meta.get("verdict", "INCONCLUSIVO")
            verdict_enum = Verdict[v_str] if v_str in Verdict.__members__ else Verdict.INCONCLUSIVO

            return VectorClaimItem(
                id=doc_id,
                statement=doc_text,
                verdict=verdict_enum,
                confidence=float(meta.get("confidence", 0.5)),
                summary=str(meta.get("summary", "")),
                reasons=reasons,
                sources=sources,
                created_at=meta.get("created_at"),
                category=meta.get("category"),
                metadata=meta,
            )
        except Exception as e:
            logger.warning("Falha ao recuperar alegação %s do ChromaDB: %s", claim_id, e)
            return None

    def delete_claim(self, claim_id: str) -> bool:
        """Remove uma alegação do banco vetorial."""
        try:
            self.collection.delete(ids=[claim_id])
            logger.info("Alegação %s removida do ChromaDB.", claim_id)
            return True
        except Exception as e:
            logger.warning("Erro ao remover alegação %s: %s", claim_id, e)
            return False

    def count(self) -> int:
        """Retorna o número total de alegações indexadas."""
        try:
            return self.collection.count()
        except Exception:
            return 0

    def clear(self) -> None:
        """Remove todas as alegações da coleção atual."""
        try:
            self.client.delete_collection(name=self.collection_name)
            self._collection = None
            logger.info("Coleção '%s' do ChromaDB limpa.", self.collection_name)
        except Exception as e:
            logger.warning("Erro ao limpar coleção ChromaDB: %s", e)

    def get_stats(self) -> dict[str, Any]:
        """Retorna métricas operacionais do serviço vetorial."""
        return {
            "total_claims": self.count(),
            "collection_name": self.collection_name,
            "persist_directory": self.persist_directory,
            "enabled": self.settings.VECTOR_SEARCH_ENABLED,
            "similarity_threshold": self.min_similarity,
        }


# Singleton para injeção de dependência FastAPI e orquestrador
_vector_kb_instance: VectorClaimKB | None = None


def get_vector_kb(settings: Settings | None = None) -> VectorClaimKB:
    """Retorna instância singleton do serviço de banco vetorial."""
    global _vector_kb_instance
    if _vector_kb_instance is None:
        _vector_kb_instance = VectorClaimKB(settings=settings)
    return _vector_kb_instance
