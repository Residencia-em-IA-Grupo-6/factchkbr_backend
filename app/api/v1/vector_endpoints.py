import logging
from typing import Any
from fastapi import APIRouter, Depends, HTTPException, status

from app.config import Settings, get_settings
from app.schemas.vector import (
    VectorClaimItem,
    VectorIndexRequest,
    VectorIndexResponse,
    VectorSearchRequest,
    VectorSearchResponse,
    VectorStatsResponse,
)
from app.services.vector_kb import VectorClaimKB, get_vector_kb

logger = logging.getLogger("factchkbr.api.v1.vector")

router = APIRouter(prefix="/vector", tags=["Vector Database"])


def get_vector_kb_dep() -> VectorClaimKB:
    """Injeção de dependência para o serviço de banco de dados vetorial."""
    return get_vector_kb()


@router.get(
    "/stats",
    response_model=VectorStatsResponse,
    status_code=status.HTTP_200_OK,
    summary="Estatísticas da Base Vetorial",
    description="Retorna métricas operacionais do banco vetorial ChromaDB (total de alegações, status, diretório).",
)
def get_vector_stats(
    vector_kb: VectorClaimKB = Depends(get_vector_kb_dep),
) -> VectorStatsResponse:
    stats = vector_kb.get_stats()
    return VectorStatsResponse(
        total_claims=stats["total_claims"],
        collection_name=stats["collection_name"],
        persist_directory=stats["persist_directory"],
        enabled=stats["enabled"],
    )


@router.post(
    "/search",
    response_model=VectorSearchResponse,
    status_code=status.HTTP_200_OK,
    summary="Busca Semântica de Alegações",
    description="Pesquisa no ChromaDB alegações fáticas semanticamente próximas à consulta, com suas fundamentações e fontes.",
)
def search_vector_claims(
    request: VectorSearchRequest,
    vector_kb: VectorClaimKB = Depends(get_vector_kb_dep),
) -> VectorSearchResponse:
    results = vector_kb.search_claims(
        query=request.query,
        limit=request.limit,
        min_similarity=request.min_similarity,
        verdict_filter=request.verdict_filter,
    )
    return VectorSearchResponse(
        query=request.query,
        total_found=len(results),
        results=results,
    )


@router.post(
    "/index",
    response_model=VectorIndexResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Indexar Alegação no Banco Vetorial",
    description="Indexa uma alegação com seu veredito de veracidade, justificativas e fontes na memória persistente.",
)
def index_claim(
    request: VectorIndexRequest,
    vector_kb: VectorClaimKB = Depends(get_vector_kb_dep),
) -> VectorIndexResponse:
    claim_id = vector_kb.add_claim(
        statement=request.statement,
        verdict=request.verdict,
        confidence=request.confidence,
        summary=request.summary,
        reasons=request.reasons,
        sources=request.sources,
        category=request.category,
    )
    return VectorIndexResponse(
        id=claim_id,
        status="indexed",
        statement=request.statement,
        verdict=request.verdict,
    )


@router.get(
    "/claims/{claim_id}",
    response_model=VectorClaimItem,
    status_code=status.HTTP_200_OK,
    summary="Obter Alegação por ID",
    description="Recupera uma alegação e seus metadados persistidos pelo seu identificador único.",
)
def get_claim_by_id(
    claim_id: str,
    vector_kb: VectorClaimKB = Depends(get_vector_kb_dep),
) -> VectorClaimItem:
    item = vector_kb.get_claim(claim_id)
    if not item:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Alegação com ID '{claim_id}' não encontrada na base vetorial.",
        )
    return item


@router.delete(
    "/claims/{claim_id}",
    status_code=status.HTTP_200_OK,
    summary="Remover Alegação da Base Vetorial",
    description="Remove uma alegação do ChromaDB pelo seu identificador único.",
)
def delete_claim_by_id(
    claim_id: str,
    vector_kb: VectorClaimKB = Depends(get_vector_kb_dep),
) -> dict[str, Any]:
    deleted = vector_kb.delete_claim(claim_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Alegação com ID '{claim_id}' não encontrada ou não pôde ser removida.",
        )
    return {"status": "deleted", "id": claim_id}
 
 
@router.post(
    "/deduplicate",
    status_code=status.HTTP_200_OK,
    summary="Unificar Registros Duplicados",
    description="Varre o ChromaDB, agrupa alegações equivalentes, mescla fontes e fundamentações e remove registros redundantes.",
)
def deduplicate_claims(
    vector_kb: VectorClaimKB = Depends(get_vector_kb_dep),
) -> dict[str, Any]:
    return vector_kb.deduplicate_collection()


@router.post(
    "/reset",
    status_code=status.HTTP_200_OK,
    summary="Resetar Coleção Vetorial",
    description="Remove todas as alegações e recria a coleção vetorial limpa no ChromaDB.",
)
def reset_vector_collection(
    vector_kb: VectorClaimKB = Depends(get_vector_kb_dep),
) -> dict[str, Any]:
    total_removed = vector_kb.reset()
    return {
        "status": "reset",
        "total_removed": total_removed,
        "collection_name": vector_kb.collection_name,
    }

