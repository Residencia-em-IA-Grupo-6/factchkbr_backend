import os
import tempfile
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.schemas.analysis import AnalyzeResponse, SubClaimAnalysis, Verdict
from app.schemas.vector import VectorClaimItem
from app.services.vector_kb import VectorClaimKB, get_vector_kb


@pytest.fixture
def temp_vector_kb():
    """Cria uma instância isolada do VectorClaimKB em diretório temporário."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        kb = VectorClaimKB(
            persist_directory=tmp_dir,
            collection_name="test_fact_claims",
        )
        yield kb


def test_add_and_get_claim(temp_vector_kb):
    """Testa adição e recuperação de uma alegação no banco vetorial."""
    claim_id = temp_vector_kb.add_claim(
        statement="Chá de boldo cura diabetes tipo 2 em 3 dias",
        verdict=Verdict.FAKE,
        confidence=0.98,
        summary="Não existem evidências médicas de que boldo cure diabetes.",
        reasons=[
            "Diabetes é uma condição crônica sem cura por fitoterapia.",
            "Boldo possui propriedades digestivas, não antidiabéticas comprovadas.",
        ],
        sources=[
            "https://saude.gov.br/boatos/cha-boldo-diabetes",
            "https://sbem.org.br/posicionamento-fitoterapia",
        ],
        category="PUBLIC_HEALTH",
    )

    assert claim_id is not None
    assert len(claim_id) == 24
    assert temp_vector_kb.count() == 1

    # Recupera por ID
    item = temp_vector_kb.get_claim(claim_id)
    assert item is not None
    assert item.id == claim_id
    assert item.statement == "Chá de boldo cura diabetes tipo 2 em 3 dias"
    assert item.verdict == Verdict.FAKE
    assert item.confidence == 0.98
    assert len(item.reasons) == 2
    assert len(item.sources) == 2
    assert item.category == "PUBLIC_HEALTH"


def test_upsert_deterministic_id(temp_vector_kb):
    """Garante que indexar a mesma alegação atualiza o registro sem duplicar."""
    id1 = temp_vector_kb.add_claim(
        statement="Água com limão emagrece",
        verdict=Verdict.FAKE,
        confidence=0.80,
    )
    assert temp_vector_kb.count() == 1

    id2 = temp_vector_kb.add_claim(
        statement="  Água com limão emagrece  ",
        verdict=Verdict.FAKE,
        confidence=0.95,
        summary="Atualização de justificativa.",
    )
    assert id1 == id2
    assert temp_vector_kb.count() == 1

    updated = temp_vector_kb.get_claim(id1)
    assert updated.confidence == 0.95
    assert updated.summary == "Atualização de justificativa."


def test_search_claims_semantic_similarity(temp_vector_kb):
    """Testa busca vetorial semântica e limiares de similaridade."""
    temp_vector_kb.add_claim(
        statement="Vacinas de RNA causam alteração no DNA humano",
        verdict=Verdict.FAKE,
        confidence=0.99,
        summary="Vacinas de mRNA não penetram o núcleo celular e não alteram o DNA.",
        reasons=["O mRNA degrada rapidamente no citoplasma celular."],
        sources=["https://who.int/vaccines/mrna-safety"],
    )
    temp_vector_kb.add_claim(
        statement="Aspirina previne novos infartos em pacientes de alto risco",
        verdict=Verdict.VERDADEIRO,
        confidence=0.95,
        summary="Ação antiplaquetária comprovada para prevenção secundária cardiovascular.",
        sources=["https://who.int/cardiovascular"],
    )

    assert temp_vector_kb.count() == 2

    # Busca com variação semântica da vacina
    matches = temp_vector_kb.search_claims(
        query="vacina de RNA mensageiro altera o genoma e o DNA",
        limit=2,
        min_similarity=0.40,
    )
    assert len(matches) >= 1
    top_match = matches[0]
    assert "Vacinas de RNA" in top_match.statement
    assert top_match.verdict == Verdict.FAKE
    assert top_match.similarity > 0.40

    # Busca com filtro de veredito
    true_matches = temp_vector_kb.search_claims(
        query="infarto e aspirina prevenção cardiovascular",
        limit=2,
        verdict_filter=Verdict.VERDADEIRO,
    )
    assert len(true_matches) == 1
    assert true_matches[0].verdict == Verdict.VERDADEIRO


def test_delete_claim(temp_vector_kb):
    """Testa remoção de alegações pelo ID."""
    doc_id = temp_vector_kb.add_claim(
        statement="Alegação temporária para exclusão",
        verdict=Verdict.INCONCLUSIVO,
    )
    assert temp_vector_kb.count() == 1

    deleted = temp_vector_kb.delete_claim(doc_id)
    assert deleted is True
    assert temp_vector_kb.count() == 0
    assert temp_vector_kb.get_claim(doc_id) is None


def test_add_from_analysis(temp_vector_kb):
    """Testa indexação em lote a partir do resultado consolidado do pipeline."""
    analysis = AnalyzeResponse(
        claim="Melatonina cura insônia e trata depressão profunda",
        verdict=Verdict.SUSPEITO,
        confidence=0.85,
        summary="Melatonina auxilia no ciclo circadiano, mas não cura depressão.",
        reasons=["Melatonina é hormônio regulador do sono, sem indicação antidepressiva isolada."],
        sources=["https://anvisa.gov.br/melatonina"],
        sub_claims=[
            SubClaimAnalysis(
                statement="Melatonina auxilia no início do sono",
                verdict=Verdict.VERDADEIRO,
                confidence=0.90,
                justification="Estudos clínicos comprovam eficácia para indução do sono.",
                sources=["https://anvisa.gov.br/melatonina"],
            ),
            SubClaimAnalysis(
                statement="Melatonina cura depressão profunda",
                verdict=Verdict.FAKE,
                confidence=0.95,
                justification="Não há respaldo científico para efeito curativo em depressão maior.",
                sources=["https://who.int/mental-health"],
            ),
        ],
    )

    ids = temp_vector_kb.add_from_analysis(analysis, category="BIOMEDICINE")
    assert len(ids) == 3
    assert temp_vector_kb.count() == 3

    # Verifica se a sub-alegação sobre depressão pode ser recuperada
    matches = temp_vector_kb.search_claims("melatonina cura depressão", limit=1, min_similarity=0.50)
    assert len(matches) == 1
    assert matches[0].verdict == Verdict.FAKE


def test_persistence_across_instances():
    """Garante que dados persistem em disco ao recriar o cliente no mesmo diretório."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Instância 1: indexa
        kb1 = VectorClaimKB(persist_directory=tmp_dir, collection_name="persistent_test")
        doc_id = kb1.add_claim(
            statement="Gargarejo com água morna e sal elimina coronavírus da garganta",
            verdict=Verdict.FAKE,
            confidence=0.99,
            summary="O vírus se replica nas células respiratórias e o gargarejo não atinge o foco viral.",
            sources=["https://who.int/coronavirus-mythbusters"],
        )
        assert kb1.count() == 1

        # Instância 2: lê do mesmo diretório
        kb2 = VectorClaimKB(persist_directory=tmp_dir, collection_name="persistent_test")
        assert kb2.count() == 1
        item = kb2.get_claim(doc_id)
        assert item is not None
        assert item.verdict == Verdict.FAKE
        assert "gargarejo" in item.summary.lower()


def test_api_vector_endpoints(monkeypatch):
    """Testa os endpoints HTTP FastAPI /api/v1/vector/."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        test_kb = VectorClaimKB(persist_directory=tmp_dir, collection_name="api_test_claims")
        
        # Injeta test_kb como singleton
        import app.services.vector_kb as vk_module
        monkeypatch.setattr(vk_module, "_vector_kb_instance", test_kb)

        client = TestClient(app)

        # 1. Stats inicial
        res_stats = client.get("/api/v1/vector/stats")
        assert res_stats.status_code == 200
        data_stats = res_stats.json()
        assert data_stats["total_claims"] == 0
        assert data_stats["enabled"] is True

        # 2. Indexa nova alegação
        payload = {
            "statement": "Ivermectina previne infecção por dengue",
            "verdict": "FAKE",
            "confidence": 0.95,
            "summary": "Não há eficácia comprovada contra o vírus da dengue.",
            "reasons": ["Estudos clínicos não demonstram redução de carga viral."],
            "sources": ["https://saude.gov.br/dengue-ivermectina"],
            "category": "PUBLIC_HEALTH",
        }
        res_index = client.post("/api/v1/vector/index", json=payload)
        assert res_index.status_code == 201
        data_index = res_index.json()
        claim_id = data_index["id"]
        assert claim_id is not None
        assert data_index["status"] == "indexed"

        # 3. Busca semântica
        res_search = client.post(
            "/api/v1/vector/search",
            json={
                "query": "tomar ivermectina para não pegar dengue",
                "limit": 2,
                "min_similarity": 0.40,
            },
        )
        assert res_search.status_code == 200
        data_search = res_search.json()
        assert data_search["total_found"] >= 1
        assert data_search["results"][0]["verdict"] == "FAKE"

        # 4. Obtém por ID
        res_get = client.get(f"/api/v1/vector/claims/{claim_id}")
        assert res_get.status_code == 200
        assert res_get.json()["id"] == claim_id

        # 5. Remove por ID
        res_del = client.delete(f"/api/v1/vector/claims/{claim_id}")
        assert res_del.status_code == 200
        assert res_del.json()["status"] == "deleted"

        # Confirma que foi removido
        res_get_after = client.get(f"/api/v1/vector/claims/{claim_id}")
        assert res_get_after.status_code == 404


@pytest.mark.asyncio
async def test_fact_check_api_bypasses_external_searches_on_vector_cache_hit(monkeypatch):
    """Garante que alegações previamente analisadas no ChromaDB dispensam buscas externas."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        test_kb = VectorClaimKB(persist_directory=tmp_dir, collection_name="fastpath_claims")
        test_kb.add_claim(
            statement="A Anvisa proibiu a venda de lote de azeite adulterado no país.",
            verdict=Verdict.FAKE,
            confidence=0.95,
            summary="Azeite adulterado desmentido por órgãos oficiais.",
            reasons=["Boato sem fundamentação em registros da Anvisa."],
            sources=["https://anvisa.gov.br/comunicado-azeite"],
        )

        import app.services.vector_kb as vk_module
        monkeypatch.setattr(vk_module, "_vector_kb_instance", test_kb)

        from app.analyzers.fact_check_api import FactCheckApiAnalyzer
        analyzer = FactCheckApiAnalyzer()

        # Monkeypatch das funções de busca externa para garantir que NENHUMA seja chamada
        called_external = []
        async def fake_search(*args, **kwargs):
            called_external.append(True)
            return []

        monkeypatch.setattr(analyzer, "search_google_fact_check", fake_search)
        monkeypatch.setattr(analyzer, "search_lateral_reading", fake_search)
        monkeypatch.setattr(analyzer, "search_clinical_trials", fake_search)
        monkeypatch.setattr(analyzer, "search_biomedical_literature", fake_search)

        result = await analyzer.analyze("A Anvisa proibiu a venda de lote de azeite adulterado no país.", [])

        # Nenhuma busca externa deve ter sido chamada
        assert len(called_external) == 0
        assert result.verdict == Verdict.FAKE
        assert result.raw_details.get("vector_cache_hit") is True
        assert result.raw_details.get("google_fact_check_count") == 0
        assert result.raw_details.get("lateral_reading_count") == 0
        assert "⚡ Alegação já analisada previamente" in result.summary
        await analyzer.aclose()


def test_canonical_deduplication_punctuation_and_whitespace(temp_vector_kb):
    """Testa se variações de pontuação e aspas geram o mesmo ID canônico e não duplicam."""
    # 1. Primeira inserção com ponto final e aspas
    id1 = temp_vector_kb.add_claim(
        statement='"A Anvisa proibiu a venda de lote de azeite adulterado no país."',
        verdict=Verdict.FAKE,
        confidence=0.75,
        summary="Resumo inicial.",
        sources=["https://anvisa.gov.br/fonte1"],
        reasons=["Motivo 1"],
    )

    # 2. Segunda inserção sem ponto e sem aspas
    id2 = temp_vector_kb.add_claim(
        statement="A Anvisa proibiu a venda de lote de azeite adulterado no país",
        verdict=Verdict.FAKE,
        confidence=0.85,
        summary="Resumo mais completo sobre azeite adulterado.",
        sources=["https://anvisa.gov.br/fonte2"],
        reasons=["Motivo 2"],
    )

    # 3. Terceira inserção com múltiplos espaços e pontuação mista
    id3 = temp_vector_kb.add_claim(
        statement="  A Anvisa proibiu a venda de lote de azeite adulterado no país...  ",
        verdict=Verdict.FAKE,
        confidence=0.90,
        sources=["https://anvisa.gov.br/fonte1", "https://anvisa.gov.br/fonte3"],
    )

    assert id1 == id2 == id3
    assert temp_vector_kb.count() == 1

    item = temp_vector_kb.get_claim(id1)
    assert item is not None
    assert item.statement == "A Anvisa proibiu a venda de lote de azeite adulterado no país"
    # Fontes mescladas sem duplicatas
    assert len(item.sources) == 3
    assert "https://anvisa.gov.br/fonte1" in item.sources
    assert "https://anvisa.gov.br/fonte2" in item.sources
    assert "https://anvisa.gov.br/fonte3" in item.sources
    # Motivos mesclados
    assert len(item.reasons) == 2


def test_deduplicate_collection_merges_records(temp_vector_kb):
    """Testa a rotina deduplicate_collection() ao encontrar registros com IDs distintos para textos equivalentes."""
    # Insere manualmente na coleção com 2 IDs diferentes simulando o estado legado anterior
    temp_vector_kb.collection.upsert(
        ids=["id_legado_com_ponto", "7e59790e122d1085cf6431fe"],
        documents=[
            "A Anvisa proibiu a venda de lote de azeite adulterado no país.",
            "A Anvisa proibiu a venda de lote de azeite adulterado no país",
        ],
        metadatas=[
            {
                "verdict": "FAKE",
                "confidence": 0.70,
                "summary": "Resumo legado com ponto.",
                "sources_json": '["https://fonte-legada.com/1"]',
                "reasons_json": '["Razao 1"]',
                "created_at": "2026-10-01T00:00:00Z",
                "claim_type": "primary",
                "category": "PUBLIC_HEALTH",
            },
            {
                "verdict": "FAKE",
                "confidence": 0.90,
                "summary": "Resumo completo da alegação sem ponto.",
                "sources_json": '["https://fonte-nova.com/2"]',
                "reasons_json": '["Razao 2"]',
                "created_at": "2026-10-02T00:00:00Z",
                "claim_type": "primary",
                "category": "PUBLIC_HEALTH",
            },
        ],
    )
    assert temp_vector_kb.count() == 2

    # Executa a deduplicação
    res = temp_vector_kb.deduplicate_collection()
    assert res["total_before"] == 2
    assert res["total_after"] == 1
    assert res["merged"] == 1
    assert "id_legado_com_ponto" in res["removed_ids"]
    assert temp_vector_kb.count() == 1

    canonical_id = temp_vector_kb.generate_claim_id("A Anvisa proibiu a venda de lote de azeite adulterado no país")
    item = temp_vector_kb.get_claim(canonical_id)
    assert item is not None
    assert len(item.sources) == 2
    assert "https://fonte-legada.com/1" in item.sources
    assert "https://fonte-nova.com/2" in item.sources
    assert len(item.reasons) == 2
    assert item.confidence == 0.90


def test_api_deduplicate_endpoint(monkeypatch):
    """Testa o endpoint POST /api/v1/vector/deduplicate."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        test_kb = VectorClaimKB(persist_directory=tmp_dir, collection_name="api_dedup_test")
        import app.services.vector_kb as vk_module
        monkeypatch.setattr(vk_module, "_vector_kb_instance", test_kb)

        client = TestClient(app)
        res = client.post("/api/v1/vector/deduplicate")
        assert res.status_code == 200
        data = res.json()
        assert "total_before" in data
        assert "total_after" in data
        assert "merged" in data


