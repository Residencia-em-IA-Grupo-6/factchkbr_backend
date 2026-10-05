import io
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from app.analyzers.fact_check_api import FactCheckApiAnalyzer, EvidenceItem, SourceTier
from app.schemas.analysis import Verdict
from app.services.health_kb import HealthLocalKB


@pytest.fixture
def temp_kb(tmp_path: Path) -> HealthLocalKB:
    """Cria uma instância temporária isolada do HealthLocalKB."""
    db_file = tmp_path / "test_kb.sqlite3"
    kb = HealthLocalKB(db_path=db_file)
    csv_content = (
        "TIPO_PRODUTO;NOME_PRODUTO;DATA_FINALIZACAO_PROCESSO;CATEGORIA_REGULATORIA;"
        "NUMERO_REGISTRO_PRODUTO;DATA_VENCIMENTO_REGISTRO;NUMERO_PROCESSO;CLASSE_TERAPEUTICA;"
        "EMPRESA_DETENTORA_REGISTRO;SITUACAO_REGISTRO;PRINCIPIO_ATIVO\n"
        '"MEDICAMENTO";"OZEMPIC";"30/08/2022";"Biológico";117660036;"082032";"25351";"ANTIDIABETICOS";"NOVO NORDISK";"Ativo";"semaglutida"\n'
        '"MEDICAMENTO";"CLOROQUINA SUSPENSA";"10/01/2020";"Novo";100000000;"012025";"25352";"ANTIMALARICOS";"LAB TEST";"Cancelado";"difosfato de cloroquina"\n'
    )
    stream = io.StringIO(csv_content)
    kb.populate_from_csv_stream(stream)
    return kb


def test_health_kb_query_medication(temp_kb: HealthLocalKB):
    """Verifica se a busca de medicamentos retorna os campos corretos da Anvisa."""
    meds = temp_kb.search_medication("ozempic")
    assert len(meds) == 1
    m = meds[0]
    assert m["trade_name"] == "OZEMPIC"
    assert m["active_principle"] == "semaglutida"
    assert m["therapeutic_class"] == "ANTIDIABETICOS"
    assert m["registration_status"] == "Ativo"
    assert m["company"] == "NOVO NORDISK"


def test_health_kb_inspect_inactive_medication(temp_kb: HealthLocalKB):
    """Verifica se medicamentos inativos/cancelados geram alertas regulatórios."""
    inspection = temp_kb.inspect_health_claim("Uso de cloroquina suspensa para tratamento")
    assert inspection["is_available"] is True
    assert inspection["has_health_entities"] is True
    assert len(inspection["medications_found"]) == 1
    assert len(inspection["regulatory_alerts"]) == 1
    assert "Cancelado" in inspection["regulatory_alerts"][0]


def test_health_kb_graceful_fallback_when_unavailable(tmp_path: Path):
    """
    Testa se o HealthLocalKB lida com falha de rede/ausência de banco sem lançar exceções,
    marcando is_available() como False.
    """
    db_file = tmp_path / "inexistent.sqlite3"
    kb = HealthLocalKB(db_path=db_file)

    with patch.object(kb, "bootstrap_from_anvisa", return_value=False):
        assert kb.is_available(auto_bootstrap=True) is False
        assert kb.is_available(auto_bootstrap=False) is False

        inspection = kb.inspect_health_claim("Texto qualquer sobre medicamento")
        assert inspection["is_available"] is False
        assert inspection["has_health_entities"] is False
        assert inspection["medications_found"] == []


def test_inspect_local_health_kb_in_analyzer(temp_kb: HealthLocalKB):
    """Testa se o analisador converte registros da Anvisa em EvidenceItems com SourceTier TIER1."""
    analyzer = FactCheckApiAnalyzer()

    with patch("app.services.health_kb.get_health_kb", return_value=temp_kb):
        evidences = analyzer.inspect_local_health_kb("Ozempic é eficaz para diabetes")
        assert len(evidences) >= 1
        ev = evidences[0]
        assert ev.source_name == "Anvisa (Dados Abertos Oficiais)"
        assert ev.source_tier == SourceTier.TIER1_OFFICIAL_OR_IFCN.value
        assert "semaglutida" in ev.snippet
        assert ev.is_relevant is True


@pytest.mark.asyncio
async def test_search_clinical_trials_parsing():
    """Testa o parsing de estudos e status da API v2 do ClinicalTrials.gov."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "studies": [
            {
                "protocolSection": {
                    "identificationModule": {
                        "nctId": "NCT01234567",
                        "briefTitle": "Safety and Efficacy of Semaglutide in Adults",
                    },
                    "statusModule": {
                        "overallStatus": "COMPLETED",
                    },
                    "descriptionModule": {
                        "briefSummary": "A randomized controlled phase 3 clinical trial evaluating semaglutide.",
                    },
                }
            }
        ]
    }

    mock_client = AsyncMock()
    mock_client.get.return_value = mock_resp

    analyzer = FactCheckApiAnalyzer(http_client=mock_client)
    evidences = await analyzer.search_clinical_trials("semaglutide")

    assert len(evidences) == 1
    ev = evidences[0]
    assert "NCT01234567" in ev.url
    assert "COMPLETED" in ev.title
    assert "ClinicalTrials.gov" in ev.source_name
    assert ev.source_tier == SourceTier.TIER1_OFFICIAL_OR_IFCN.value


@pytest.mark.asyncio
async def test_search_biomedical_literature_parsing():
    """Testa o parsing de publicações biomédicas do Europe PMC / PubMed."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "resultList": {
            "result": [
                {
                    "id": "35891234",
                    "source": "MED",
                    "title": "Effects of once-weekly semaglutide in patients with type 2 diabetes",
                    "journalTitle": "The Lancet",
                    "pubYear": "2023",
                    "authorString": "Smith J, et al.",
                    "abstractText": "In this randomized clinical trial, semaglutide significantly reduced HbA1c levels.",
                }
            ]
        }
    }

    mock_client = AsyncMock()
    mock_client.get.return_value = mock_resp

    analyzer = FactCheckApiAnalyzer(http_client=mock_client)
    evidences = await analyzer.search_biomedical_literature("semaglutide diabetes")

    assert len(evidences) == 1
    ev = evidences[0]
    assert "35891234" in ev.url
    assert "The Lancet" in ev.source_name
    assert "Lancet" in ev.title
    assert ev.source_tier == SourceTier.TIER1_OFFICIAL_OR_IFCN.value


@pytest.mark.asyncio
async def test_check_single_claim_fallback_to_camada2_when_local_unavailable():
    """
    Verifica a regra de negócio fundamental:
    Se a base local da Anvisa estiver indisponível, o sistema depende exclusivamente
    da Camada 2 (APIs externas) sem falhar ou gerar exceção.
    """
    mock_client = AsyncMock()
    # Simula resposta de checagem externa
    mock_fc_resp = MagicMock()
    mock_fc_resp.status_code = 200
    mock_fc_resp.json.return_value = {}
    mock_client.get.return_value = mock_fc_resp

    analyzer = FactCheckApiAnalyzer(http_client=mock_client)

    # Força indisponibilidade da base local
    unavailable_kb = MagicMock()
    unavailable_kb.is_available.return_value = False

    with patch("app.services.health_kb.get_health_kb", return_value=unavailable_kb):
        result = await analyzer.check_single_claim("Tratamento milagroso sem comprovação científica")

        assert result["anvisa_local_count"] == 0
        assert "verdict" in result
        assert result["verdict"] in (Verdict.FAKE, Verdict.INCONCLUSIVO, Verdict.SUSPEITO, Verdict.VERDADEIRO)
