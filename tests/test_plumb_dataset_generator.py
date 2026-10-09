import csv
import tempfile
from pathlib import Path
import pytest

from app.services.plumb_classifier import (
    PlumbBinaryResult,
    get_plumb_classifier,
)
from app.services.plumb_dataset_generator import (
    ClaimCondenser,
    PlumbDatasetGenerator,
    PlumbDatasetRecord,
)


def test_claim_condenser_clean_raw_text():
    """Testa higienização de ruídos, marcações HTML e cabeçalhos sensacionalistas."""
    raw = "🚨 BOMBA!! <p>Confira a matéria completa em https://site.com/noticia. Foto: Divulgação.</p> Água com limão ajuda na digestão."
    clean = ClaimCondenser.clean_raw_text(raw)
    assert "<p>" not in clean
    assert "https://" not in clean
    assert "🚨" not in clean
    assert "BOMBA" not in clean
    assert "Água com limão ajuda na digestão" in clean


def test_claim_condenser_extract_lead():
    """Testa extração de alegação concisa a partir do lide jornalístico."""
    condenser = ClaimCondenser()
    title = "Anvisa aprova novo tratamento para diabetes tipo 2"
    body = (
        "A Agência Nacional de Vigilância Sanitária (Anvisa) aprovou nesta terça-feira "
        "o registro de uma nova medicação injetável para pacientes adultos com diabetes tipo 2. "
        "O produto apresentou alta eficácia nos ensaios clínicos de fase 3."
    )
    claim = condenser.extract_lead_claim(body, title=title)
    assert len(claim) >= 15
    assert len(claim) <= 220
    assert "Anvisa" in claim or "diabetes" in claim


def test_claim_condenser_strategies():
    """Testa extração com estratégias 'lead', 'spacy' e 'full'."""
    condenser = ClaimCondenser()
    text = (
        "O Ministério da Saúde iniciou campanha nacional de vacinação contra a gripe. "
        "A meta é imunizar pelo menos 90% do público-alvo prioritário em todo o país. "
        "As doses já foram distribuídas para todas as unidades básicas de saúde."
    )
    # Lead
    claims_lead = condenser.extract_claims(text, strategy="lead", max_claims=1)
    assert len(claims_lead) == 1
    assert "Ministério da Saúde" in claims_lead[0] or "vacinação" in claims_lead[0]

    # Spacy com até 2 claims
    claims_spacy = condenser.extract_claims(text, strategy="spacy", max_claims=2)
    assert 1 <= len(claims_spacy) <= 2


def test_plumb_dataset_generator_delimiter_and_column_detection():
    """Testa auto-detecção de delimitador e nome de coluna de texto."""
    with tempfile.NamedTemporaryFile("w", suffix=".tsv", delete=False) as f:
        f.write("id\tconteudo\tdata\n1\tNoticia de teste\t2026-10-09\n")
        tsv_path = f.name

    delim = PlumbDatasetGenerator.detect_delimiter(tsv_path)
    assert delim == "\t"

    headers = ["id", "conteudo", "data"]
    col = PlumbDatasetGenerator.find_text_column(headers)
    assert col == "conteudo"

    headers_alt = ["noticia_id", "materia_jornalistica", "autor"]
    col_alt = PlumbDatasetGenerator.find_text_column(headers_alt)
    assert col_alt == "materia_jornalistica"


def test_plumb_binary_classifier_live():
    """Testa a avaliação binária V/F do Plumb-4B."""
    clf = get_plumb_classifier()
    agent = clf._ensure_loaded()
    if agent is None:
        pytest.skip("Modelo Plumb-4B não disponível no ambiente")

    res = clf.evaluate_claim_binary("A Terra gira em torno do Sol.")
    assert isinstance(res, PlumbBinaryResult)
    assert res.verdict in ("V", "F")
    assert 0.0 <= res.confidence <= 1.0
    assert "V" in res.probabilities
    assert "F" in res.probabilities
    assert res.verdict == "V"


def test_plumb_dataset_generator_process_item():
    """Testa geração de registro PlumbDatasetRecord para uma notícia."""
    clf = get_plumb_classifier()
    agent = clf._ensure_loaded()
    if agent is None:
        pytest.skip("Modelo Plumb-4B não disponível no ambiente")

    generator = PlumbDatasetGenerator(classifier=clf)
    records = generator.process_item(
        item_id="101",
        raw_text="A Anvisa determinou a suspensão da venda de um lote de medicamento adulterado no Brasil.",
        title="Anvisa suspende lote de medicamento",
        strategy="lead",
    )

    assert len(records) == 1
    r = records[0]
    assert isinstance(r, PlumbDatasetRecord)
    assert r.id == "101"
    assert len(r.alegacao) > 10
    assert r.tema in ("Saúde", "Política", "Entretenimento", "Esportes", "Economia", "Outros")
    assert r.veredito in ("V", "F")
    assert 0.0 <= r.veredito_confianca <= 1.0
    assert r.prob_v >= 0.0
    assert r.prob_f >= 0.0

