import pytest
from unittest.mock import AsyncMock, patch

from app.config import Settings
from app.core.health_gatekeeper import (
    HealthTopicGatekeeper,
    HealthGatekeeperDecision,
)
from app.core.orchestrator import FactCheckOrchestrator
from app.schemas.analysis import Verdict


@pytest.fixture
def gatekeeper():
    settings = Settings(HEALTH_ONLY_MODE=True)
    gk = HealthTopicGatekeeper(settings=settings)
    yield gk


def test_heuristic_rejection_general_non_health(gatekeeper: HealthTopicGatekeeper):
    """Rejeita mensagens gerais sem qualquer relação com saúde."""
    text1 = "voto nas eleições não pode ser utilizado como prova de vida do INSS"
    decision1 = gatekeeper.evaluate_heuristic(text1)
    assert decision1 is not None
    assert decision1.allows_verification is False
    assert decision1.is_health_topic is False
    assert decision1.category == "OUT_OF_SCOPE"

    text2 = "Flamengo vence o Palmeiras na final da Libertadores por 2 a 1"
    decision2 = gatekeeper.evaluate_heuristic(text2)
    assert decision2 is not None
    assert decision2.allows_verification is False
    assert decision2.category == "OUT_OF_SCOPE"


def test_heuristic_rejection_political_health_polemic(gatekeeper: HealthTopicGatekeeper):
    """
    Rejeita declarações e retórica política envolvendo o termo saúde,
    atendendo à exigência de evitar casos como a fala atribuída a Lula.
    """
    # Exemplo explicitamente fornecido pelo usuário
    text1 = "Lula diz que saúde é para quem pode pagar, quem não pode que morra no esquecimento"
    decision1 = gatekeeper.evaluate_heuristic(text1)
    assert decision1 is not None
    assert decision1.allows_verification is False
    assert decision1.is_political_polemic is True
    assert decision1.category == "POLITICAL_POLEMIC"

    text2 = "Bolsonaro discursou contra o ministério da saúde em comício de campanha eleitoral"
    decision2 = gatekeeper.evaluate_heuristic(text2)
    assert decision2 is not None
    assert decision2.allows_verification is False
    assert decision2.is_political_polemic is True
    assert decision2.category == "POLITICAL_POLEMIC"


def test_heuristic_approval_biomedical_and_sanitary(gatekeeper: HealthTopicGatekeeper):
    """Permite verificação imediata para afirmações médicas, vacinas e vigilância sanitária."""
    text1 = "A vacina da dengue reduz em 80% as internações hospitalares"
    decision1 = gatekeeper.evaluate_heuristic(text1)
    assert decision1 is not None
    assert decision1.allows_verification is True
    assert decision1.is_health_topic is True
    assert decision1.category in ("BIOMEDICAL_HEALTH", "PUBLIC_HEALTH")

    text2 = "Anvisa proibiu lote de azeite adulterado com substância tóxica"
    decision2 = gatekeeper.evaluate_heuristic(text2)
    assert decision2 is not None
    assert decision2.allows_verification is True
    assert decision2.category in ("BIOMEDICAL_HEALTH", "PUBLIC_HEALTH")

    text3 = "Ivermectina cura o coronavírus em pacientes graves"
    decision3 = gatekeeper.evaluate_heuristic(text3)
    assert decision3 is not None
    assert decision3.allows_verification is True

    text4 = "Cientistas descobriram que bananas muito maduras produzem um composto que destrói células cancerígenas"
    decision4 = gatekeeper.evaluate_heuristic(text4)
    assert decision4 is not None
    assert decision4.allows_verification is True
    assert decision4.is_health_topic is True
    assert decision4.category == "BIOMEDICAL_HEALTH"


@pytest.mark.asyncio
async def test_gatekeeper_caching(gatekeeper: HealthTopicGatekeeper):
    """Valida o cache em memória para chamadas repetidas."""
    text = "Vacina contra gripe causa paralisia facial"
    res1 = await gatekeeper.evaluate(text)
    assert res1.allows_verification is True

    # Segunda chamada deve ser servida diretamente pelo cache
    res2 = await gatekeeper.evaluate(text)
    assert res2 == res1
    await gatekeeper.aclose()


import httpx

@pytest.mark.asyncio
async def test_gatekeeper_llm_fallback_on_ambiguous_quote():
    """Valida a resolução semântica via LLM quando há ambiguidade na regra heurística."""
    settings = Settings(HEALTH_ONLY_MODE=True)
    gk = HealthTopicGatekeeper(settings=settings)

    # Simula resposta do LLM para caso onde político cita um fato médico
    mock_llm_content = (
        '{"allows_verification": true, "is_health_topic": true, "is_political_polemic": false, '
        '"category": "BIOMEDICAL_HEALTH", "reason": "Citação contendo alegação médica falseável sobre vacinas."}'
    )
    mock_resp = httpx.Response(
        status_code=200,
        json={"choices": [{"message": {"content": mock_llm_content}}]},
        request=httpx.Request("POST", "http://test/chat/completions"),
    )

    with patch.object(gk.http_client, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_resp

        # Força caso que retorna None na heurística (ex: sobreposição mista)
        decision = await gk.evaluate("Deputado declarou durante audiência pública que vacina contém microchip de rna")
        assert decision.allows_verification is True
        assert decision.category == "BIOMEDICAL_HEALTH"

    await gk.aclose()


@pytest.mark.asyncio
async def test_orchestrator_early_exit_on_non_health():
    """
    Valida que o orquestrador aborta imediatamente com INCONCLUSIVO (confiança 0.0)
    sem disparar analisadores se o texto não for de saúde.
    """
    settings = Settings(HEALTH_ONLY_MODE=True)
    orchestrator = FactCheckOrchestrator(settings=settings)

    step_trace = []
    def on_step(name, result, dur):
        step_trace.append(name)

    res = await orchestrator.analyze(
        "voto nas eleições não pode ser utilizado como prova de vida do INSS",
        urls=[],
        on_step=on_step,
    )

    assert res.verdict == Verdict.INCONCLUSIVO
    assert res.confidence == 0.0
    assert "fora do escopo" in res.summary.lower()
    assert step_trace == ["health_gatekeeper"]

    await orchestrator.aclose()


@pytest.mark.asyncio
async def test_orchestrator_early_exit_on_political_health_polemic():
    """
    Valida que polêmicas políticas com 'saúde' são bloqueadas logo na etapa inicial.
    """
    settings = Settings(HEALTH_ONLY_MODE=True)
    orchestrator = FactCheckOrchestrator(settings=settings)

    step_trace = []
    def on_step(name, result, dur):
        step_trace.append(name)

    res = await orchestrator.analyze(
        "Lula diz que saúde é para quem pode pagar, quem não pode que morra no esquecimento",
        urls=[],
        on_step=on_step,
    )

    assert res.verdict == Verdict.INCONCLUSIVO
    assert res.confidence == 0.0
    assert "POLITICAL_POLEMIC" in res.summary
    assert step_trace == ["health_gatekeeper"]

    await orchestrator.aclose()


def test_plumb_topic_classifier_direct():
    """Valida a inferência direta do PlumbTopicClassifier para as principais categorias."""
    from app.services.plumb_classifier import get_plumb_classifier
    clf = get_plumb_classifier()

    # 1. Saúde (mesmo sem estar no dicionário léxico tradicional)
    res_mounjaro = clf.classify("Adesivo Mounjaro - emagrecimento rápido sem passar fome")
    assert res_mounjaro is not None
    assert res_mounjaro.topic == "Saúde"
    assert res_mounjaro.is_health is True

    # 2. Política
    res_pol = clf.classify("Lula assina medida provisória que altera regras para as eleições de 2026")
    assert res_pol is not None
    assert res_pol.topic in ("Política", "Economia")
    assert res_pol.is_health is False

    # 3. Esportes
    res_esp = clf.classify("Flamengo vence por 3 a 0 no Maracanã e assume a liderança do Brasileirão")
    assert res_esp is not None
    assert res_esp.topic == "Esportes"
    assert res_esp.is_health is False

    # 4. Entretenimento
    res_ent = clf.classify("Novo filme da Marvel estreia quebrando recordes de bilheteria mundial nos cinemas")
    assert res_ent is not None
    assert res_ent.topic == "Entretenimento"
    assert res_ent.is_health is False

    # 5. Economia
    res_eco = clf.classify("Banco Central decide manter a taxa Selic em 10,50% ao ano após reunião do Copom")
    assert res_eco is not None
    assert res_eco.topic == "Economia"
    assert res_eco.is_health is False


@pytest.mark.asyncio
async def test_gatekeeper_plumb_evaluates_mounjaro_ad(gatekeeper: HealthTopicGatekeeper):
    """
    Garante que anúncios de produtos de saúde/emagrecimento (como adesivo Mounjaro)
    são aprovados pelo classificador Plumb-4B, corrigindo o falso-bloqueio léxico anterior.
    """
    text = (
        "Adesivo Mounjaro – Faça o teste Conheça o adesivo preferido das famosas. "
        "Um método natural que está viralizando por imitar os efeitos das canetinhas, "
        "auxiliando no bem-estar de forma leve e acessível. 100% Natural Nova Fórmula mais poderosa."
    )
    decision = await gatekeeper.evaluate(text)
    assert decision.allows_verification is True
    assert decision.is_health_topic is True
    assert decision.detected_topic == "Saúde"
    assert decision.category in ("BIOMEDICAL_HEALTH", "PUBLIC_HEALTH")


@pytest.mark.asyncio
async def test_gatekeeper_plumb_blocks_sports_and_entertainment(gatekeeper: HealthTopicGatekeeper):
    """
    Garante que textos de esportes e entretenimento são bloqueados com justificativa informativa.
    """
    text_esporte = "Flamengo e Palmeiras disputam o título da Copa do Brasil no próximo domingo."
    decision_esp = await gatekeeper.evaluate(text_esporte)
    assert decision_esp.allows_verification is False
    assert decision_esp.category == "OUT_OF_SCOPE"
    assert decision_esp.detected_topic == "Esportes"

    text_cinema = "Filme brasileiro vence festival de cinema internacional e recebe aplausos dos jurados."
    decision_cin = await gatekeeper.evaluate(text_cinema)
    assert decision_cin.allows_verification is False
    assert decision_cin.category == "OUT_OF_SCOPE"
    assert decision_cin.detected_topic == "Entretenimento"

