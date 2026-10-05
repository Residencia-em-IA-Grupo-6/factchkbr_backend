import asyncio
import hashlib
import json
import logging
import re
from collections import OrderedDict
from typing import Any
import httpx
from pydantic import BaseModel, Field

from app.config import Settings, get_settings

logger = logging.getLogger("factchkbr.core.health_gatekeeper")

# ==============================================================================
# 1. LÉXICO E PADRÕES TEMÁTICOS DE SAÚDE & VIGILÂNCIA SANITÁRIA
# ==============================================================================

HEALTH_TERMS: set[str] = {
    # Doenças, patologias e sintomas
    "dengue", "covid", "covid-19", "coronavírus", "coronavirus", "sars", "sars-cov-2",
    "gripe", "influenza", "h1n1", "câncer", "cancer", "tumor", "tumores", "carcinoma",
    "infarto", "avc", "derrame", "diabetes", "diabético", "diabética", "hipertensão",
    "autismo", "autista", "alzheimer", "parkinson", "hiv", "aids", "hepatite",
    "tuberculose", "pneumonia", "varíola", "variola", "mpox", "malária", "malaria",
    "cólera", "colera", "zika", "chikungunya", "febre amarela", "sarampo", "rubéola",
    "rubeola", "tétano", "tetano", "meningite", "lepra", "hanseníase", "hanseniase",
    "sepse", "obesidade", "depressão", "depressao", "ansiedade", "esquizofrenia",
    "trombose", "miocardite", "pericardite", "embolia", "paralisia", "cegueira",
    "febre", "tosse", "falta de ar", "náusea", "vomito", "vômito", "diarreia",
    "infecção", "infeccao", "inflamação", "inflamacao", "alergia", "asma", "oropouche",
    "doença", "doenca", "doenças", "doencas", "enfermidade", "enfermidades",
    "epidemia", "pandemia", "surto", "surtos", "patologia", "patologias",
    "síndrome", "sindrome", "sequela", "sequelas", "mortalidade", "óbito", "obito",
    "dor no peito", "arritmia", "insuficiência renal", "cirrose",

    # Vacinas, imunização e biologia
    "vacina", "vacinas", "vacinação", "vacinacao", "imunização", "imunizacao",
    "imunizante", "imunizantes", "vacinado", "vacinada", "vacinados", "vacinadas",
    "anticorpo", "anticorpos", "imunidade", "imunológico", "imunologico", "dna", "rna",
    "microchip na vacina", "efeito da vacina", "reação da vacina",

    # Medicamentos, fármacos, terapias e toxicologia
    "remédio", "remedio", "remédios", "remedios", "medicamento", "medicamentos",
    "fármaco", "farmaco", "fármacos", "farmacos", "comprimido", "comprimidos",
    "cápsula", "capsula", "ampola", "injeção", "injecao", "antibiótico", "antibiotico",
    "antibióticos", "analgésico", "analgesico", "anti-inflamatório", "antiinflamatorio",
    "ivermectina", "cloroquina", "hidroxicloroquina", "dipirona", "paracetamol",
    "ibuprofeno", "aspirina", "insulina", "corticoide", "corticóide", "quimioterapia",
    "radioterapia", "ozônio", "ozonio", "ozonioterapia", "chá medicinal", "chá de",
    "cura", "curar", "tratamento", "terapia", "terapêutico", "terapeutico",
    "posologia", "dosagem", "dose", "doses", "efeito colateral", "efeitos colaterais",
    "efeito adverso", "reação adversa", "adulterado", "adulterada", "contaminado",
    "contaminada", "contaminação", "contaminacao", "intoxicação", "intoxicacao",
    "veneno", "tóxico", "toxico", "letal", "bula", "substância",

    # Autoridades Sanitárias, Clínicas e Profissionais de Saúde
    "anvisa", "sus", "oms", "who", "fiocruz", "butantan", "instituto butantan",
    "ministério da saúde", "ministerio da saude", "secretaria de saúde", "secretaria da saude",
    "vigilância sanitária", "vigilancia sanitaria", "saúde pública", "saude publica",
    "médico", "medico", "médica", "medica", "médicos", "medicos", "médicas", "medicas",
    "enfermeiro", "enfermeira", "enfermeiros", "hospital", "hospitais", "clínica",
    "clinica", "uti", "cti", "leito", "leitos", "internação", "internacao",
    "cirurgia", "transplante", "diagnóstico", "diagnostico", "exame", "pronto-socorro",
    "paciente", "pacientes", "saúde", "saude"
}

# Subconjunto de termos estritamente biomédicos/farmacológicos específicos
# (usado para diferenciar se há alegação médica substantiva mesmo citando político)
SPECIFIC_BIOMEDICAL_TERMS: set[str] = {
    "vacina", "vacinas", "vacinação", "dengue", "covid", "coronavírus", "câncer",
    "infarto", "avc", "autismo", "ivermectina", "cloroquina", "hidroxicloroquina",
    "dipirona", "paracetamol", "antibiótico", "adulterado", "contaminado", "intoxicação",
    "anvisa", "cura", "efeito colateral", "efeito adverso", "trombose", "miocardite",
    "oropouche", "zika", "chikungunya", "sarampo", "remédio", "medicamento", "quimioterapia"
}

# Figuras públicas e papéis políticos
RE_POLITICAL_FIGURES = re.compile(
    r"\b(?:"
    r"lula|bolsonaro|ciro(?:\s+gomes)?|moro|dilma|temer|haddad|tarcísio|tarcisio|"
    r"zema|doria|dória|nikolas|janones|boulos|"
    r"deputad[oa]s?|senador(?:a|es)?|governador(?:a|es)?|prefeit[oa]s?|vereador(?:a|es)?|"
    r"presidentes?|parlamentar(?:es)?|candidat[oa]s?|ministr[oa]s?|"
    r"partido|pt|pl|psol|mdb|psdb|centrão|centrao|oposição|oposicao|bancada"
    r")\b",
    re.IGNORECASE,
)

# Verbos de declaração / atribuição discursiva
RE_SPEECH_VERBS = re.compile(
    r"\b(?:"
    r"diz|disse|dizem|afirma|afirmou|afirmam|declarou|declara|declararam|"
    r"falou|fala|falam|discursou|discursa|discursaram|prometeu|promete|"
    r"criticou|critica|acusou|acusa|atacou|ataca|defendeu|defende|"
    r"postou|publicou|tuitou|escreveu|garantiu|garante"
    r")\b",
    re.IGNORECASE,
)

# Expressões e palavras-chave de polêmica partidária / retórica eleitoral
RE_POLITICAL_RHETORIC_CONTEXT = re.compile(
    r"\b(?:"
    r"quem não pode que morra|morra no esquecimento|para quem pode pagar|"
    r"eleição|eleições|eleicao|eleicoes|campanha eleitoral|comício|comicio|"
    r"voto|votos|votar|votação|votacao|urna|urnas|"
    r"corrupção|corrupcao|propina|desvio|mensalão|mensalao|petrolão|petrolao|"
    r"rachadinha|orçamento secreto|orcamento secreto|emenda parlamentar|"
    r"comunismo|comunista|fascismo|fascista|esquerda|direita|golpe"
    r")\b",
    re.IGNORECASE,
)


class HealthGatekeeperDecision(BaseModel):
    """Decisão estruturada do Gatekeeper Temático de Saúde."""
    is_health_topic: bool = Field(..., description="Indica se o texto aborda tema de saúde biomédica ou sanitária")
    is_political_polemic: bool = Field(default=False, description="Indica se é polêmica ou retórica política sobre saúde")
    allows_verification: bool = Field(..., description="True se e somente se o texto deve prosseguir para checagem")
    category: str = Field(..., description="BIOMEDICAL_HEALTH | PUBLIC_HEALTH | POLITICAL_POLEMIC | OUT_OF_SCOPE")
    reason: str = Field(..., description="Justificativa da decisão")
    matched_signals: list[str] = Field(default_factory=list, description="Sinais léxicos detectados")


# ==============================================================================
# 2. PROMPT DO CLASSIFICADOR SEMÂNTICO (LLM)
# ==============================================================================

GATEKEEPER_SYSTEM_PROMPT = """Você é o Gatekeeper Temático e Filtro de Escopo do FactChkBR.
Sua única responsabilidade é determinar se o texto recebido trata de um tema substantivo de SAÚDE (biomedicina, doenças, tratamentos, vacinas, medicamentos, órgãos sanitários) ou se deve ser RECUSADO por se tratar de polêmica política, retórica partidária ou outro assunto fora de escopo.

DIRETRIZES DE DECISÃO:
1. DENTRO DO ESCOPO (allows_verification: true):
   - Afirmações factuais falseáveis sobre doenças, sintomas, infecções e epidemias (ex: dengue, covid, câncer, etc.).
   - Eficácia, composição, posologia ou supostos riscos de vacinas e medicamentos (ex: 'vacina causa trombose', 'ivermectina cura covid').
   - Tratamentos, remédios caseiros ou químicos, cirurgias e procedimentos clínicos.
   - Atos regulatórios e alertas de vigilância sanitária oficial (Anvisa, Ministério da Saúde, SUS, OMS, Fiocruz) e produtos adulterados/contaminados.
   - Mesmo que cite uma figura pública, SE a alegação contiver um fato médico/biológico concreto falseável pela ciência médica (ex: 'Político X disse que vacina causa HIV'), deve ser PERMITIDO para checagem científica do fato sanitário.

2. RECUSADO / FORA DO ESCOPO - POLÍTICA (allows_verification: false):
   - Falas, citações, opiniões ou declarações atribuídas a políticos ou figuras públicas usando a palavra 'saúde' apenas como retórica, ataque, juízo de valor ou polêmica eleitoral/ideológica (Exemplo: "Lula diz que saúde é para quem pode pagar, quem não pode que morra no esquecimento", "Bolsonaro criticou hospitais em discurso", "Deputado afirma que governo cortou verba da saúde para emendas").
   - Disputas sobre corrupção, orçamentos, emendas parlamentares ou slogans de campanha.

3. RECUSADO / FORA DO ESCOPO - OUTROS TEMAS (allows_verification: false):
   - Qualquer texto sem relação direta com saúde (ex: eleições, votos, urnas, aposentadoria, INSS geral, esportes, economia, entretenimento).

FORMATO DE RESPOSTA OBRIGATÓRIO (JSON estrito):
{
  "allows_verification": true | false,
  "is_health_topic": true | false,
  "is_political_polemic": true | false,
  "category": "BIOMEDICAL_HEALTH" | "PUBLIC_HEALTH" | "POLITICAL_POLEMIC" | "OUT_OF_SCOPE",
  "reason": "Explicação sucinta de 1 ou 2 frases em português."
}"""


class HealthTopicGatekeeper:
    """
    Filtro de Escopo Temático (Gatekeeper):
    Valida a informação se e somente se o tema for estritamente relacionado à saúde
    (doenças, tratamentos, vacinas, medicamentos, vigilância sanitária).
    Descarta precocemente temas não relacionados à saúde e polêmicas/retóricas políticas.
    """

    _cache: OrderedDict[str, HealthGatekeeperDecision] = OrderedDict()
    _CACHE_MAXSIZE = 512

    def __init__(self, http_client: httpx.AsyncClient | None = None, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._owned_client = http_client is None
        self.http_client = http_client or httpx.AsyncClient(
            timeout=10.0,
            limits=httpx.Limits(max_keepalive_connections=10, max_connections=20),
        )

    async def aclose(self) -> None:
        if self._owned_client:
            await self.http_client.aclose()

    def evaluate_heuristic(self, text: str) -> HealthGatekeeperDecision | None:
        """
        Avaliação rápida baseada em regras léxicas determinísticas (CPU, 0ms).
        Retorna HealthGatekeeperDecision se a decisão for categórica, ou None para análise semântica LLM.
        """
        norm = text.lower()
        matched_health = [t for t in HEALTH_TERMS if re.search(rf"\b{re.escape(t)}\b", norm)]

        # 1. Se não houver NENHUM sinal léxico de saúde -> FORA DE ESCOPO imediato
        if not matched_health:
            return HealthGatekeeperDecision(
                is_health_topic=False,
                is_political_polemic=False,
                allows_verification=False,
                category="OUT_OF_SCOPE",
                reason="O tema da mensagem não é relacionado à saúde, medicina, doenças, vacinas, medicamentos ou vigilância sanitária.",
                matched_signals=[],
            )

        has_political_fig = bool(RE_POLITICAL_FIGURES.search(norm))
        has_speech_verb = bool(RE_SPEECH_VERBS.search(norm))
        has_pol_rhetoric = bool(RE_POLITICAL_RHETORIC_CONTEXT.search(norm))
        matched_biomedical = [t for t in SPECIFIC_BIOMEDICAL_TERMS if re.search(rf"\b{re.escape(t)}\b", norm)]

        # 2. Polêmica política com o termo 'saúde' sem conteúdo biomédico concreto
        # Exemplo: "Lula diz que saúde é para quem pode pagar, quem não pode que morra no esquecimento"
        if has_political_fig and (has_speech_verb or has_pol_rhetoric):
            if not matched_biomedical:
                return HealthGatekeeperDecision(
                    is_health_topic=False,
                    is_political_polemic=True,
                    allows_verification=False,
                    category="POLITICAL_POLEMIC",
                    reason="Declaração de cunho político/retórico atribuída a figura pública. O FactChkBR valida exclusivamente fatos biomédicos e sanitários, excluindo polêmicas político-partidárias.",
                    matched_signals=matched_health[:4],
                )

        # 3. Fato sanitário ou biomédico claro sem qualquer elemento político
        # Exemplo: "Vacina da dengue reduz internações", "Anvisa proibiu lote de azeite adulterado"
        if matched_biomedical and not has_political_fig and not has_pol_rhetoric:
            return HealthGatekeeperDecision(
                is_health_topic=True,
                is_political_polemic=False,
                allows_verification=True,
                category="BIOMEDICAL_HEALTH" if any(b in ("vacina", "dengue", "câncer", "infarto", "remédio") for b in matched_biomedical) else "PUBLIC_HEALTH",
                reason="A alegação trata de tema biomédico ou sanitário substantivo passível de validação científica.",
                matched_signals=matched_biomedical[:4],
            )

        # Casos com sobreposição ou ambiguidade: requer validação semântica LLM
        return None

    async def evaluate(self, text: str) -> HealthGatekeeperDecision:
        """
        Executa a validação de escopo temático com cache LRU:
        1. Fast-path heurístico em CPU (0ms).
        2. Classificação semântica via LLM para casos com nuances ou declarações mistas.
        """
        cache_key = hashlib.sha256(text.strip().encode("utf-8")).hexdigest()
        if cache_key in self._cache:
            self._cache.move_to_end(cache_key)
            return self._cache[cache_key]

        # 1. Tenta resolução rápida via regras determinísticas
        fast_decision = self.evaluate_heuristic(text)
        if fast_decision is not None:
            self._save_cache(cache_key, fast_decision)
            return fast_decision

        # 2. Consulta semântica via LLM (Ollama ou OpenAI)
        try:
            endpoint = self.settings.get_llm_endpoint()
            model = self.settings.get_llm_model()
            headers = self.settings.get_llm_headers()

            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": GATEKEEPER_SYSTEM_PROMPT},
                    {"role": "user", "content": f"Texto: \"{text.strip()}\""},
                ],
                "response_format": {"type": "json_object"},
                "temperature": 0.0,
            }

            resp = await self.http_client.post(endpoint, headers=headers, json=payload, timeout=12.0)
            if resp.status_code == 200:
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                parsed = json.loads(content)

                allows = bool(parsed.get("allows_verification", False))
                is_health = bool(parsed.get("is_health_topic", False))
                is_pol = bool(parsed.get("is_political_polemic", False))
                cat = str(parsed.get("category", "OUT_OF_SCOPE")).upper()
                reason = str(parsed.get("reason", "Avaliação de escopo temático."))

                # Salvaguarda: se classificado como polêmica política, força allows_verification=False
                if is_pol:
                    allows = False

                norm = text.lower()
                signals = [t for t in HEALTH_TERMS if re.search(rf"\b{re.escape(t)}\b", norm)][:4]

                decision = HealthGatekeeperDecision(
                    is_health_topic=is_health,
                    is_political_polemic=is_pol,
                    allows_verification=allows,
                    category=cat,
                    reason=reason,
                    matched_signals=signals,
                )
                self._save_cache(cache_key, decision)
                return decision

        except Exception as exc:
            logger.warning("Falha na avaliação semântica do gatekeeper via LLM (%s): %s", type(exc).__name__, exc)

        # Fallback defensivo: se LLM indisponível e texto tinha algum termo de saúde genérico com político
        norm = text.lower()
        has_political_fig = bool(RE_POLITICAL_FIGURES.search(norm))
        has_speech_verb = bool(RE_SPEECH_VERBS.search(norm))
        matched_health = [t for t in HEALTH_TERMS if re.search(rf"\b{re.escape(t)}\b", norm)]

        if has_political_fig and has_speech_verb:
            fallback = HealthGatekeeperDecision(
                is_health_topic=False,
                is_political_polemic=True,
                allows_verification=False,
                category="POLITICAL_POLEMIC",
                reason="Declaração de cunho político atribuída a figura pública (fora do escopo biomédico).",
                matched_signals=matched_health[:4],
            )
        elif matched_health:
            fallback = HealthGatekeeperDecision(
                is_health_topic=True,
                is_political_polemic=False,
                allows_verification=True,
                category="BIOMEDICAL_HEALTH",
                reason="Tema de saúde detectado com sinais biomédicos nas regras de contingência.",
                matched_signals=matched_health[:4],
            )
        else:
            fallback = HealthGatekeeperDecision(
                is_health_topic=False,
                is_political_polemic=False,
                allows_verification=False,
                category="OUT_OF_SCOPE",
                reason="Tema não relacionado à saúde biomédica ou sanitária.",
                matched_signals=[],
            )

        self._save_cache(cache_key, fallback)
        return fallback

    def _save_cache(self, key: str, value: HealthGatekeeperDecision) -> None:
        self._cache[key] = value
        if len(self._cache) > self._CACHE_MAXSIZE:
            self._cache.popitem(last=False)
