import asyncio
import hashlib
import json
import logging
import re
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any

# Permite execução direta via `python app/analyzers/claim_extractor.py`
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import httpx

from app.config import get_settings
from app.core.base import BaseAnalyzer
from app.core.registry import register_analyzer
from app.schemas.analysis import AnalyzerResult
from app.schemas.claim_extraction import (
    AtomicAssertion,
    ClaimExtractionContract,
    KnowledgeTriple,
    VerificationSourceType,
)

logger = logging.getLogger("factchkbr.analyzers.claim_extractor")

# ==============================================================================
# PADRÕES REGEX & MORFOLOGIA VERBAL (GATEKEEPER & FALLBACK OFFLINE)
# ==============================================================================

RE_ALARM_SYMBOLS = re.compile(r"[🚨⚠️💣🔥🛑📢👀⚡🇧🇷❌‼️⁉️]+")

OPINION_PATTERNS = re.compile(
    r"\b(?:eu acho|eu acredito|minha opinião|penso que|na minha visão|vergonha|absurdo|"
    r"maravilhosa|abençoe|deus abençoe|que deus|lindo demais|horroroso|incompetente e antipático)\b",
    re.IGNORECASE,
)

# Padrões de ruídos sensacionalistas, emojis e apelos à ação
NOISE_PATTERNS = re.compile(
    r"\b(?:"
    r"compartilhe[ms]?|repass[ae][ms]?|divulgu?e[ms]?|espalh[ae][ms]?|viraliz[ae][ms]?|"
    r"veja[ms]? antes que apaguem|veja[ms]?|olh[ae][ms]?|assist[ae][ms]?|acord[ae][ms]?|salv[ae][ms]?|"
    r"antes que apaguem|apaguem|"
    r"não deixe[ms]? de (?:repassar|compartilhar)|não deixe a mídia esconder|mande[ms]? para todos|leia[ms]?|"
    r"cliqu[ae][ms]?|acess[ae][ms]?|"
    r"(?:a\s+)?(?:grande\s+|tradicional\s+)?m[ií]dia\s+(?:n[aã]o\s+vai\s+(?:mostrar|noticiar|passar|falar)|esconde|n[aã]o\s+mostra|cala|abafa)|"
    r"n[aã]o\s+passa\s+na\s+tv|a\s+tv\s+n[aã]o\s+mostra|a\s+globo\s+n[aã]o\s+mostra|"
    r"bom dia|boa tarde|boa noite|ol[aá] pessoal|ol[aá] a todos|paz do senhor|"
    r"gra[cç]as a deus|am[eé]m|fwd|encaminhad[ao]|"
    r"que vergonha|que absurdo|inaceit[aá]vel|inacredit[aá]vel|isso é uma vergonha|"
    r"isso é um absurdo|parabéns aos envolvidos|deus nos livre|deus nos acuda|"
    r"vamos orar|temos que orar|oremos|lament[aá]vel|vergonhoso|"
    r"urgente|bomba|alerta|aten[cç][aã]o brasil|aten[cç][aã]o|cuidado|"
    r"voc[eê] sabia|voc[eê]s sabiam|at[eé] quando|o que acham|ser[aá] verdade|ser[aá] que"
    r")\b",
    re.IGNORECASE,
)

# Voz Passiva Analítica: auxiliar + particípio
PASSIVE_VOICE_PATTERN = re.compile(
    r"\b(?:foi|foram|é|são|será|serão|está sendo|estão sendo|acaba de ser|acabou de ser)\s+"
    r"(?:[a-zA-ZáéíóúâêîôûãõçÁÉÍÓÚÂÊÎÔÛÃÕÇ]+(?:ado|ada|ados|adas|ido|ida|idos|idas)|"
    r"preso|presa|presos|presas|morto|morta|mortos|mortas|eleito|eleita|eleitos|eleitas|"
    r"feito|feita|feitos|feitas|dito|dita|ditos|ditas|visto|vista|vistos|vistas|"
    r"descoberto|descoberta|descobertos|descobertas|suspenso|suspensa|suspensos|suspensas|"
    r"confiscado|confiscada|confiscados|confiscadas|interceptado|interceptada)\b",
    re.IGNORECASE,
)

# Locuções Verbais de Ação
VERBAL_PERIPHRASIS_PATTERN = re.compile(
    r"\b(?:está|estão|estava|estavam|vem|vêm|vinha|vinham|começou a|começaram a|acabou de|acabaram de|"
    r"tentou|tentaram|pretende|pretendem|decidiu|decidiram|vai|vão|podem?)\s+"
    r"[a-zA-ZáéíóúâêîôûãõçÁÉÍÓÚÂÊÎÔÛÃÕÇ]+(?:ando|endo|indo|ar|er|ir)\b",
    re.IGNORECASE,
)

# Sufixos Verbais Indicativos
PAST_INDICATIVE_SUFFIXES = re.compile(
    r"\b[a-zA-ZáéíóúâêîôûãõçÁÉÍÓÚÂÊÎÔÛÃÕÇ]{3,}(?:ou|aram|eu|eram|iu|iram)\b",
    re.IGNORECASE,
)
IMPERFECT_INDICATIVE_SUFFIXES = re.compile(
    r"\b[a-zA-ZáéíóúâêîôûãõçÁÉÍÓÚÂÊÎÔÛÃÕÇ]{3,}(?:ava|avam|ia|iam)\b",
    re.IGNORECASE,
)
FUTURE_INDICATIVE_SUFFIXES = re.compile(
    r"\b[a-zA-ZáéíóúâêîôûãõçÁÉÍÓÚÂÊÎÔÛÃÕÇ]{3,}(?:ará|arão|erá|erão|irá|irão)\b",
    re.IGNORECASE,
)

# Verbos Irregulares no Passado
PAST_IRREGULAR_VERBS = re.compile(
    r"\b(?:disse|disseram|teve|tiveram|esteve|estiveram|fez|fizeram|deu|deram|"
    r"pôs|puseram|trouxe|trouxeram|veio|vieram|viu|viram|ouviu|ouviram|"
    r"quis|quiseram|pôde|puderam|soube|souberam|houve)\b",
    re.IGNORECASE,
)

# Verbos de Ação / Declaração / Fato no Presente do Indicativo
PRESENT_DECLARATIVE_VERBS = re.compile(
    r"\b(?:"
    r"extrai|extraem|come|comem|consome|consomem|bebe|bebem|ingere|ingerem|"
    r"retira|retiram|remove|removem|opera|operam|injeta|injetam|aplica|aplicam|"
    r"toma|tomam|usa|usam|engole|engolem|vomita|vomitam|infecta|infectam|"
    r"contamina|contaminam|transmite|transmitem|atinge|atingem|afeta|afetam|"
    r"cura|curam|mata|matam|morre|morrem|falece|falecem|sofre|sofrem|"
    r"encontra|encontram|descobre|descobrem|mostra|mostram|grava|gravam|"
    r"filma|filmam|flagra|flagram|"
    r"contrai|contraem|atrai|atraem|distrai|distraem|subtrai|subtraem|sai|saem|cai|caem|"
    r"faz|fazem|diz|dizem|traz|trazem|vai|vão|vem|vêm|vê|veem|ouve|ouvem|"
    r"quer|querem|pode|podem|sabe|sabem|tem|têm|é|são|está|estão|dá|dão|põe|põem|"
    r"altera|alteram|causa|causam|provoca|provocam|gera|geram|destr[oó]i|destroem|"
    r"afirma|afirmam|declara|declaram|revela|revelam|confirma|confirmam|"
    r"esconde|escondem|publica|publicam|pro[ií]be|pro[ií]bem|autoriza|autorizam|"
    r"aprova|aprovam|cancela|cancelam|aumenta|aumentam|reduz|reduzem|"
    r"cobra|cobram|compra|compram|vende|vendem"
    r")\b",
    re.IGNORECASE,
)

NON_VERB_SUFFIX_EXCLUSIONS = {
    "museu", "troféu", "judeu", "breu", "plebeu", "céu", "meu", "seu", "teu",
    "ouro", "touro", "louro", "besouro", "show", "fuzil", "barril", "gentil", "abril",
    "trava", "brava", "escrava", "oitava", "dia", "guia", "bacia", "magia", "copia",
    "padaria", "farmácia", "drogaria", "delegacia", "mídia", "notícia", "família",
    "polícia", "estratégia", "maioria", "minoria", "energia", "pandemia", "indústria",
    "maracujá", "guaraná", "pará", "alvará", "carajá", "tamanduá", "jacarandá",
}

CORE_ORGANIZATIONS = {
    "stf", "tse", "stj", "tcu", "anvisa", "pf", "prf", "ibge", "mec", "sus", "fiocruz",
    "petrobras", "receita federal", "governo federal", "governo", "congresso", "senado",
    "câmara dos deputados", "ministério da saúde", "ministério da justiça", "ministério",
    "banco central", "bacen", "inss", "anatel", "ans", "aneel", "antt", "cvm", "ibama",
}

CORE_ROLES_OR_PERSONS = {
    "médico", "médicos", "doutor", "doutores", "cirurgião", "cirurgiões", "cientista",
    "cientistas", "especialista", "especialistas", "pesquisador", "pesquisadores",
    "ministro", "ministros", "presidente", "governador", "prefeito", "senador", "deputado",
    "juiz", "juízes", "delegado", "policial", "policiais", "alexandre de moraes", "moraes",
    "bolsonaro", "lula", "tarcísio", "sérgio moro", "moro",
}

LOCATIONS = {
    "brasil", "brasília", "são paulo", "rio de janeiro", "minas gerais", "bahia",
    "paraná", "sul", "nordeste", "sudeste", "norte", "centro-oeste", "eua", "china",
}


def infer_source_types(text: str, entities: dict[str, list[str]]) -> list[VerificationSourceType]:
    """Infere tipos de fontes esperadas a partir de entidades e vocabulário semântico."""
    sources: list[VerificationSourceType] = []
    combined = (text + " " + " ".join(e for cat in entities.values() for e in cat)).lower()

    if any(k in combined for k in ("anvisa", "anatel", "bacen", "aneel", "ans", "antt", "cvm", "ibama")):
        sources.append(VerificationSourceType.AGENCIA_REGULADORA)
    if any(k in combined for k in ("ministério", "governo", "receita federal", "diário oficial", "prefeitura", "presidência")):
        sources.append(VerificationSourceType.ORGAO_OFICIAL)
    if any(k in combined for k in ("stf", "stj", "tse", "tcu", "cnj", "justiça", "tribunal", "moraes", "juiz", "vara", "moro", "sérgio moro")):
        sources.append(VerificationSourceType.PODER_JUDICIARIO)
    if any(k in combined for k in ("médico", "doutor", "fiocruz", "ibge", "ipea", "inpe", "usp", "universidade", "estudo", "pesquisa", "hospital", "cirurgia")):
        sources.append(VerificationSourceType.INSTITUTO_PESQUISA)
    if any(k in combined for k in ("vacina", "remédio", "remédios", "medicamento", "saúde", "infarto", "coração", "lote", "adulterado")):
        if VerificationSourceType.AGENCIA_REGULADORA not in sources:
            sources.append(VerificationSourceType.AGENCIA_REGULADORA)
        if VerificationSourceType.ORGAO_OFICIAL not in sources:
            sources.append(VerificationSourceType.ORGAO_OFICIAL)

    if not sources:
        sources = [VerificationSourceType.DADOS_PUBLICOS, VerificationSourceType.AGENCIA_CHECAGEM]
    return sources


def map_source_type(val: Any) -> VerificationSourceType | None:
    """Mapeia strings flexíveis ou alucinações de LLM para valores estritos do enum."""
    if isinstance(val, VerificationSourceType):
        return val
    if not isinstance(val, str):
        return None
    normalized = val.strip().upper().replace(" ", "_").replace("-", "_")
    for st in VerificationSourceType:
        if st.value == normalized:
            return st
    val_lower = val.lower()
    if any(k in val_lower for k in ("médic", "doutor", "saúde", "pesquisa", "hospital", "estudo", "cirurgia")):
        return VerificationSourceType.INSTITUTO_PESQUISA
    if any(k in val_lower for k in ("tribunal", "juiz", "vara", "justiça", "judiciário", "stf", "moro")):
        return VerificationSourceType.PODER_JUDICIARIO
    if any(k in val_lower for k in ("anvisa", "regulador", "agência reguladora", "anatel")):
        return VerificationSourceType.AGENCIA_REGULADORA
    if any(k in val_lower for k in ("governo", "oficial", "ministério", "diário", "receita")):
        return VerificationSourceType.ORGAO_OFICIAL
    if any(k in val_lower for k in ("checagem", "fact-check", "lupa", "fatos")):
        return VerificationSourceType.AGENCIA_CHECAGEM
    return VerificationSourceType.DADOS_PUBLICOS


# ==============================================================================
# 1. PRÉ-PROCESSAMENTO & GATEKEEPER VIA SPACY / MORFOLOGIA (CPU)
# ==============================================================================

class SpacyPreprocessor:
    """Higienização de ruídos residuais, extração de entidades e gatekeeping sintático."""

    def __init__(self) -> None:
        self.nlp = None
        try:
            import spacy
            for model in ("pt_core_news_sm", "pt_core_news_md", "pt_core_news_lg"):
                try:
                    self.nlp = spacy.load(model, exclude=["lemmatizer"])
                    logger.info("spaCy carregado com sucesso para NER e triagem: %s", model)
                    break
                except Exception:
                    continue
        except ImportError:
            logger.debug("spaCy não disponível no ambiente. Utilizando gatekeeper morfológico determinístico.")

    def clean_text(self, text: str) -> str:
        """Remove emojis de alarme, pontuações de pânico e decanta alertas sensacionalistas."""
        t = re.sub(r"[🚨⚠️💣🔥🛑📢👀⚡🇧🇷❌‼️⁉️]", " ", text)
        t = re.sub(r"\?{2,}", "?", t)
        t = re.sub(r"!{2,}", ".", t)
        t = re.sub(r"\.{2,}", ".", t)
        t = re.sub(r"\s+", " ", t).strip()

        # Decapita cabeçalhos de alarme isolados no início
        t = re.sub(
            r"^(?:(?:URGENTE|BOMBA|ALERTA|ATENÇÃO BRASIL)[!.:\s-]*)+",
            "",
            t,
            flags=re.IGNORECASE,
        ).strip()
        t = re.sub(r"^[\s,;.:-]+", "", t).strip()
        return t

    def extract_entities(self, text: str) -> dict[str, list[str]]:
        """Identifica entidades nomeadas categorizadas (PER, ORG, LOC)."""
        entities: dict[str, list[str]] = {"PER": [], "ORG": [], "LOC": []}
        if not text:
            return entities

        # 1. Se spaCy estiver carregado, utiliza o modelo estatístico
        if self.nlp is not None:
            try:
                doc = self.nlp(text)
                for ent in doc.ents:
                    lbl = ent.label_
                    val = ent.text.strip()
                    if lbl in entities and val not in entities[lbl]:
                        entities[lbl].append(val)
                return entities
            except Exception:
                pass

        # 2. Heurística determinística complementar / fallback
        words = text.split()
        for idx, w in enumerate(words):
            w_clean = re.sub(r"^[^\w]+|[^\w]+$", "", w)
            w_lower = w_clean.lower()

            if w_lower in CORE_ORGANIZATIONS or (w_clean.isupper() and len(w_clean) >= 2 and w_lower not in NON_VERB_SUFFIX_EXCLUSIONS):
                if w_clean not in entities["ORG"]:
                    entities["ORG"].append(w_clean)
            elif w_lower in CORE_ROLES_OR_PERSONS:
                if w_clean.capitalize() not in entities["PER"]:
                    entities["PER"].append(w_clean.capitalize())
            elif w_lower in LOCATIONS:
                if w_clean.capitalize() not in entities["LOC"]:
                    entities["LOC"].append(w_clean.capitalize())

        # Expressões compostas conhecidas
        t_lower = text.lower()
        if "sérgio moro" in t_lower or "sergio moro" in t_lower:
            if "Sérgio Moro" not in entities["PER"]:
                entities["PER"].append("Sérgio Moro")
        if "ministério da saúde" in t_lower and "Ministério da Saúde" not in entities["PER"]:
            entities["PER"].append("Ministério da Saúde")
        if "alexandre de moraes" in t_lower and "Alexandre de Moraes" not in entities["PER"]:
            entities["PER"].append("Alexandre de Moraes")
        if "receita federal" in t_lower and "Receita Federal" not in entities["ORG"]:
            entities["ORG"].append("Receita Federal")
        if "são paulo" in t_lower and "São Paulo" not in entities["LOC"]:
            entities["LOC"].append("São Paulo")
        if "brasília" in t_lower and "Brasília" not in entities["LOC"]:
            entities["LOC"].append("Brasília")

        return entities

    def analyze(self, text: str) -> tuple[dict[str, list[str]], bool]:
        """
        Processa o texto em único passe (CPU):
        - Extrai entidades (PER, ORG, LOC).
        - Atua como gatekeeper: retorna True se houver estrutura sintática mínima
          para formular uma alegação factual falseável (verbo + substantivo/entidade).
        """
        entities = self.extract_entities(text)
        if not text or len(text.strip()) < 8:
            return entities, False

        # Descarte antecipado de opiniões ou cumprimentos puros
        if OPINION_PATTERNS.search(text) and ("acho" in text.lower() or "deus" in text.lower() or "vergonha" in text.lower()):
            return entities, False

        # 1. Se spaCy disponível, validação via POS tags
        if self.nlp is not None:
            doc = self.nlp(text)
            has_verb = any(t.pos_ in ("VERB", "AUX") for t in doc)
            has_content_nominal = any(t.pos_ in ("NOUN", "PROPN", "NUM") for t in doc) or any(entities.values())
            has_min_length = len(doc) >= 4
            is_viable = has_verb and has_content_nominal and has_min_length
            return entities, is_viable

        # 2. Gatekeeper determinístico morfológico (caso spaCy não esteja instalado)
        verb_candidates = (
            list(PASSIVE_VOICE_PATTERN.finditer(text)) +
            list(VERBAL_PERIPHRASIS_PATTERN.finditer(text)) +
            list(PAST_INDICATIVE_SUFFIXES.finditer(text)) +
            list(IMPERFECT_INDICATIVE_SUFFIXES.finditer(text)) +
            list(FUTURE_INDICATIVE_SUFFIXES.finditer(text)) +
            list(PRESENT_DECLARATIVE_VERBS.finditer(text)) +
            list(PAST_IRREGULAR_VERBS.finditer(text))
        )
        has_verb = any(m.group(0).lower() not in NON_VERB_SUFFIX_EXCLUSIONS for m in verb_candidates)
        tokens = text.split()
        has_content = has_verb and (len(tokens) >= 4 or any(entities.values()))

        return entities, bool(has_content)


# Alias para retrocompatibilidade com suítes de teste existentes
SpacyNERCleaner = SpacyPreprocessor


# ==============================================================================
# 2. MOTOR DE DECOMPOSIÇÃO BASEADO EM LLM COM CACHE E FALLBACK RESILIENTE
# ==============================================================================

SYSTEM_PROMPT = """Você é um motor analítico especializado em extração, normalização e decomposição de alegações factuais para sistemas automatizados de checagem de fatos (Fact-Checking Pipeline).
Sua única responsabilidade é processar textos pré-filtrados e decompô-los em proposições atômicas, falseáveis e independentes.

### DIRETRIZES FUNDAMENTAIS:
1. ATOMICIDADE:
   - Divida alegações compostas em afirmações unitárias. Cada fato deve poder ser classificado como 'Verdadeiro' ou 'Falso' de forma totalmente independente.

2. SEPARAÇÃO RIGOROSA DE ATRIBUIÇÃO (CITAÇÃO vs. CONTEÚDO):
   - Se o texto afirma que uma entidade declarou algo (ex: 'X disse que Y aconteceu'), gere OBRIGATORIAMENTE duas alegações separadas:
     a) A alegação de atribuição/fala: se X realmente declarou aquilo.
     b) A alegação de mérito: se Y realmente aconteceu no mundo real.

3. NORMALIZAÇÃO SEMÂNTICA SEM ALUCINAÇÃO:
   - Elimine hipérboles, sensacionalismo e exclamações.
   - Converta termos informais ou coloquiais para linguagem formal e objetiva.
   - NUNCA invente fatos ausentes. Preserve estritamente entidades, locais, datas e números informados no texto original.
   - Se uma fonte for vaga ("médicos afirmam"), preserve a fonte genérica ("médicos não identificados").

4. FORMATAÇÃO E ESTRUTURA:
   - Responda EXCLUSIVAMENTE em formato JSON com a chave raiz 'assertions'.
   - Cada objeto deve conter 'id', 'statement', 'triple' (com subject, predicate, object), 'suggested_source_types' e 'is_check_worthy'."""

DECOMPOSITION_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "assertions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "statement": {"type": "string"},
                    "triple": {
                        "type": "object",
                        "properties": {
                            "subject": {"type": "string"},
                            "predicate": {"type": "string"},
                            "object": {"type": "string"},
                        },
                        "required": ["subject", "predicate", "object"],
                        "additionalProperties": False,
                    },
                    "suggested_source_types": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": [
                                "ORGAO_OFICIAL",
                                "AGENCIA_REGULADORA",
                                "PODER_JUDICIARIO",
                                "INSTITUTO_PESQUISA",
                                "AGENCIA_CHECAGEM",
                                "DADOS_PUBLICOS",
                            ],
                        },
                    },
                    "is_check_worthy": {"type": "boolean"},
                },
                "required": ["id", "statement", "triple", "is_check_worthy"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["assertions"],
    "additionalProperties": False,
}


class LLMClaimDecomposer:
    """Decomposição semântica e estruturação via LLM com cache LRU em memória e fallback resiliente."""

    def __init__(self, http_client: httpx.AsyncClient | None = None, cache_maxsize: int = 1024) -> None:
        self.settings = get_settings()
        self._owned_client = http_client is None
        self.http_client = http_client or httpx.AsyncClient(
            timeout=45.0 if self.settings.LLM_PROVIDER.lower() == "ollama" else 15.0,
            limits=httpx.Limits(max_keepalive_connections=25, max_connections=50),
        )
        self._cache_maxsize = cache_maxsize
        self._cache: OrderedDict[str, list[AtomicAssertion]] = OrderedDict()

    def _get_from_cache(self, key: str) -> list[AtomicAssertion] | None:
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def _save_to_cache(self, key: str, assertions: list[AtomicAssertion]) -> None:
        self._cache[key] = assertions
        if len(self._cache) > self._cache_maxsize:
            self._cache.popitem(last=False)

    async def aclose(self) -> None:
        """Encerra cliente HTTP se instanciado internamente."""
        if self._owned_client:
            await self.http_client.aclose()

    def _extract_triple_syntactic(self, clause: str, entities: dict[str, list[str]]) -> KnowledgeTriple:
        """Extrai tripla semântica sujeito-predicado-objeto via morfologia."""
        clean_s = clause.strip(" ,;.:-")
        subject = "Fato reportado"
        for cat in ("PER", "ORG", "LOC"):
            for ent in entities.get(cat, []):
                if ent.lower() in clean_s.lower():
                    subject = ent
                    break
            if subject != "Fato reportado":
                break

        if subject == "Fato reportado":
            np_match = re.match(
                r"^(?:O|A|Os|As|Um|Uma)?\s*([a-zA-ZáéíóúâêîôûãõçÁÉÍÓÚÂÊÎÔÛÃÕÇ]+(?:\s+[a-zA-ZáéíóúâêîôûãõçÁÉÍÓÚÂÊÎÔÛÃÕÇ]+){0,3})\b",
                clean_s,
            )
            if np_match:
                subject = np_match.group(0).strip()

        predicate = "afirmação sobre"
        verb_candidates = (
            list(PASSIVE_VOICE_PATTERN.finditer(clean_s)) +
            list(VERBAL_PERIPHRASIS_PATTERN.finditer(clean_s)) +
            list(PAST_INDICATIVE_SUFFIXES.finditer(clean_s)) +
            list(IMPERFECT_INDICATIVE_SUFFIXES.finditer(clean_s)) +
            list(FUTURE_INDICATIVE_SUFFIXES.finditer(clean_s)) +
            list(PRESENT_DECLARATIVE_VERBS.finditer(clean_s)) +
            list(PAST_IRREGULAR_VERBS.finditer(clean_s))
        )
        valid_verbs = [m for m in verb_candidates if m.group(0).lower() not in NON_VERB_SUFFIX_EXCLUSIONS]
        if valid_verbs:
            first_verb = sorted(valid_verbs, key=lambda m: m.start())[0]
            v_start = first_verb.start()
            v_end = first_verb.end()
            predicate = clean_s[v_start:v_end].strip()
            after_verb = clean_s[v_end:].strip()
            obj = after_verb if len(after_verb) > 2 else "ocorrência descrita"
        else:
            obj = clean_s

        return KnowledgeTriple(subject=subject, predicate=predicate, object=obj)

    def decompose_syntactic(
        self,
        cleaned_text: str,
        entities: dict[str, list[str]],
    ) -> tuple[list[AtomicAssertion], list[str]]:
        """Decomposição sintática e morfológica determinística (fallback autossuficiente)."""
        raw_sentences = [s.strip() for s in re.split(r"[.!?\n]+", cleaned_text) if s.strip()]
        clause_splitter = re.compile(
            r"\b(?:porque|por que|já que|visto que|pois|mas|porém|contudo|todavia|enquanto|e que)\b",
            re.IGNORECASE,
        )

        atomic_candidates: list[str] = []
        discarded: list[str] = []

        for sent in raw_sentences:
            if NOISE_PATTERNS.search(sent) and (len(sent.split()) <= 6 or "mídia" in sent.lower() or "tv" in sent.lower() or "apaguem" in sent.lower() or "repassem" in sent.lower()):
                discarded.append(sent)
                continue

            sub_clauses = [c.strip() for c in clause_splitter.split(sent) if c.strip()]
            for clause in sub_clauses:
                cleaned_clause = re.sub(
                    r"^(?:veja|olhe|compartilhe|repassem|atenção brasil|atenção|urgente|bomba)[!:\s,-]*",
                    "",
                    clause,
                    flags=re.IGNORECASE,
                ).strip(" ,;.:-")

                if not cleaned_clause:
                    continue

                if NOISE_PATTERNS.search(cleaned_clause) and (len(cleaned_clause.split()) <= 6 or "apaguem" in cleaned_clause.lower() or "mídia" in cleaned_clause.lower()):
                    discarded.append(cleaned_clause)
                    continue

                verb_candidates = (
                    list(PASSIVE_VOICE_PATTERN.finditer(cleaned_clause)) +
                    list(VERBAL_PERIPHRASIS_PATTERN.finditer(cleaned_clause)) +
                    list(PAST_INDICATIVE_SUFFIXES.finditer(cleaned_clause)) +
                    list(IMPERFECT_INDICATIVE_SUFFIXES.finditer(cleaned_clause)) +
                    list(FUTURE_INDICATIVE_SUFFIXES.finditer(cleaned_clause)) +
                    list(PRESENT_DECLARATIVE_VERBS.finditer(cleaned_clause)) +
                    list(PAST_IRREGULAR_VERBS.finditer(cleaned_clause))
                )
                has_verb = any(m.group(0).lower() not in NON_VERB_SUFFIX_EXCLUSIONS for m in verb_candidates)

                if has_verb and len(cleaned_clause.split()) >= 3:
                    atomic_candidates.append(cleaned_clause)
                else:
                    discarded.append(cleaned_clause)

        if not atomic_candidates and cleaned_text and not OPINION_PATTERNS.search(cleaned_text):
            atomic_candidates = [cleaned_text]

        sources = infer_source_types(cleaned_text, entities)
        assertions: list[AtomicAssertion] = []
        for idx, candidate in enumerate(atomic_candidates, 1):
            triple = self._extract_triple_syntactic(candidate, entities)
            assertions.append(
                AtomicAssertion(
                    id=idx,
                    statement=candidate,
                    triple=triple,
                    suggested_source_types=sources,
                    is_check_worthy=True,
                )
            )
        return assertions, discarded

    async def decompose(
        self,
        cleaned_text: str,
        entities: dict[str, list[str]],
    ) -> tuple[list[AtomicAssertion], str, list[str]]:
        """
        Decompõe texto com verificação de cache LRU, Structured Outputs na LLM
        e fallback transparente para motor determinístico em caso de indisponibilidade.
        """
        cache_key = hashlib.sha256(cleaned_text.encode("utf-8")).hexdigest()
        cached = self._get_from_cache(cache_key)
        if cached is not None:
            return cached, f"cache:{self.settings.get_llm_model()}", []

        user_prompt = (
            f"Texto: \"{cleaned_text}\"\n"
            f"Entidades pré-detectadas: {json.dumps(entities, ensure_ascii=False)}\n\n"
            "Decomponha o texto acima em proposições atômicas, separando citações/declarações do conteúdo factual subjacente "
            "e normalizando a linguagem para termos objetivos."
        )

        payload = {
            "model": self.settings.get_llm_model(),
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "claim_decomposition",
                    "strict": True,
                    "schema": DECOMPOSITION_JSON_SCHEMA,
                },
            },
            "temperature": 0.0,
        }

        # Se for OpenAI mas não houver chave configurada, utiliza fallback direto
        provider = self.settings.LLM_PROVIDER.lower()
        if provider == "openai" and not self.settings.OPENAI_API_KEY:
            fallback, discarded = self.decompose_syntactic(cleaned_text, entities)
            return fallback, "syntactic_fallback", discarded

        try:
            resp = await self.http_client.post(
                self.settings.get_llm_endpoint(),
                headers=self.settings.get_llm_headers(),
                json=payload,
            )

            # Se json_schema não for aceito pelo modelo local, tenta modo json_object
            if resp.status_code == 400:
                payload["response_format"] = {"type": "json_object"}
                resp = await self.http_client.post(
                    self.settings.get_llm_endpoint(),
                    headers=self.settings.get_llm_headers(),
                    json=payload,
                )

            if resp.status_code == 200:
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                parsed = json.loads(content)
                raw_assertions = parsed.get("assertions", [])
                inferred_sources = infer_source_types(cleaned_text, entities)

                assertions: list[AtomicAssertion] = []
                for item in raw_assertions:
                    cleaned_sources: list[str] = []
                    for s in item.get("suggested_source_types", []):
                        mapped = map_source_type(s)
                        if mapped and mapped.value not in cleaned_sources:
                            cleaned_sources.append(mapped.value)
                    if not cleaned_sources:
                        cleaned_sources = [s.value for s in inferred_sources]
                    item["suggested_source_types"] = cleaned_sources
                    assertions.append(AtomicAssertion.model_validate(item))

                if assertions:
                    self._save_to_cache(cache_key, assertions)
                    return assertions, f"llm:{self.settings.get_llm_model()}", []

        except Exception as e:
            logger.debug("LLM offline ou inacessível (%s). Ativando fallback determinístico.", e)

        # Fallback determinístico offline (garante resiliência e testes verdes)
        fallback_assertions, discarded = self.decompose_syntactic(cleaned_text, entities)
        return fallback_assertions, "syntactic_fallback", discarded


# ==============================================================================
# 3. ANALISADOR REGISTRADO (BaseAnalyzer)
# ==============================================================================

@register_analyzer("claim_extractor", weight=0.0)
class ClaimExtractorAnalyzer(BaseAnalyzer):
    """
    Analisador de Extração Factual em 2 Etapas:
    1. Higienização & Gatekeeper (spaCy/Morfologia em CPU - descarta não-fatos a custo zero).
    2. Decomposição Semântica Atômica (LLM com Structured Outputs, Cache LRU e Fallback Resiliente).
    """

    def __init__(self) -> None:
        super().__init__()
        self.settings = get_settings()
        self.preprocessor = SpacyPreprocessor()
        self.http_client = httpx.AsyncClient(
            timeout=45.0 if self.settings.LLM_PROVIDER.lower() == "ollama" else 15.0,
            limits=httpx.Limits(max_keepalive_connections=25, max_connections=50),
        )
        self.decomposer = LLMClaimDecomposer(self.http_client)

    async def aclose(self) -> None:
        """Encerra o pool de conexões HTTP ao finalizar a aplicação."""
        await self.http_client.aclose()

    async def extract_contract(self, text: str) -> ClaimExtractionContract:
        """Executa o pipeline em camadas e produz o contrato estrito Pydantic."""
        cleaned = self.preprocessor.clean_text(text)
        entities, is_viable = self.preprocessor.analyze(cleaned)

        # Camada 1 (Gatekeeper): Se não houver estrutura mínima factual, descarta antecipadamente
        if not is_viable:
            return ClaimExtractionContract(
                original_text=text,
                cleaned_text=cleaned,
                entities=entities,
                assertions=[],
                discarded_fragments=[cleaned] if cleaned else [],
                engine_used="spacy_gatekeeper_short_circuit",
            )

        # Camada 2: Decomposição Atômica via LLM / Fallback
        assertions, engine_used, discarded = await self.decomposer.decompose(cleaned, entities)

        return ClaimExtractionContract(
            original_text=text,
            cleaned_text=cleaned,
            entities=entities,
            assertions=assertions,
            discarded_fragments=discarded,
            engine_used=engine_used,
        )

    async def analyze(self, text: str, urls: list[str]) -> AnalyzerResult:
        """Interface padrão do BaseAnalyzer consumida pelo FactCheckOrchestrator."""
        contract = await self.extract_contract(text)
        primary_assertion = contract.assertions[0] if contract.assertions else None
        primary_claim = primary_assertion.statement if primary_assertion else None

        return AnalyzerResult(
            analyzer_name="claim_extractor",
            verdict=None,
            confidence=0.0,
            claim=primary_claim,
            summary=(
                f"{len(contract.assertions)} proposição(ões) atômica(s) identificada(s)."
                if primary_claim
                else "Nenhuma asserção factual identificada (ruído, opinião ou conversação)."
            ),
            reasons=[f"Claim principal: \"{primary_claim}\""] if primary_claim else ["Sem predicado factual checável."],
            sources=["Extrator FactChkBR (spaCy Gatekeeper + LLM Structured Outputs)"],
            raw_details={
                "contract": contract.model_dump(),
                "engine_used": contract.engine_used,
                "assertions": [a.model_dump() for a in contract.assertions],
                "entities": contract.entities,
                "extracted_claim": primary_claim or "",
                "claims_found": len(contract.assertions),
            },
        )


# ==============================================================================
# CLI E FORMATAÇÃO VISUAL PARA TESTES
# ==============================================================================

def format_cli_result(result: AnalyzerResult) -> None:
    """Imprime relatório estruturado e legível no terminal."""
    raw = result.raw_details
    contract = raw.get("contract", {})
    assertions = raw.get("assertions", [])
    entities = raw.get("entities", {})
    engine = raw.get("engine_used", "desconhecido")

    print("\n" + "=" * 75)
    if "llm" in engine:
        print(f"⚙️  MOTOR EXECUTOR: LLM Local ({engine})")
    elif "gatekeeper" in engine:
        print(f"⚙️  MOTOR EXECUTOR: Gatekeeper Sintático ({engine})")
    else:
        print(f"⚙️  MOTOR EXECUTOR: Motor Algorítmico / Sintático (Fallback Offline)")

    if result.claim:
        print(f"🎯 ALEGAÇÃO PRINCIPAL ISOLADA (CLAIM):\n   👉 \"{result.claim}\"\n")
    else:
        print("⚠️  NENHUMA ALEGAÇÃO FACTUAL IDENTIFICADA (Texto opinativo/saudação/ruído)\n")

    if assertions:
        print(f"🧬 PROPOSIÇÕES ATÔMICAS & TRIPLAS (SPO) [{len(assertions)}]:")
        for a in assertions:
            idx = a.get("id", "-")
            stmt = a.get("statement", "")
            triple = a.get("triple", {})
            s = triple.get("subject", "?")
            p = triple.get("predicate", "?")
            o = triple.get("object", "?")
            sources = a.get("suggested_source_types", [])
            print(f"   [{idx}] Statement: \"{stmt}\"")
            print(f"       ├─ Tripla:  ({s} ➔ {p} ➔ {o})")
            print(f"       └─ Fontes:  {sources}")
        print()

    if any(entities.values()):
        print("🏷️  ENTIDADES IDENTIFICADAS (NER):")
        for category, items in entities.items():
            if items:
                print(f"   • {category}: {items}")
        print()

    discarded = contract.get("discarded_fragments", [])
    if discarded:
        print(f"🗑️  FRAGMENTOS/RUÍDOS DESCARTADOS ({len(discarded)}):")
        for d in discarded:
            print(f"   ✖ \"{d}\"")
        print()

    print(f"📊 MÉTRICAS:")
    print(f"   • Proposições atômicas: {raw.get('claims_found', 0)}")
    print(f"   • Fragmentos de ruído:  {len(discarded)}")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    import select
    import warnings

    warnings.filterwarnings("ignore", category=RuntimeWarning)

    async def _run_cli() -> None:
        extractor = ClaimExtractorAnalyzer()
        try:
            # 1. Se passou o texto como argumento direto via linha de comando:
            if len(sys.argv) > 1:
                text = " ".join(sys.argv[1:])
                print("\n🔍 Processando texto nas 3 camadas (Contrato, NER & Decomposição)...")
                res = await extractor.analyze(text, [])
                format_cli_result(res)
                return

            # 2. Modo Interativo Contínuo
            print("=" * 75)
            print("🔎 FactChkBR - Extrator Factual (spaCy Gatekeeper + LLM Structured Outputs)")
            print("Cole qualquer mensagem ou notícia abaixo e pressione ENTER.")
            print("Digite 'sair' ou pressione Ctrl+C para encerrar.")
            print("=" * 75)

            while True:
                try:
                    print("📥 Cole o texto a ser analisado:")
                    first_line = input("> ").strip()
                    if not first_line:
                        continue
                    if first_line.lower() in ("sair", "exit", "quit", "q"):
                        print("Encerrando testador.")
                        break

                    lines = [first_line]
                    try:
                        while select.select([sys.stdin], [], [], 0.05)[0]:
                            extra = sys.stdin.readline()
                            if not extra:
                                break
                            lines.append(extra.strip())
                    except Exception:
                        pass

                    full_text = " ".join(line for line in lines if line)
                    res = await extractor.analyze(full_text, [])
                    format_cli_result(res)

                except (KeyboardInterrupt, EOFError):
                    print("\nSessão encerrada.")
                    break
        finally:
            await extractor.aclose()

    asyncio.run(_run_cli())