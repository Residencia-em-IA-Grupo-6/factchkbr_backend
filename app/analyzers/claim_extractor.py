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
import spacy

from app.config import get_settings
from app.core.base import BaseAnalyzer
from app.core.registry import register_analyzer
from app.schemas.analysis import AnalyzerResult
from app.schemas.claim_extraction import (
    AtomicAssertion,
    ClaimCategory,
    ClaimExtractionContract,
    KnowledgeTriple,
    VerificationSourceType,
)

logger = logging.getLogger("factchkbr.analyzers.claim_extractor")

# ==============================================================================
# LIMPEZA & PADRÕES ESTRUTURAIS (CPU)
# ==============================================================================

RE_ALARM_SYMBOLS = re.compile(r"[🚨⚠️💣🔥🛑📢👀⚡🇧🇷❌‼️⁉️]+")
RE_PUNCT_COLLAPSE = re.compile(r"([!?.]){2,}")
RE_WHITESPACE = re.compile(r"\s+")
RE_ALARM_HEADERS = re.compile(
    r"^(?:(?:URGENTE|BOMBA|ALERTA|ATENÇÃO BRASIL|COMPARTILHEM?|REPASSEM?)[!.:\s-]*)+",
    re.IGNORECASE,
)

# Padrão para detecção de vazamento de tradução para inglês por LLMs menores
ENGLISH_WORDS_PATTERN = re.compile(
    r"\b(?:the|is|are|was|were|can|cannot|could|should|would|will|and|or|of|in|on|at|by|with|from|to|be|been|being|voting|used|claim)\b",
    re.IGNORECASE,
)


# Siglas de órgãos e entidades institucionais para enriquecer reconhecimento de entidades
KNOWN_ORGS = {
    "STF", "TSE", "STJ", "TCU", "ANVISA", "PF", "PRF", "IBGE", "MEC", "SUS",
    "FIOCRUZ", "PETROBRAS", "RECEITA FEDERAL", "GOVERNO FEDERAL", "INSS", "ANATEL",
    "ANS", "ANEEL", "ANTT", "CVM", "IBAMA"
}

KNOWN_ROLES = {
    "médico", "médicos", "doutor", "doutores", "cirurgião", "cirurgiões",
    "cientista", "cientistas", "especialista", "especialistas", "pesquisador", "pesquisadores"
}


def map_source_type(val: Any) -> VerificationSourceType | None:
    """Normaliza strings ou retornos de LLM para o enum estrito VerificationSourceType."""
    if isinstance(val, VerificationSourceType):
        return val
    if not isinstance(val, str):
        return None
    normalized = val.strip().upper().replace(" ", "_").replace("-", "_")
    for st in VerificationSourceType:
        if st.value == normalized:
            return st
    return None


def map_claim_category(val: Any) -> ClaimCategory:
    """Normaliza strings ou retornos de LLM para o enum ClaimCategory."""
    if isinstance(val, ClaimCategory):
        return val
    if isinstance(val, str):
        norm = val.strip().upper().replace(" ", "_").replace("-", "_")
        for cat in ClaimCategory:
            if cat.value == norm:
                return cat
    return ClaimCategory.FACTUAL_CLAIM


def infer_source_types(text: str = "", entities: dict[str, list[str]] | None = None) -> list[VerificationSourceType]:
    """Helper de retrocompatibilidade para inferência de fontes padrão."""
    return [VerificationSourceType.DADOS_PUBLICOS, VerificationSourceType.AGENCIA_CHECAGEM]


# ==============================================================================
# 1. PRÉ-PROCESSAMENTO & GATEKEEPER VIA SPACY (CPU)
# ==============================================================================

HEALTH_FACTUAL_VERBS = {
    "emagrece", "emagrecem", "emagreça", "emagrecer",
    "engorda", "engordam", "engordar",
    "cura", "curam", "curar", "cure",
    "mata", "matam", "matar",
    "salva", "salvam", "salvar",
    "alivia", "aliviam", "aliviar",
    "previne", "previnem", "prevenir",
    "protege", "protegem", "proteger",
    "causa", "causam", "causar",
    "provoca", "provocam", "provocar",
    "combate", "combatem", "combater",
    "funciona", "funcionam", "funcionar",
    "melhora", "melhoram", "melhorar",
    "piora", "pioram", "piorar",
    "aumenta", "aumentam", "aumentar",
    "diminui", "diminuem", "diminuir",
    "reduz", "reduzem", "reduzir",
    "trata", "tratam", "tratar",
    "limpa", "limpam", "limpar",
    "desintoxica", "desintoxicam", "desintoxicar",
    "seca", "secam", "secar",
}


def is_verb_like_token(token: Any) -> bool:
    """Detecta predicados verbais mesmo quando o spaCy pt_core_news_sm comete erros de etiquetagem (ex: 'emagrece' como ADJ)."""
    if token.pos_ in ("VERB", "AUX"):
        return True
    lower = token.text.lower()
    if lower in HEALTH_FACTUAL_VERBS:
        return True
    if lower.endswith(("ecer", "ece", "ecem", "eceu", "ecerá", "ecendo")):
        return True
    return False


class SpacyPreprocessor:
    """Higienização de ruídos estruturais, extração de entidades e gatekeeping sintático."""
    _nlp: Any = None

    def __init__(self) -> None:
        if SpacyPreprocessor._nlp is None:
            for model in ("pt_core_news_sm", "pt_core_news_md", "pt_core_news_lg"):
                try:
                    SpacyPreprocessor._nlp = spacy.load(model, exclude=["lemmatizer"])
                    logger.info("spaCy carregado com sucesso (singleton): %s", model)
                    break
                except Exception:
                    continue
            else:
                raise RuntimeError(
                    "Nenhum modelo do spaCy foi encontrado. "
                    "Execute: python -m spacy download pt_core_news_sm"
                )
        self.nlp = SpacyPreprocessor._nlp

    def extract_syntactic_triple(self, text: str) -> dict[str, str]:
        """Extrai aproximação sintática (sujeito, predicado, objeto) via spaCy para fallback estruturado."""
        doc = self.nlp(text)
        subj = ""
        pred = ""
        obj = ""

        # Tenta localizar verbo principal (inclusive com detecção robusta de verbos biológicos/saúde)
        pred_idx = -1
        for i, t in enumerate(doc):
            if is_verb_like_token(t):
                pred = t.text
                pred_idx = i
                break

        if pred_idx != -1:
            subj_tokens = [doc[i].text for i in range(pred_idx)]
            obj_tokens = [doc[i].text for i in range(pred_idx + 1, len(doc))]
            subj = " ".join(subj_tokens).strip()
            obj = " ".join(obj_tokens).strip()

        if not pred:
            for t in doc:
                if "subj" in t.dep_ and not subj:
                    subj = " ".join(w.text for w in t.subtree).strip()
                elif t.pos_ in ("VERB", "AUX") and not pred:
                    pred = t.text
                elif ("obj" in t.dep_ or "obl" in t.dep_) and not obj:
                    obj = " ".join(w.text for w in t.subtree).strip()

        return {
            "subject": subj or "Sujeito",
            "predicate": pred or "declara",
            "object": obj or text.strip(),
        }

    def clean_text(self, text: str) -> str:
        """Remove emojis de alarme, normaliza pontuações repetidas e limpa alertas de manchete."""
        t = RE_ALARM_SYMBOLS.sub(" ", text)
        t = re.sub(r"\?{2,}", "?", t)
        t = re.sub(r"!{2,}", ".", t)
        t = re.sub(r"\.{2,}", ".", t)
        t = RE_WHITESPACE.sub(" ", t).strip()

        # Remove prefixos puramente sensacionalistas de manchete (ex: BOMBA!! URGENTE: ...)
        t = RE_ALARM_HEADERS.sub("", t).strip()
        t = re.sub(r"^[\s,;.:-]+", "", t).strip()
        return t

    def extract_entities(self, text: str) -> dict[str, list[str]]:
        """Extrai entidades nomeadas categorizadas (PER, ORG, LOC) via spaCy."""
        entities: dict[str, list[str]] = {"PER": [], "ORG": [], "LOC": []}
        if not text:
            return entities

        doc = self.nlp(text)
        for ent in doc.ents:
            lbl = ent.label_
            text_clean = ent.text.strip()
            if text_clean.upper() in KNOWN_ORGS:
                lbl = "ORG"
            elif text_clean.lower() in KNOWN_ROLES:
                lbl = "PER"

            if lbl in entities and text_clean not in entities[lbl]:
                entities[lbl].append(text_clean)

        for token in doc:
            tok_text = token.text.strip()
            if tok_text.lower() in KNOWN_ROLES and tok_text not in entities["PER"]:
                entities["PER"].append(tok_text)
            elif tok_text.upper() in KNOWN_ORGS and tok_text not in entities["ORG"]:
                entities["ORG"].append(tok_text)

        return entities

    def analyze(self, text: str) -> tuple[dict[str, list[str]], bool]:
        """
        Processa o texto em único passe (CPU com spaCy):
        - Extrai entidades (PER, ORG, LOC).
        - Atua como gatekeeper: retorna True se houver viabilidade sintática mínima
          para formular uma oração (presença de verbo/predicado + termo nominal/sujeito).
        """
        entities = self.extract_entities(text)
        if not text or len(text.strip()) < 6:
            return entities, False

        # Descarta saudações, bênçãos ou opiniões puras sem dados empíricos
        opinion_patterns = re.compile(
            r"^(?:bom dia|boa tarde|boa noite|olá|ola|que deus|deus te|deus abençoe|eu acho|na minha opinião|eu acredito|penso que|me parece|que absurdo|que vergonha)\b",
            re.IGNORECASE,
        )
        if opinion_patterns.search(text.strip()):
            return entities, False

        doc = self.nlp(text)
        has_verb = any(is_verb_like_token(t) for t in doc) or any(t.dep_ in ("nsubj", "nsubj:pass") for t in doc)
        has_content = any(t.pos_ in ("NOUN", "PROPN", "NUM") for t in doc) or any(entities.values())
        has_min_length = len(doc) >= 2

        is_viable = has_verb and has_content and has_min_length
        return entities, is_viable


# Alias para retrocompatibilidade com suítes de teste
SpacyNERCleaner = SpacyPreprocessor


# ==============================================================================
# 2. MOTOR DE DECOMPOSIÇÃO SEMÂNTICA EXCLUSIVAMENTE VIA LLM
# ==============================================================================

SYSTEM_PROMPT = """Você é um especialista em Fact-Checking e Extração Semântica de Alegações (Claim Extraction & Check-Worthiness).
/no_think
Sua responsabilidade é analisar o texto recebido de redes sociais ou fontes públicas e estruturar suas proposições atômicas, identificando com precisão a alegação central a ser checada.

### REQUISITOS FUNDAMENTAIS:
- IDIOMA ESTRITAMENTE EM PORTUGUÊS (pt-BR): Todos os campos ('primary_claim', 'statement', 'triple.subject', 'triple.predicate', 'triple.object') DEVEM ser mantidos rigorosamente em PORTUGUÊS (pt-BR). É expressamente PROIBIDO traduzir termos para o inglês ou qualquer outro idioma.
- FIDELIDADE LEXICAL: NÃO invente palavras, neologismos ou sufixos inexistentes (ex: NUNCA altere "emagrece" para "emagessa" ou "limonete"). Preserve exatamente o vocabulário e a grafia das palavras do texto original em português.
- PRESERVAÇÃO DENOTATIVA: Mantenha a alegação central expressa em português de forma clara, denotativa e fiel ao sentido pretendido.
- COMPLETUDE DAS ASSERÇÕES: Cada asserção ('statement') DEVE ser uma oração completa e inteligível (ex: 'É falso que o voto não pode ser usado no INSS'). NUNCA retorne fragmentos incompletos como apenas 'É falso' ou 'Não procede'.
- ESTRUTURA DO JSON DE RESPOSTA:
Responda OBRIGATORIAMENTE em JSON válido contendo o array 'assertions':
{
  "primary_claim": "Texto conciso da alegação central a ser checada",
  "assertions": [
    {
      "id": 1,
      "statement": "Oração completa da proposição factual",
      "category": "FACTUAL_CLAIM",
      "triple": {
        "subject": "sujeito",
        "predicate": "predicado verbal",
        "object": "objeto ou complemento"
      },
      "suggested_source_types": ["DADOS_PUBLICOS", "AGENCIA_CHECAGEM"],
      "is_check_worthy": true
    }
  ]
}
- RESPOSTA DIRETA EM JSON: Não adicione blocos de reflexão ou monólogo interno. Responda única e exclusivamente o objeto JSON solicitado."""

DECOMPOSITION_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "primary_claim": {
            "type": ["string", "null"],
            "description": "Alegação factual central substantiva a ser checada, ou null se não houver fato falseável.",
        },
        "assertions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "statement": {"type": "string"},
                    "category": {
                        "type": "string",
                        "enum": [
                            "FACTUAL_CLAIM",
                            "ATTRIBUTION",
                            "CONVERSATIONAL_NOISE",
                            "OPINION",
                        ],
                    },
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
                "required": ["id", "statement", "category", "triple", "suggested_source_types", "is_check_worthy"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["primary_claim", "assertions"],
    "additionalProperties": False,
}


class LLMClaimDecomposer:
    """Decomposição semântica e estruturação via LLM com cache LRU em memória compartilhada."""
    _shared_cache: OrderedDict[str, tuple[str | None, list[AtomicAssertion]]] = OrderedDict()

    def __init__(self, http_client: httpx.AsyncClient | None = None, cache_maxsize: int = 1024) -> None:
        self.settings = get_settings()
        self._owned_client = http_client is None
        self.http_client = http_client or httpx.AsyncClient(
            timeout=65.0 if self.settings.LLM_PROVIDER.lower() == "ollama" else 15.0,
            limits=httpx.Limits(max_keepalive_connections=25, max_connections=50),
        )
        self._cache_maxsize = cache_maxsize

    def _get_from_cache(self, key: str) -> tuple[str | None, list[AtomicAssertion]] | None:
        if key in self._shared_cache:
            self._shared_cache.move_to_end(key)
            return self._shared_cache[key]
        return None

    def _save_to_cache(self, key: str, value: tuple[str | None, list[AtomicAssertion]]) -> None:
        self._shared_cache[key] = value
        if len(self._shared_cache) > self._cache_maxsize:
            self._shared_cache.popitem(last=False)

    async def aclose(self) -> None:
        """Encerra cliente HTTP se instanciado internamente."""
        if self._owned_client:
            await self.http_client.aclose()

    async def decompose(
        self,
        cleaned_text: str,
        entities: dict[str, list[str]],
    ) -> tuple[str | None, list[AtomicAssertion]]:
        """
        Decompõe texto em proposições atômicas e identifica alegação central via LLM.
        """
        cache_key = hashlib.sha256(cleaned_text.encode("utf-8")).hexdigest()
        cached = self._get_from_cache(cache_key)
        if cached is not None:
            return cached

        user_prompt = (
            f"Texto: \"{cleaned_text}\"\n"
            f"Entidades pré-detectadas: {json.dumps(entities, ensure_ascii=False)}\n\n"
            "Analise o texto acima, classifique as proposições atômicas e identifique a alegação central substantiva a ser checada. "
            "Responda OBRIGATORIAMENTE em português (pt-BR)."
        )

        is_ollama = self.settings.LLM_PROVIDER.lower() == "ollama"
        if is_ollama:
            payload = {
                "model": self.settings.get_llm_model(),
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                "stream": False,
                "think": False,
                "format": "json",
                "options": {
                    "temperature": 0.0,
                    "num_predict": 512,
                },
            }
        else:
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

        try:
            resp = await self.http_client.post(
                self.settings.get_llm_endpoint(),
                headers=self.settings.get_llm_headers(),
                json=payload,
            )

            # Se json_schema não for aceito pelo provedor/modelo (OpenAI compat), tenta modo json_object
            if not is_ollama and resp.status_code == 400:
                payload["response_format"] = {"type": "json_object"}
                resp = await self.http_client.post(
                    self.settings.get_llm_endpoint(),
                    headers=self.settings.get_llm_headers(),
                    json=payload,
                )

            resp.raise_for_status()

            data = resp.json()
            if "message" in data:
                content = data["message"].get("content", "")
            elif "choices" in data and data["choices"]:
                content = data["choices"][0].get("message", {}).get("content", "")
            else:
                content = ""

            # Sanitização: remove blocos residuais de pensamento <think> e fences markdown ```json ... ```
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
            if "```" in content:
                fence_m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", content)
                if fence_m:
                    content = fence_m.group(1).strip()
            start = content.find("{")
            end = content.rfind("}")
            if start != -1 and end != -1 and end > start:
                content = content[start : end + 1]

            parsed = json.loads(content)
            primary_claim = parsed.get("primary_claim")
            raw_assertions = parsed.get("assertions", [])

            # Auto-healing: se o modelo responder em formato plano (sem o array 'assertions')
            if not raw_assertions and (parsed.get("statement") or primary_claim):
                flat_stmt = parsed.get("statement") or primary_claim or cleaned_text
                flat_sub = parsed.get("triple.subject") or parsed.get("subject") or ""
                flat_pred = parsed.get("triple.predicate") or parsed.get("predicate") or ""
                flat_obj = parsed.get("triple.object") or parsed.get("object") or ""
                raw_assertions = [{
                    "id": 1,
                    "statement": str(flat_stmt),
                    "category": "FACTUAL_CLAIM",
                    "triple": {
                        "subject": str(flat_sub or ""),
                        "predicate": str(flat_pred or ""),
                        "object": str(flat_obj or ""),
                    },
                    "suggested_source_types": ["DADOS_PUBLICOS", "AGENCIA_CHECAGEM"],
                    "is_check_worthy": True,
                }]

            # Salvaguarda lexical para frases concisas (evita que o modelo invente radicais como "emagessa")
            if primary_claim and len(cleaned_text.split()) <= 8:
                orig_words = set(re.findall(r"\b[A-Za-zÀ-ÿ]{3,}\b", cleaned_text.lower()))
                claim_words = set(re.findall(r"\b[A-Za-zÀ-ÿ]{3,}\b", primary_claim.lower()))
                diff = claim_words - orig_words
                for w in diff:
                    if w in ("emagessa", "limonete") or (len(diff) > 2 and len(orig_words) <= 3):
                        logger.warning(
                            "Distorção léxica detectada no primary_claim ('%s' vs '%s'). Restaurando texto original.",
                            primary_claim,
                            cleaned_text,
                        )
                        primary_claim = cleaned_text
                        break

            # Salvaguarda contra contaminação linguística (ex: modelo menor traduzindo para inglês)
            cleaned_has_en = bool(ENGLISH_WORDS_PATTERN.search(cleaned_text))
            if primary_claim and isinstance(primary_claim, str) and not cleaned_has_en:
                if bool(ENGLISH_WORDS_PATTERN.search(primary_claim)):
                    logger.warning(
                        "Tradução indevida para inglês detectada no primary_claim ('%s'). Restaurando texto original em português.",
                        primary_claim,
                    )
                    primary_claim = cleaned_text

            assertions: list[AtomicAssertion] = []
            for idx, item in enumerate(raw_assertions, 1):
                if isinstance(item, str):
                    item = {
                        "id": idx,
                        "statement": item,
                        "category": "FACTUAL_CLAIM",
                        "triple": {"subject": "", "predicate": "", "object": ""},
                        "suggested_source_types": ["DADOS_PUBLICOS", "AGENCIA_CHECAGEM"],
                        "is_check_worthy": True,
                    }
                elif not isinstance(item, dict):
                    continue

                # Normaliza categoria taxonômica
                item["category"] = map_claim_category(item.get("category"))

                # Salvaguarda linguística e de completude em statement
                stmt = item.get("statement", "")
                if stmt and isinstance(stmt, str):
                    if not cleaned_has_en and bool(ENGLISH_WORDS_PATTERN.search(stmt)):
                        stmt = cleaned_text
                    if len(stmt.split()) < 3 and (primary_claim or cleaned_text):
                        stmt = primary_claim or cleaned_text
                    item["statement"] = stmt

                # Normaliza fontes
                cleaned_sources: list[str] = []
                for s in item.get("suggested_source_types", []):
                    mapped = map_source_type(s)
                    if mapped and mapped.value not in cleaned_sources:
                        cleaned_sources.append(mapped.value)
                if not cleaned_sources:
                    cleaned_sources = [
                        VerificationSourceType.DADOS_PUBLICOS.value,
                        VerificationSourceType.AGENCIA_CHECAGEM.value,
                    ]
                item["suggested_source_types"] = cleaned_sources
                assertions.append(AtomicAssertion.model_validate(item))

            result = (primary_claim, assertions)
            self._save_to_cache(cache_key, result)
            return result

        except Exception as exc:
            logger.warning("Falha na decomposição via LLM (%s): %s", type(exc).__name__, exc)
            return None, []


def select_primary_assertion(
    assertions: list[AtomicAssertion],
    suggested_primary: str | None = None,
) -> AtomicAssertion | None:
    """
    Seleciona a asserção central para checagem baseando-se estritamente na taxonomia
    semântica e na relevância para verificação (zero listas de vocabulário hard-coded).
    """
    if not assertions:
        return None

    # 1. Se o modelo sugeriu um claim primário, localiza a asserção que melhor corresponde
    if suggested_primary:
        norm_target = suggested_primary.strip().lower()
        for a in assertions:
            stmt = a.statement.strip().lower()
            if stmt == norm_target or norm_target in stmt or stmt in norm_target:
                return a
        # Se não houver correspondência exata, busca uma FACTUAL_CLAIM com checabilidade
        factuals = [a for a in assertions if a.category == ClaimCategory.FACTUAL_CLAIM and a.is_check_worthy]
        if factuals:
            return factuals[0]

    # 2. Prioridade: asserção classificada como fato empírico substantivo (FACTUAL_CLAIM)
    for a in assertions:
        if a.category == ClaimCategory.FACTUAL_CLAIM and a.is_check_worthy:
            return a

    # 3. Segunda opção: asserção de atribuição checável (ex: declaração com mérito)
    for a in assertions:
        if a.category == ClaimCategory.ATTRIBUTION and a.is_check_worthy:
            return a

    # 4. Qualquer asserção restante marcada como passível de verificação
    for a in assertions:
        if a.is_check_worthy:
            return a

    return None


# ==============================================================================
# 3. ANALISADOR REGISTRADO (BaseAnalyzer)
# ==============================================================================

@register_analyzer("claim_extractor", weight=0.0)
class ClaimExtractorAnalyzer(BaseAnalyzer):
    """
    Analisador de Extração Factual em 2 Etapas:
    1. Higienização & Gatekeeper (spaCy em CPU - descarta não-fatos a custo zero).
    2. Decomposição Semântica Atômica (LLM com Structured Outputs e Cache LRU).
    """

    def __init__(self) -> None:
        super().__init__()
        self.settings = get_settings()
        self.preprocessor = SpacyPreprocessor()
        self.http_client = httpx.AsyncClient(
            timeout=65.0 if self.settings.LLM_PROVIDER.lower() == "ollama" else 15.0,
            limits=httpx.Limits(max_keepalive_connections=25, max_connections=50),
        )
        self.decomposer = LLMClaimDecomposer(self.http_client)

    async def aclose(self) -> None:
        """Encerra o pool de conexões HTTP ao finalizar a aplicação."""
        await self.http_client.aclose()

    async def extract_contract(self, text: str) -> ClaimExtractionContract:
        """Executa o pipeline (spaCy -> LLM) e produz o contrato estrito Pydantic."""
        cleaned = self.preprocessor.clean_text(text)
        entities, is_viable = await asyncio.to_thread(self.preprocessor.analyze, cleaned)

        # Camada 1 (Gatekeeper spaCy): Se não houver estrutura mínima factual, descarta antecipadamente
        if not is_viable:
            return ClaimExtractionContract(
                original_text=text,
                cleaned_text=cleaned,
                primary_claim=None,
                entities=entities,
                assertions=[],
                discarded_fragments=[cleaned] if cleaned else [],
                engine_used="spacy_gatekeeper_short_circuit",
            )

        # Camada 2: Decomposição Semântica e Atômica via LLM
        primary_claim, assertions = await self.decomposer.decompose(cleaned, entities)
        engine_used = f"llm:{self.settings.get_llm_model()}"

        # Fallback de resiliência: se o LLM falhou/está indisponível mas a oração é viável pelo spaCy
        if not assertions and is_viable:
            logger.info("LLM indisponível para decomposição. Ativando fallback sintático estruturado do spaCy.")
            triple_dict = await asyncio.to_thread(self.preprocessor.extract_syntactic_triple, cleaned)
            fallback_assertion = AtomicAssertion(
                id=1,
                statement=cleaned,
                category=ClaimCategory.FACTUAL_CLAIM,
                triple=KnowledgeTriple(
                    subject=triple_dict.get("subject", "Sujeito"),
                    predicate=triple_dict.get("predicate", "declara"),
                    object=triple_dict.get("object", "objeto"),
                ),
                suggested_source_types=[
                    VerificationSourceType.DADOS_PUBLICOS.value,
                    VerificationSourceType.AGENCIA_CHECAGEM.value,
                ],
                is_check_worthy=True,
            )
            assertions = [fallback_assertion]
            primary_claim = cleaned
            engine_used = "spacy_syntactic_fallback"

        # Seleciona asserção principal usando a taxonomia semântica (sem hardcoded)
        primary_assertion = select_primary_assertion(assertions, primary_claim)
        final_primary_claim = primary_claim or (primary_assertion.statement if primary_assertion else None)

        # Fragmentos descartados por serem ruído conversacional ou opinião
        discarded = [
            a.statement for a in assertions
            if a.category in (ClaimCategory.CONVERSATIONAL_NOISE, ClaimCategory.OPINION)
        ]

        return ClaimExtractionContract(
            original_text=text,
            cleaned_text=cleaned,
            primary_claim=final_primary_claim,
            entities=entities,
            assertions=assertions,
            discarded_fragments=discarded,
            engine_used=engine_used,
        )

    async def analyze(self, text: str, urls: list[str]) -> AnalyzerResult:
        """Interface padrão do BaseAnalyzer consumida pelo FactCheckOrchestrator."""
        contract = await self.extract_contract(text)
        primary_claim = contract.primary_claim

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
        print(f"⚙️  MOTOR EXECUTOR: {engine}")

    if result.claim:
        print(f"🎯 ALEGAÇÃO PRINCIPAL ISOLADA (CLAIM):\n   👉 \"{result.claim}\"\n")
    else:
        print("⚠️  NENHUMA ALEGAÇÃO FACTUAL IDENTIFICADA (Texto opinativo/saudação/ruído)\n")

    if assertions:
        print(f"🧬 PROPOSIÇÕES ATÔMICAS & TRIPLAS (SPO) [{len(assertions)}]:")
        for a in assertions:
            idx = a.get("id", "-")
            stmt = a.get("statement", "")
            cat = a.get("category", "")
            triple = a.get("triple", {})
            s = triple.get("subject", "?")
            p = triple.get("predicate", "?")
            o = triple.get("object", "?")
            sources = a.get("suggested_source_types", [])
            print(f"   [{idx}] Statement: \"{stmt}\"")
            print(f"       ├─ Categoria: {cat}")
            print(f"       ├─ Tripla:    ({s} ➔ {p} ➔ {o})")
            print(f"       └─ Fontes:    {sources}")
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
                print("\n🔍 Processando texto nas 2 camadas (spaCy Gatekeeper + LLM Structured Outputs)...")
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