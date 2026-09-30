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
    ClaimExtractionContract,
    KnowledgeTriple,
    VerificationSourceType,
)

logger = logging.getLogger("factchkbr.analyzers.claim_extractor")

# ==============================================================================
# LIMPEZA & PADRÕES DE ALARME / RUÍDO (CPU)
# ==============================================================================

RE_ALARM_SYMBOLS = re.compile(r"[🚨⚠️💣🔥🛑📢👀⚡🇧🇷❌‼️⁉️]+")
RE_PUNCT_COLLAPSE = re.compile(r"([!?.]){2,}")
RE_WHITESPACE = re.compile(r"\s+")
RE_ALARM_HEADERS = re.compile(
    r"^(?:(?:URGENTE|BOMBA|ALERTA|ATENÇÃO BRASIL|VEJA|OLHE|COMPARTILHEM?|REPASSEM?)[!.:\s-]*)+",
    re.IGNORECASE,
)

OPINION_PATTERNS = re.compile(
    r"\b(?:eu acho|eu acredito|minha opinião|penso que|na minha visão|vergonha|absurdo|"
    r"maravilhosa|abençoe|deus abençoe|que deus|lindo demais|horroroso|incompetente e antipático)\b",
    re.IGNORECASE,
)

# Siglas de órgãos e entidades institucionais relevantes para Fact-Checking
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
    """Normaliza strings ou alucinações de LLM para o enum estrito VerificationSourceType."""
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


def infer_source_types(text: str, entities: dict[str, list[str]]) -> list[VerificationSourceType]:
    """Infere tipos de fontes oficiais recomendadas a partir do texto e entidades detectadas."""
    sources: list[VerificationSourceType] = []
    combined = (text + " " + " ".join(e for cat in entities.values() for e in cat)).lower()

    if any(k in combined for k in ("anvisa", "anatel", "bacen", "aneel", "ans", "antt", "cvm", "ibama")):
        sources.append(VerificationSourceType.AGENCIA_REGULADORA)
    if any(k in combined for k in ("ministério", "governo", "receita federal", "diário oficial", "prefeitura", "presidência")):
        sources.append(VerificationSourceType.ORGAO_OFICIAL)
    if any(k in combined for k in ("stf", "stj", "tse", "tcu", "cnj", "justiça", "tribunal", "moraes", "juiz", "vara", "moro")):
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


# ==============================================================================
# 1. PRÉ-PROCESSAMENTO & GATEKEEPER VIA SPACY (CPU)
# ==============================================================================

class SpacyPreprocessor:
    """Higienização de ruídos, extração de entidades e gatekeeping sintático com spaCy."""

    def __init__(self) -> None:
        for model in ("pt_core_news_sm", "pt_core_news_md", "pt_core_news_lg"):
            try:
                self.nlp = spacy.load(model, exclude=["lemmatizer"])
                logger.info("spaCy carregado com sucesso: %s", model)
                break
            except Exception:
                continue
        else:
            raise RuntimeError(
                "Nenhum modelo do spaCy foi encontrado. "
                "Execute: python -m spacy download pt_core_news_sm"
            )

    def clean_text(self, text: str) -> str:
        """Remove emojis de alarme, pontuações de pânico e decanta alertas sensacionalistas."""
        t = RE_ALARM_SYMBOLS.sub(" ", text)
        t = re.sub(r"\?{2,}", "?", t)
        t = re.sub(r"!{2,}", ".", t)
        t = re.sub(r"\.{2,}", ".", t)
        t = RE_WHITESPACE.sub(" ", t).strip()

        # Remove alertas de topo repetidos (ex: BOMBA!! URGENTE: ...)
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
            # Calibração taxonômica de siglas institucionais e cargos
            if text_clean.upper() in KNOWN_ORGS:
                lbl = "ORG"
            elif text_clean.lower() in KNOWN_ROLES:
                lbl = "PER"

            if lbl in entities and text_clean not in entities[lbl]:
                entities[lbl].append(text_clean)

        # Identifica papéis, profissões e órgãos que o modelo de NER pequeno possa não ter marcado
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
          para formular uma alegação factual (verbo + substantivo/entidade).
        - Descarta na raiz ruídos, saudações e opiniões declaradas.
        """
        entities = self.extract_entities(text)
        if not text or len(text.strip()) < 8:
            return entities, False

        # Descarte antecipado de opiniões declaradas ou cumprimentos
        if OPINION_PATTERNS.search(text) and any(w in text.lower() for w in ("acho", "deus", "vergonha", "opinião", "bom dia")):
            return entities, False

        doc = self.nlp(text)
        has_verb = any(t.pos_ in ("VERB", "AUX") for t in doc) or any(t.dep_ == "nsubj" for t in doc)
        has_content = any(t.pos_ in ("NOUN", "PROPN", "NUM") for t in doc) or any(entities.values())
        has_min_length = len(doc) >= 4

        is_viable = has_verb and has_content and has_min_length
        return entities, is_viable


# Alias para retrocompatibilidade com suítes de teste
SpacyNERCleaner = SpacyPreprocessor


# ==============================================================================
# 2. MOTOR DE DECOMPOSIÇÃO EXCLUSIVAMENTE VIA LLM (COM CACHE LRU)
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
    """Decomposição semântica e estruturação exclusivamente via LLM com cache LRU em memória."""

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

    async def decompose(
        self,
        cleaned_text: str,
        entities: dict[str, list[str]],
    ) -> list[AtomicAssertion]:
        """
        Decompõe texto em proposições atômicas exclusivamente via LLM.
        Não utiliza fallback determinístico.
        """
        cache_key = hashlib.sha256(cleaned_text.encode("utf-8")).hexdigest()
        cached = self._get_from_cache(cache_key)
        if cached is not None:
            return cached

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

        resp = await self.http_client.post(
            self.settings.get_llm_endpoint(),
            headers=self.settings.get_llm_headers(),
            json=payload,
        )

        # Se json_schema não for aceito pelo provedor/modelo, tenta modo json_object
        if resp.status_code == 400:
            payload["response_format"] = {"type": "json_object"}
            resp = await self.http_client.post(
                self.settings.get_llm_endpoint(),
                headers=self.settings.get_llm_headers(),
                json=payload,
            )

        resp.raise_for_status()

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

        self._save_to_cache(cache_key, assertions)
        return assertions


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
        entities, is_viable = self.preprocessor.analyze(cleaned)

        # Camada 1 (Gatekeeper spaCy): Se não houver estrutura mínima factual, descarta antecipadamente
        if not is_viable:
            return ClaimExtractionContract(
                original_text=text,
                cleaned_text=cleaned,
                entities=entities,
                assertions=[],
                discarded_fragments=[cleaned] if cleaned else [],
                engine_used="spacy_gatekeeper_short_circuit",
            )

        # Camada 2: Decomposição Atômica via LLM com recuperação resiliente
        try:
            assertions = await self.decomposer.decompose(cleaned, entities)
            engine = f"llm:{self.settings.get_llm_model()}"
        except Exception as e:
            logger.warning("Falha ou timeout na decomposição via LLM (%s): %s. Preservando asserção direta.", self.settings.get_llm_model(), e)
            assertions = [
                AtomicAssertion(
                    id=1,
                    statement=cleaned,
                    triple=KnowledgeTriple(
                        subject=entities.get("ORG", [""])[0] if entities.get("ORG") else (entities.get("PER", [""])[0] if entities.get("PER") else "Brasil"),
                        predicate="afirma/relata",
                        object=cleaned[:120],
                    ),
                    is_check_worthy=True,
                    suggested_source_types=infer_source_types(cleaned, entities),
                )
            ]
            engine = f"llm_contingency:{type(e).__name__}"

        return ClaimExtractionContract(
            original_text=text,
            cleaned_text=cleaned,
            entities=entities,
            assertions=assertions,
            discarded_fragments=[],
            engine_used=engine,
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