import asyncio
import json
import logging
import re
import sys
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
# MOTOR MORFOLÓGICO E SINTÁTICO ADAPTATIVO (PORTUGUÊS)
# ==============================================================================

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

# Sufixos de Pretérito Perfeito (-ou, -aram, -eu, -eram, -iu, -iram)
PAST_INDICATIVE_SUFFIXES = re.compile(
    r"\b[a-zA-ZáéíóúâêîôûãõçÁÉÍÓÚÂÊÎÔÛÃÕÇ]{3,}(?:ou|aram|eu|eram|iu|iram)\b",
    re.IGNORECASE,
)

# Sufixos de Pretérito Imperfeito (-ava, -avam, -ia, -iam)
IMPERFECT_INDICATIVE_SUFFIXES = re.compile(
    r"\b[a-zA-ZáéíóúâêîôûãõçÁÉÍÓÚÂÊÎÔÛÃÕÇ]{3,}(?:ava|avam|ia|iam)\b",
    re.IGNORECASE,
)

# Sufixos de Futuro do Presente (-ará, -arão, -erá, -erão, -irá, -irão)
FUTURE_INDICATIVE_SUFFIXES = re.compile(
    r"\b[a-zA-ZáéíóúâêîôûãõçÁÉÍÓÚÂÊÎÔÛÃÕÇ]{3,}(?:ará|arão|erá|erão|irá|irão)\b",
    re.IGNORECASE,
)

# Sufixos de Condicional (-aria, -ariam)
CONDITIONAL_INDICATIVE_SUFFIXES = re.compile(
    r"\b[a-zA-ZáéíóúâêîôûãõçÁÉÍÓÚÂÊÎÔÛÃÕÇ]{3,}(?:aria|ariam|eria|eriam|iria|iriam)\b",
    re.IGNORECASE,
)

# Verbos Irregulares de Alta Frequência
PAST_IRREGULAR_VERBS = re.compile(
    r"\b(?:disse|disseram|teve|tiveram|esteve|estiveram|fez|fizeram|deu|deram|"
    r"pôs|puseram|trouxe|trouxeram|veio|vieram|viu|viram|ouviu|ouviram|"
    r"quis|quiseram|pôde|puderam|soube|souberam|houve)\b",
    re.IGNORECASE,
)

# Verbos de Ação / Causalidade / Fatos no Presente do Indicativo
PRESENT_DECLARATIVE_VERBS = re.compile(
    r"\b(?:"
    # Ações biológicas, médicas, cirúrgicas, consumo e saúde
    r"extrai|extraem|come|comem|consome|consomem|bebe|bebem|ingere|ingerem|"
    r"retira|retiram|remove|removem|opera|operam|injeta|injetam|aplica|aplicam|"
    r"toma|tomam|usa|usam|engole|engolem|vomita|vomitam|infecta|infectam|"
    r"contamina|contaminam|transmite|transmitem|atinge|atingem|afeta|afetam|"
    r"cura|curam|mata|matam|morre|morrem|falece|falecem|sofre|sofrem|"
    r"encontra|encontram|descobre|descobrem|mostra|mostram|grava|gravam|"
    r"filma|filmam|flagra|flagram|"
    # Verbos terminados em -air (3ª pessoa singular e plural)
    r"contrai|contraem|atrai|atraem|distrai|distraem|subtrai|subtraem|sai|saem|cai|caem|"
    # Irregulares de ação e causação no presente
    r"faz|fazem|diz|dizem|traz|trazem|vai|vão|vem|vêm|vê|veem|ouve|ouvem|"
    r"quer|querem|pode|podem|sabe|sabem|tem|têm|é|são|está|estão|dá|dão|põe|põem|"
    # Políticos, institucionais, econômicos e declarativos
    r"altera|alteram|causa|causam|provoca|provocam|gera|geram|destr[oó]i|destroem|"
    r"afirma|afirmam|declara|declaram|revela|revelam|confirma|confirmam|"
    r"esconde|escondem|publica|publicam|pro[ií]be|pro[ií]bem|autoriza|autorizam|"
    r"aprova|aprovam|cancela|cancelam|aumenta|aumentam|reduz|reduzem|"
    r"cobra|cobram|compra|compram|vende|vendem"
    r")\b",
    re.IGNORECASE,
)

# Substantivos que coincidem com sufixos verbais
NON_VERB_SUFFIX_EXCLUSIONS = {
    "museu", "troféu", "judeu", "breu", "plebeu", "céu", "meu", "seu", "teu",
    "ouro", "touro", "louro", "besouro", "show", "fuzil", "barril", "gentil", "abril",
    "trava", "brava", "escrava", "oitava", "dia", "guia", "bacia", "magia", "copia",
    "padaria", "farmácia", "drogaria", "delegacia", "mídia", "notícia", "família",
    "polícia", "estratégia", "maioria", "minoria", "energia", "pandemia", "indústria",
    "maracujá", "guaraná", "pará", "alvará", "carajá", "tamanduá", "jacarandá",
}

# Padrões de ruídos sensacionalistas, emojis e apelos à ação
NOISE_PATTERNS = re.compile(
    r"\b(?:"
    r"compartilhe[ms]?|repass[ae][ms]?|divulgu?e[ms]?|espalh[ae][ms]?|viraliz[ae][ms]?|"
    r"veja[ms]? antes que apaguem|veja[ms]?|olh[ae][ms]?|assist[ae][ms]?|acord[ae][ms]?|salv[ae][ms]?|"
    r"não deixe[ms]? de (?:repassar|compartilhar)|mande[ms]? para todos|leia[ms]?|"
    r"cliqu[ae][ms]?|acess[ae][ms]?|"
    r"(?:a\s+)?(?:grande\s+|tradicional\s+)?m[ií]dia\s+(?:n[aã]o\s+vai\s+(?:mostrar|noticiar|passar|falar)|esconde|n[aã]o\s+mostra|cala|abafa)|"
    r"n[aã]o\s+passa\s+na\s+tv|a\s+tv\s+n[aã]o\s+mostra|a\s+globo\s+n[aã]o\s+mostra|"
    r"bom dia|boa tarde|boa noite|ol[aá] pessoal|ol[aá] a todos|paz do senhor|"
    r"gra[cç]as a deus|am[eé]m|fwd|encaminhad[ao]|"
    r"que vergonha|que absurdo|inaceit[aá]vel|inacredit[aá]vel|isso é uma vergonha|"
    r"isso é um absurdo|parabéns aos envolvidos|deus nos livre|deus nos acuda|"
    r"vamos orar|temos que orar|oremos|lament[aá]vel|vergonhoso|"
    r"urgente|bomba|alerta|aten[cç][aã]o|cuidado|"
    r"voc[eê] sabia|voc[eê]s sabiam|at[eé] quando|o que acham|ser[aá] verdade|ser[aá] que"
    r")\b",
    re.IGNORECASE,
)

CORE_ORGANIZATIONS = re.compile(
    r"\b(?:"
    r"stf|tse|stj|sus|anvisa|fiocruz|oms|minist[eé]rio(?:\s+da\s+[a-zA-Záéíóúâêîôûãõç]+)?|governo(?:\s+federal)?|"
    r"senado|c[aâ]mara|pol[ií]cia(?:\s+federal|\s+rodovi[aá]ria)?|pf|prf|"
    r"hospital|banco central|receita federal|inss|ibge|petrobras"
    r")\b",
    re.IGNORECASE,
)

CORE_ROLES_OR_PERSONS = re.compile(
    r"\b(?:"
    r"m[eé]dico[s]?|m[eé]dica[s]?|doutor[es]?|doutora[s]?|cirurgi[aã]o[s]?|cirurgi[aã][s]?|"
    r"cientista[s]?|pesquisador[es]?|especialista[s]?|"
    r"ministro[s]?|presidente[s]?|governador[es]?|deputado[s]?|senador[es]?|juiz[es]?|ju[ií]za[s]?"
    r")\b",
    re.IGNORECASE,
)



# ==============================================================================
# CAMADA 2: LIMPEZA & NER (spaCy com Fallback Inteligente)
# ==============================================================================

class SpacyNERCleaner:
    """
    Camada 2: Limpeza, desruidificação e Reconhecimento de Entidades Nomeadas (NER).
    - Remove caracteres de alarme (emojis de sirene, pânico, repasse).
    - Suprime pontuações redundantes.
    - Reconhece entidades (PER, ORG, LOC) via spaCy (pt_core_news) ou fallback integrado.
    """

    def __init__(self) -> None:
        self.nlp = None
        try:
            import spacy
            for model_name in ("pt_core_news_sm", "pt_core_news_md", "pt_core_news_lg"):
                try:
                    self.nlp = spacy.load(model_name)
                    logger.info("Modelo spaCy '%s' inicializado com sucesso.", model_name)
                    break
                except Exception:
                    continue
        except ImportError:
            self.nlp = None

    def clean_text(self, text: str) -> str:
        """Remove emojis de alarme, pontuações de pânico e decanta alertas sensacionalistas."""
        # 1. Remove emojis alarmistas e caracteres especiais de mensageria
        t = re.sub(r"[🚨⚠️💣🔥🛑📢👀⚡🇧🇷❌‼️⁉️]", " ", text)

        # 2. Normaliza pontuações repetidas (ex: '???' -> '?', '!!!' -> '.')
        t = re.sub(r"\?{2,}", "?", t)
        t = re.sub(r"!{2,}", ".", t)
        t = re.sub(r"\.{2,}", ".", t)

        # 3. Normaliza espaços antes de decapar cabeçalhos
        t = re.sub(r"\s+", " ", t).strip()

        # 4. Remove cabeçalhos de alarme isolados no início (suporta múltiplos encadeados)
        t = re.sub(
            r"^(?:(?:URGENTE|BOMBA|ALERTA|ATENÇÃO BRASIL)[!.:\s-]*)+",
            "",
            t,
            flags=re.IGNORECASE,
        ).strip()

        # 5. Remove pontuação órfã no início
        t = re.sub(r"^[\s,;.:-]+", "", t).strip()
        return t


    def extract_entities(self, text: str) -> dict[str, list[str]]:
        """Identifica entidades nomeadas categorizadas (PER, ORG, LOC)."""
        entities: dict[str, list[str]] = {"PER": [], "ORG": [], "LOC": []}

        # Modo 1: spaCy instalado e modelo disponível
        if self.nlp is not None:
            try:
                doc = self.nlp(text)
                for ent in doc.ents:
                    label = ent.label_
                    val = ent.text.strip()
                    if label in entities and val not in entities[label]:
                        entities[label].append(val)
                return entities
            except Exception as e:
                logger.warning("Falha na inferência do spaCy, aplicando fallback: %s", e)

        # Modo 2: Fallback determinístico baseado em entidades estruturais e sintagmas
        # ORGs institucionais
        for m in CORE_ORGANIZATIONS.finditer(text):
            val = m.group(0).strip()
            org_norm = val.upper() if len(val) <= 4 else val.title()
            if org_norm not in entities["ORG"]:
                entities["ORG"].append(org_norm)

        # PER (Papéis de destaque: médicos, cientistas, autoridades, ministros)
        for m in CORE_ROLES_OR_PERSONS.finditer(text):
            val = m.group(0).strip().title()
            if val not in entities["PER"]:
                entities["PER"].append(val)

        # PER (Pessoas públicas / nomes próprios com maiúsculas compostas)
        per_pattern = re.compile(
            r"\b(?:[A-ZÁÉÍÓÚÂÊÎÔÛÃÕÇ][a-záéíóúâêîôûãõç]+(?:\s+(?:de|da|do|dos|das|e))?\s+[A-ZÁÉÍÓÚÂÊÎÔÛÃÕÇ][a-záéíóúâêîôûãõç]+)\b"
        )
        for m in per_pattern.finditer(text):
            candidate = m.group(0).strip()
            # Ignora se for parte de entidade institucional
            if not CORE_ORGANIZATIONS.search(candidate) and candidate not in entities["PER"]:
                entities["PER"].append(candidate)


        # LOC (Cidades, estados, países comuns em contexto de desinformação)
        loc_pattern = re.compile(
            r"\b(?:Brasil|Brasília|São Paulo|Rio de Janeiro|Minas Gerais|Bahia|Paraná|DF|SP|RJ|MG|China|EUA|Rússia)\b",
            re.IGNORECASE,
        )
        for m in loc_pattern.finditer(text):
            candidate = m.group(0).strip().title()
            if candidate not in entities["LOC"]:
                entities["LOC"].append(candidate)

        return entities


# ==============================================================================
# CAMADA 3: DECOMPOSIÇÃO (LLM com Instructor/Structured Outputs + Fallback Sintático)
# ==============================================================================

class ClaimDecomposer:
    """
    Camada 3: Decomposição Atômica e Extração de Triplas Semânticas.
    Normaliza coloquialismos e quebra períodos compostos em proposições atômicas
    independentes, extraindo a tripla (SPO) e fontes de verificação sugeridas.
    """

    def __init__(self) -> None:
        self.settings = get_settings()

    def _infer_source_types(self, text: str, entities: dict[str, list[str]]) -> list[VerificationSourceType]:
        """Infere os tipos de fontes oficiais mais adequados para averiguação da asserção."""
        sources: list[VerificationSourceType] = []
        lower = text.lower()
        orgs = [o.lower() for o in entities.get("ORG", [])]

        # Agências reguladoras, saúde pública e institutos de pesquisa
        if any(w in lower for w in ("anvisa", "vacina", "medicamento", "remédio", "remédios", "saúde", "dengue", "vírus", "azeite", "lote", "médico", "coração", "vermes", "carne", "doença", "hospital")) or any("anvisa" in o for o in orgs):
            sources.append(VerificationSourceType.INSTITUTO_PESQUISA)
            sources.append(VerificationSourceType.AGENCIA_REGULADORA)
            sources.append(VerificationSourceType.ORGAO_OFICIAL)


        # Poder judiciário e polícia
        if any(w in lower for w in ("stf", "tse", "stj", "polícia", "pf", "preso", "prisão", "ministro", "juiz", "tribunal", "urna", "eleição")) or any(o in ("stf", "tse", "pf") for o in orgs):
            sources.append(VerificationSourceType.PODER_JUDICIARIO)
            sources.append(VerificationSourceType.ORGAO_OFICIAL)

        # Dados públicos, economia e institutos
        if any(w in lower for w in ("petrobras", "combustível", "gasolina", "diesel", "imposto", "receita federal", "inss", "governo", "ibge")):
            sources.append(VerificationSourceType.DADOS_PUBLICOS)
            sources.append(VerificationSourceType.ORGAO_OFICIAL)

        if not sources:
            sources = [VerificationSourceType.ORGAO_OFICIAL, VerificationSourceType.AGENCIA_CHECAGEM]

        return list(dict.fromkeys(sources))

    def _extract_triple_from_sentence(self, sentence: str, entities: dict[str, list[str]]) -> KnowledgeTriple:
        """Extrai deterministicamente a tripla de conhecimento (sujeito, predicado, objeto)."""
        clean_s = sentence.strip().rstrip(".?!")

        # 1. Procura primeiro entidades identificadas como sujeito
        subject = "Fato Noticiado"
        all_ents = entities.get("ORG", []) + entities.get("PER", [])
        for ent in all_ents:
            if ent.lower() in clean_s.lower():
                # Encontra a posição exata
                idx = clean_s.lower().find(ent.lower())
                subject = clean_s[idx : idx + len(ent)]
                break

        # Se não achou em entidades, busca sintagma nominal inicial
        if subject == "Fato Noticiado":
            np_match = re.match(
                r"^(?:O|A|Os|As|Um|Uma)?\s*([a-zA-ZáéíóúâêîôûãõçÁÉÍÓÚÂÊÎÔÛÃÕÇ]+(?:\s+[a-zA-ZáéíóúâêîôûãõçÁÉÍÓÚÂÊÎÔÛÃÕÇ]+){0,3})\b",
                clean_s,
            )
            if np_match:
                subject = np_match.group(0).strip()

        # 2. Localiza o verbo central da oração
        predicate = "afirmação sobre"
        verb_candidates = (
            list(PASSIVE_VOICE_PATTERN.finditer(clean_s)) +
            list(VERBAL_PERIPHRASIS_PATTERN.finditer(clean_s)) +
            list(PAST_INDICATIVE_SUFFIXES.finditer(clean_s)) +
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

            # Objeto é o complemento após o predicado verbal
            after_verb = clean_s[v_end:].strip()
            obj = after_verb if len(after_verb) > 2 else "ocorrência descrita"
        else:
            obj = clean_s

        return KnowledgeTriple(
            subject=subject,
            predicate=predicate,
            object=obj,
        )

    async def decompose_via_llm(
        self,
        cleaned_text: str,
        entities: dict[str, list[str]],
    ) -> list[AtomicAssertion] | None:
        """
        Decompõe períodos compostos em proposições atômicas independentes via LLM.
        Garante tipagem rigorosa conforme o schema Pydantic.
        """
        provider = self.settings.LLM_PROVIDER
        endpoint = self.settings.get_llm_endpoint()
        model = self.settings.get_llm_model()
        headers = self.settings.get_llm_headers()

        # Se for OpenAI mas não houver chave configurada, pula para o fallback
        if provider.lower() == "openai" and not self.settings.OPENAI_API_KEY:
            return None

        system_instruction = (
            "Você é um motor analítico especializado em extração, normalização e decomposição de alegações factuais "
            "para sistemas automatizados de checagem de fatos (Fact-Checking Pipeline).\n"
            "Sua única responsabilidade é processar textos pré-filtrados (muitas vezes sensacionalistas, informais ou com ruídos "
            "de pontuação/caixa alta) e decompô-los em proposições atômicas, falseáveis e independentes.\n\n"
            "### DIRETRIZES FUNDAMENTAIS:\n"
            "1. ATOMICIDADE:\n"
            "   - Divida alegações compostas em afirmações unitárias. Cada fato deve poder ser classificado como 'Verdadeiro' ou 'Falso' de forma totalmente independente.\n"
            "2. SEPARAÇÃO RIGOROSA DE ATRIBUIÇÃO (CITAÇÃO vs. CONTEÚDO):\n"
            "   - Se o texto afirma que uma entidade declarou algo (ex: 'X disse que Y aconteceu'), gere OBRIGATORIAMENTE duas alegações separadas:\n"
            "     a) A alegação de atribuição/fala: se X realmente declarou aquilo.\n"
            "     b) A alegação de mérito: se Y realmente aconteceu no mundo real.\n"
            "3. NORMALIZAÇÃO SEMÂNTICA SEM ALUCINAÇÃO:\n"
            "   - Elimine hipérboles, sensacionalismo e exclamações.\n"
            "   - Converta termos informais ou coloquiais para linguagem formal e objetiva.\n"
            "   - NUNCA invente fatos ausentes. Preserve estritamente as entidades, locais e números informados no texto original.\n"
            "   - Se uma informação for vaga (ex.: 'médicos afirmam' sem citar nomes), preserve a fonte genérica ('médicos não identificados').\n"
            "4. FORMATAÇÃO E ESTRUTURA:\n"
            "   - Responda EXCLUSIVAMENTE em formato JSON com a chave raiz 'assertions', contendo uma lista de objetos com:\n"
            "     * id: número inteiro (1, 2, ...)\n"
            "     * statement: string (fato atômico normalizado em ordem direta)\n"
            "     * triple: objeto com subject (string), predicate (string), object (string)\n"
            "     * suggested_source_types: lista com valores válidos: ['ORGAO_OFICIAL', 'AGENCIA_REGULADORA', 'PODER_JUDICIARIO', 'INSTITUTO_PESQUISA', 'AGENCIA_CHECAGEM', 'DADOS_PUBLICOS']\n"
            "     * is_check_worthy: boolean (true se for fato verificável concreto, false se for irrelevante/não verificável)\n"
            "   - Não inclua explicações, comentários introdutórios ou markdown em torno do JSON."
        )

        prompt = (
            f"Texto: \"{cleaned_text}\"\n"
            f"Entidades pré-detectadas: {json.dumps(entities, ensure_ascii=False)}\n\n"
            "Decomponha o texto acima em proposições atômicas, separando citações/declarações do conteúdo factual subjacente "
            "e normalizando a linguagem para termos objetivos."
        )

        # Timeout ajustado para modelos locais (ex: phi3.5 / qwen3.5 rodando no Ollama)
        timeout_seconds = 45.0 if provider.lower() == "ollama" else 15.0

        try:
            async with httpx.AsyncClient(timeout=timeout_seconds) as client:
                resp = await client.post(
                    endpoint,
                    headers=headers,
                    json={
                        "model": model,
                        "messages": [
                            {"role": "system", "content": system_instruction},
                            {"role": "user", "content": prompt},
                        ],
                        "response_format": {"type": "json_object"},
                        "temperature": 0.1,
                    },
                )
                if resp.status_code == 200:
                    data = resp.json()
                    content = data["choices"][0]["message"]["content"]
                    parsed = json.loads(content)
                    raw_assertions = parsed.get("assertions", [])
                    assertions: list[AtomicAssertion] = []
                    for idx, a in enumerate(raw_assertions, 1):
                        assertions.append(AtomicAssertion.model_validate(a))
                    return assertions
                else:
                    logger.debug(
                        "LLM (%s: %s) retornou status %s: %s. Utilizando fallback sintático.",
                        provider,
                        model,
                        resp.status_code,
                        resp.text,
                    )
        except Exception as e:
            logger.debug(
                "LLM (%s: %s) em %s não respondeu (%s). Utilizando fallback sintático.",
                provider,
                model,
                endpoint,
                e,
            )

        return None


    def decompose_syntactic(
        self,
        cleaned_text: str,
        entities: dict[str, list[str]],
    ) -> tuple[list[AtomicAssertion], list[str]]:
        """
        Decomposição sintática e morfológica determinística (fallback autossuficiente).
        Desmembra orações coordenadas/subordinadas e filtra fragmentos ruidosos.
        """
        # 1. Segmentação inicial de sentenças
        raw_sentences = [
            s.strip() for s in re.split(r"[.!?\n]+", cleaned_text) if s.strip()
        ]

        atomic_candidates: list[str] = []
        discarded: list[str] = []

        # 2. Decomposição de períodos compostos em orações atômicas
        # Conectivos explicativos, causais e adversativos: porque, pois, e que, mas, já que, visto que
        clause_splitter = re.compile(
            r"\b(?:porque|por que|já que|visto que|pois|mas|porém|contudo|todavia|enquanto|e que)\b",
            re.IGNORECASE,
        )

        for sent in raw_sentences:
            # Verifica se a frase inteira é ruído puro
            if NOISE_PATTERNS.search(sent) and (len(sent.split()) <= 6 or "mídia" in sent.lower() or "tv" in sent.lower()):
                discarded.append(sent)
                continue

            sub_clauses = [c.strip() for c in clause_splitter.split(sent) if c.strip()]
            for clause in sub_clauses:
                # Decapita imperativos residuais
                cleaned_clause = re.sub(
                    r"^(?:veja|olhe|compartilhe|repassem|atenção|urgente|bomba)[!:\s,-]*",
                    "",
                    clause,
                    flags=re.IGNORECASE,
                ).strip()

                if not cleaned_clause:
                    continue

                if NOISE_PATTERNS.search(cleaned_clause):
                    discarded.append(cleaned_clause)
                    continue


                # Checa se possui verbo finito
                has_verb = (
                    PASSIVE_VOICE_PATTERN.search(cleaned_clause) is not None or
                    VERBAL_PERIPHRASIS_PATTERN.search(cleaned_clause) is not None or
                    PAST_INDICATIVE_SUFFIXES.search(cleaned_clause) is not None or
                    FUTURE_INDICATIVE_SUFFIXES.search(cleaned_clause) is not None or
                    PRESENT_DECLARATIVE_VERBS.search(cleaned_clause) is not None or
                    PAST_IRREGULAR_VERBS.search(cleaned_clause) is not None
                )

                if has_verb and len(cleaned_clause.split()) >= 3:
                    atomic_candidates.append(cleaned_clause)
                else:
                    discarded.append(cleaned_clause)

        # 3. Monta as asserções atômicas com suas triplas e fontes sugeridas
        assertions: list[AtomicAssertion] = []
        for idx, statement in enumerate(atomic_candidates, 1):
            triple = self._extract_triple_from_sentence(statement, entities)
            source_types = self._infer_source_types(statement, entities)
            assertions.append(
                AtomicAssertion(
                    id=idx,
                    statement=statement,
                    triple=triple,
                    suggested_source_types=source_types,
                    is_check_worthy=True,
                )
            )

        return assertions, discarded


# ==============================================================================
# PIPELINE INTEGRADO: ANALISADOR 4 (BaseAnalyzer)
# ==============================================================================

@register_analyzer("claim_extractor", weight=0.0)
class ClaimExtractorAnalyzer(BaseAnalyzer):
    """
    Analisador 4: Extrator e Decompositor de Alegações (3 Camadas).

    1. Camada de Contrato (Pydantic): Retorno estrito tipado (ClaimExtractionContract).
    2. Camada de Limpeza & NER (spaCy): Higienização de alarmes e extração de PER, ORG, LOC.
    3. Camada de Decomposição (LLM/Sintática): Proposições atômicas, triplas e fontes sugeridas.
    """

    def __init__(self) -> None:
        self.cleaner = SpacyNERCleaner()
        self.decomposer = ClaimDecomposer()

    async def extract_contract(self, text: str) -> ClaimExtractionContract:
        """
        Executa o pipeline completo de 3 camadas e gera o contrato estrito Pydantic.
        """
        # Camada 2: Limpeza e Reconhecimento de Entidades
        cleaned = self.cleaner.clean_text(text)
        entities = self.cleaner.extract_entities(cleaned)

        # Camada 3: Decomposição Atômica e Extração de Triplas
        assertions = await self.decomposer.decompose_via_llm(cleaned, entities)
        discarded: list[str] = []
        engine_used: str = f"llm:{self.decomposer.settings.get_llm_model()}"

        if assertions is None:
            # Fallback determinístico sintático
            assertions, discarded = self.decomposer.decompose_syntactic(cleaned, entities)
            engine_used = "syntactic_fallback"

        return ClaimExtractionContract(
            original_text=text,
            cleaned_text=cleaned,
            entities=entities,
            assertions=assertions,
            discarded_fragments=discarded,
            engine_used=engine_used,
        )

    async def analyze(self, text: str, urls: list[str]) -> AnalyzerResult:
        """
        Integração com o pipeline de orquestração do FactChkBR.
        Retorna a alegação factual principal e disponibiliza o contrato completo em raw_details.
        """
        contract = await self.extract_contract(text)

        primary_assertion = contract.assertions[0] if contract.assertions else None
        primary_claim = primary_assertion.statement if primary_assertion else None

        extracted_claims = [a.statement for a in contract.assertions]

        reasons: list[str] = []
        if primary_claim:
            reasons.append(f"Alegação factual isolada: \"{primary_claim}\"")
            if contract.discarded_fragments:
                reasons.append(
                    f"Filtrados {len(contract.discarded_fragments)} fragmento(s) de ruído ou apelos à ação."
                )
        else:
            reasons.append(
                "Nenhuma alegação factual assertiva identificada no texto (conteúdo prioritariamente opinativo ou conversacional)."
            )

        summary = (
            f"Extração e decomposição concluída: {len(contract.assertions)} proposição(ões) atômica(s) identificada(s)."
            if contract.assertions
            else "Extração concluída: texto sem proposições factuais checáveis."
        )

        return AnalyzerResult(
            analyzer_name="claim_extractor",
            verdict=None,
            confidence=0.0,
            claim=primary_claim,
            summary=summary,
            reasons=reasons,
            sources=["Extrator de Alegações FactChkBR (Contrato Pydantic, spaCy & Decomposição)"],
            raw_details={
                "contract": contract.model_dump(),
                "engine_used": contract.engine_used,
                "extracted_claims": extracted_claims,
                "extracted_claim": primary_claim or "",
                "primary_claim": primary_claim,
                "assertions": [a.model_dump() for a in contract.assertions],
                "entities": contract.entities,
                "discarded_sentences": contract.discarded_fragments,
                "total_sentences": len(contract.assertions) + len(contract.discarded_fragments),
                "factual_sentences_count": len(contract.assertions),
                "claims_found": len(contract.assertions),
                "check_worthiness_score": 1.0 if primary_claim else 0.0,
            },
        )


# ==============================================================================
# FORMATAÇÃO CLI E TESTES RÁPIDOS NO TERMINAL
# ==============================================================================

def format_cli_result(res: AnalyzerResult) -> None:
    """Imprime no terminal o resultado da extração formatado amigavelmente com as 3 camadas."""
    print("\n" + "=" * 75)
    raw = res.raw_details or {}
    engine = raw.get("engine_used", "syntactic_fallback")
    engine_label = f"LLM Local ({engine})" if "llm" in engine else "Motor Algorítmico / Sintático (Fallback Offline)"
    print(f"⚙️  MOTOR EXECUTOR: {engine_label}")

    if res.claim:
        print(f"🎯 ALEGAÇÃO PRINCIPAL ISOLADA (CLAIM):\n   👉 \"{res.claim}\"")
    else:
        print("⚠️  NENHUMA ALEGAÇÃO FACTUAL IDENTIFICADA (Texto opinativo/saudação/ruído)")


    raw = res.raw_details or {}
    assertions = raw.get("assertions", [])

    if assertions:
        print(f"\n🧬 PROPOSIÇÕES ATÔMICAS & TRIPLAS (SPO) [{len(assertions)}]:")
        for a in assertions:
            triple = a.get("triple", {})
            sources = a.get("suggested_source_types", [])
            print(f"   [{a.get('id', 1)}] Statement: \"{a.get('statement', '')}\"")
            print(f"       ├─ Tripla:  ({triple.get('subject')} ➔ {triple.get('predicate')} ➔ {triple.get('object')})")
            print(f"       └─ Fontes:  {sources}")

    entities = raw.get("entities", {})
    if any(entities.values()):
        print(f"\n🏷️  ENTIDADES IDENTIFICADAS (NER):")
        for cat, items in entities.items():
            if items:
                print(f"   • {cat}: {items}")

    discarded = raw.get("discarded_sentences", [])
    if discarded:
        print(f"\n🗑️  FRAGMENTOS/RUÍDOS DESCARTADOS ({len(discarded)}):")
        for d in discarded:
            print(f"   ✖ \"{d}\"")

    print(f"\n📊 MÉTRICAS:")
    print(f"   • Proposições atômicas: {raw.get('factual_sentences_count', 0)}")
    print(f"   • Fragmentos de ruído:  {len(discarded)}")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    import asyncio
    import select
    import sys
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    async def _run_cli() -> None:
        extractor = ClaimExtractorAnalyzer()

        # 1. Se passou o texto como argumento direto:
        if len(sys.argv) > 1:
            text = " ".join(sys.argv[1:])
            print("\n🔍 Processando texto nas 3 camadas (Contrato, NER & Decomposição)...")
            result = await extractor.analyze(text, [])
            format_cli_result(result)
            return

        # 2. Modo Interativo Contínuo
        print("=" * 75)
        print("🔎 FactChkBR - Extrator Factual (3 Camadas: Contrato, NER & Decomposição)")
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
                result = await extractor.analyze(full_text, [])
                format_cli_result(result)

            except (KeyboardInterrupt, EOFError):
                print("\nSessão encerrada.")
                break

    asyncio.run(_run_cli())