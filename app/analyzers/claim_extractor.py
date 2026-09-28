import logging
import re
from typing import Any

from app.core.base import BaseAnalyzer
from app.core.registry import register_analyzer
from app.schemas.analysis import AnalyzerResult

logger = logging.getLogger("factchkbr.analyzers.claim_extractor")

# ==============================================================================
# MOTOR MORFOLÓGICO ADAPTATIVO (REGRAS DE CONJUGAÇÃO E SINTAXE EM PORTUGUÊS)
# ==============================================================================

# 1. Voz Passiva Analítica: auxiliar + particípio regular (-ado, -ido) ou irregular
PASSIVE_VOICE_PATTERN = re.compile(
    r"\b(?:foi|foram|é|são|será|serão|está sendo|estão sendo|acaba de ser|acabou de ser)\s+"
    r"(?:[a-zA-ZáéíóúâêîôûãõçÁÉÍÓÚÂÊÎÔÛÃÕÇ]+(?:ado|ada|ados|adas|ido|ida|idos|idas)|"
    r"preso|presa|presos|presas|morto|morta|mortos|mortas|eleito|eleita|eleitos|eleitas|"
    r"feito|feita|feitos|feitas|dito|dita|ditos|ditas|visto|vista|vistos|vistas|"
    r"descoberto|descoberta|descobertos|descobertas|suspenso|suspensa|suspensos|suspensas|"
    r"confiscado|confiscada|confiscados|confiscadas|interceptado|interceptada)\b",
    re.IGNORECASE,
)

# 2. Locuções Verbais de Ação: auxiliar no indicativo + gerúndio (-ando, -endo, -indo) ou infinitivo
VERBAL_PERIPHRASIS_PATTERN = re.compile(
    r"\b(?:está|estão|estava|estavam|vem|vêm|vinha|vinham|começou a|começaram a|acabou de|acabaram de|"
    r"tentou|tentaram|pretende|pretendem|decidiu|decidiram|vai|vão|podem?)\s+"
    r"[a-zA-ZáéíóúâêîôûãõçÁÉÍÓÚÂÊÎÔÛÃÕÇ]+(?:ando|endo|indo|ar|er|ir)\b",
    re.IGNORECASE,
)

# 3. Sufixos Morfológicos de Pretérito Perfeito Regular (3ª pessoa sing. e plural)
# Cobre qualquer verbo regular das 3 conjugações (-ar: -ou/-aram; -er: -eu/-eram; -ir: -iu/-iram)
PAST_INDICATIVE_SUFFIXES = re.compile(
    r"\b[a-zA-ZáéíóúâêîôûãõçÁÉÍÓÚÂÊÎÔÛÃÕÇ]{3,}(?:ou|aram|eu|eram|iu|iram)\b",
    re.IGNORECASE,
)

# 4. Sufixos Morfológicos de Pretérito Imperfeito Regular (-ava/-avam, -ia/-iam)
IMPERFECT_INDICATIVE_SUFFIXES = re.compile(
    r"\b[a-zA-ZáéíóúâêîôûãõçÁÉÍÓÚÂÊÎÔÛÃÕÇ]{3,}(?:ava|avam|ia|iam)\b",
    re.IGNORECASE,
)

# 5. Sufixos Morfológicos de Futuro do Presente (-ará, -arão, -erá, -erão, -irá, -irão)
FUTURE_INDICATIVE_SUFFIXES = re.compile(
    r"\b[a-zA-ZáéíóúâêîôûãõçÁÉÍÓÚÂÊÎÔÛÃÕÇ]{3,}(?:ará|arão|erá|erão|irá|irão)\b",
    re.IGNORECASE,
)

# 6. Sufixos Morfológicos de Futuro do Pretérito / Condicional (-aria, -ariam, etc.)
CONDITIONAL_INDICATIVE_SUFFIXES = re.compile(
    r"\b[a-zA-ZáéíóúâêîôûãõçÁÉÍÓÚÂÊÎÔÛÃÕÇ]{3,}(?:aria|ariam|eria|eriam|iria|iriam)\b",
    re.IGNORECASE,
)

# 7. Verbos Irregulares de Alta Frequência no Pretérito do Indicativo
PAST_IRREGULAR_VERBS = re.compile(
    r"\b(?:disse|disseram|teve|tiveram|esteve|estiveram|fez|fizeram|deu|deram|"
    r"pôs|puseram|trouxe|trouxeram|veio|vieram|viu|viram|ouviu|ouviram|"
    r"quis|quiseram|pôde|puderam|soube|souberam|houve)\b",
    re.IGNORECASE,
)

# 8. Verbos Declarativos de Ação / Causalidade no Presente do Indicativo
PRESENT_DECLARATIVE_VERBS = re.compile(
    r"\b(?:mata|matam|altera|alteram|causa|causam|provoca|provocam|gera|geram|"
    r"cura|curam|destr[oó]i|destroem|infecta|infectam|contamina|contaminam|"
    r"afirma|afirmam|declara|declaram|revela|revelam|confirma|confirmam|"
    r"esconde|escondem|publica|publicam|pro[ií]be|pro[ií]bem|autoriza|autorizam|"
    r"aprova|aprovam|cancela|cancelam|aumenta|aumentam|reduz|reduzem|"
    r"cobra|cobram|compra|compram|vende|vendem|tem|têm|é|são|está|estão)\b",
    re.IGNORECASE,
)

# Substantivos e palavras comuns que terminam com sufixos verbais (falsos positivos morfológicos)
NON_VERB_SUFFIX_EXCLUSIONS = {
    # -eu / -ou
    "museu", "troféu", "judeu", "breu", "plebeu", "céu", "meu", "seu", "teu",
    "ouro", "touro", "louro", "besouro", "show",
    # -iu / -il
    "fuzil", "barril", "gentil", "abril",
    # -ava / -ia
    "trava", "brava", "escrava", "oitava", "dia", "guia", "bacia", "magia", "copia",
    "padaria", "farmácia", "drogaria", "delegacia",
    # -ará
    "maracujá", "guaraná", "pará", "alvará", "carajá", "tamanduá", "jacarandá",
}

# ==============================================================================
# PADRÕES DE RUÍDO, APELOS À AÇÃO E SUBJETIVIDADE
# ==============================================================================

NOISE_PATTERNS = re.compile(
    r"\b(?:"
    # Imperativos de compartilhamento / chamadas
    r"compartilhe[ms]?|repass[ae][ms]?|divulgu?e[ms]?|espalh[ae][ms]?|viraliz[ae][ms]?|"
    r"veja[ms]? antes que apaguem|veja[ms]?|olh[ae][ms]?|assist[ae][ms]?|acord[ae][ms]?|salv[ae][ms]?|"
    r"não deixe[ms]? de (?:repassar|compartilhar)|mande[ms]? para todos|leia[ms]?|"
    r"cliqu[ae][ms]?|acess[ae][ms]?|"
    # Saudações e interjeições conversacionais
    r"bom dia|boa tarde|boa noite|ol[aá] pessoal|ol[aá] a todos|paz do senhor|"
    r"gra[cç]as a deus|am[eé]m|fwd|encaminhad[ao]|"
    # Julgamentos e desabafos puramente opinativos
    r"que vergonha|que absurdo|inaceit[aá]vel|inacredit[aá]vel|isso é uma vergonha|"
    r"isso é um absurdo|parabéns aos envolvidos|deus nos livre|deus nos acuda|"
    r"vamos orar|temos que orar|oremos|lament[aá]vel|vergonhoso|"
    # Rótulos sensacionalistas isolados
    r"urgente|bomba|alerta|aten[cç][aã]o|cuidado|"
    # Perguntas retóricas / indutivas
    r"voc[eê] sabia|voc[eê]s sabiam|at[eé] quando|o que acham|ser[aá] verdade|ser[aá] que"
    r")\b",
    re.IGNORECASE,
)

# Entidades institucionais, científicas ou políticas frequentes em fact-checking
CORE_ENTITIES = re.compile(
    r"\b(?:"
    r"stf|tse|stj|sus|anvisa|fiocruz|oms|minist[eé]rio|governo|presidente|"
    r"senado|c[aâ]mara|pol[ií]cia|pf|prf|vacina[s]?|v[ií]rus|medicamento[s]?|"
    r"rem[eé]dio[s]?|covid|c[aâ]ncer|dna|rna|urna[s]?|elei[cç][aã]o|elei[cç][oõ]es|"
    r"ministro[s]?|governador[es]?|deputado[s]?|senador[es]?|m[eé]dico[s]?|"
    r"pesquisa|estudo|hospital|banco central|receita federal|inss|ibge"
    r")\b",
    re.IGNORECASE,
)


@register_analyzer("claim_extractor", weight=0.0)
class ClaimExtractorAnalyzer(BaseAnalyzer):
    """
    Analisador 4: Extrator Adaptativo de Alegações (Claim Extraction / Check-worthiness).

    Opera de forma universal e adaptativa através de um motor morfológico da língua portuguesa:
    - Reconhece qualquer verbo do português em modo indicativo por sufixos de conjugação
      (passado, presente, futuro, voz passiva e locuções verbais de ação).
    - Detecta sintagmas nominais sujeitos e entidades envolvidas.
    - Filtra ruídos conversacionais (saudações, imperativos de compartilhamento, opiniões).

    NÃO emite veredito de veracidade (verdict=None, confidence=0.0). Atua no estágio inicial
    do pipeline para isolar a alegação factual checável ('claim') para o orquestrador.
    """

    def clean_text(self, text: str) -> str:
        """Remove formatações de markdown e caracteres decorativos da mensagem."""
        cleaned = re.sub(r"[\*_~`#]", " ", text)
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned.strip()

    def segment_sentences(self, text: str) -> list[str]:
        """Divide o texto em sentenças candidatas normalizadas."""
        raw_parts = re.split(r"(?:[.!?…\n\r]+|\s{2,})", text)
        candidates: list[str] = []
        for part in raw_parts:
            cleaned = re.sub(r"^[\W\d_]+|[\W_]+$", "", part).strip()
            # Descarta fragmentos mínimos (< 5 caracteres ou < 2 palavras)
            if len(cleaned) >= 5 and len(cleaned.split()) >= 2:
                candidates.append(cleaned)
        return candidates

    def detect_verbs(self, sentence: str) -> list[str]:
        """
        Detecta verbos finitos e construções verbais de forma universal por morfologia.
        Agnóstico a listas fixas: identifica qualquer verbo novo do português.
        """
        detected: list[str] = []

        # 1. Voz passiva analítica (ex: 'foi aprovado', 'foram presos', 'está sendo investigado')
        for m in PASSIVE_VOICE_PATTERN.finditer(sentence):
            detected.append(m.group(0))

        # 2. Locuções verbais de ação (ex: 'acabou de suspender', 'está planejando')
        for m in VERBAL_PERIPHRASIS_PATTERN.finditer(sentence):
            detected.append(m.group(0))

        # 3. Pretérito Perfeito regular por sufixo (-ou, -aram, -eu, -eram, -iu, -iram)
        for m in PAST_INDICATIVE_SUFFIXES.finditer(sentence):
            w = m.group(0)
            if w.lower() not in NON_VERB_SUFFIX_EXCLUSIONS:
                detected.append(w)

        # 4. Pretérito Imperfeito regular (-ava, -avam, -ia, -iam)
        for m in IMPERFECT_INDICATIVE_SUFFIXES.finditer(sentence):
            w = m.group(0)
            if w.lower() not in NON_VERB_SUFFIX_EXCLUSIONS:
                detected.append(w)

        # 5. Futuro do Presente regular (-ará, -arão, -erá, -erão, -irá, -irão)
        for m in FUTURE_INDICATIVE_SUFFIXES.finditer(sentence):
            w = m.group(0)
            if w.lower() not in NON_VERB_SUFFIX_EXCLUSIONS:
                detected.append(w)

        # 6. Futuro do Pretérito / Condicional (-aria, -ariam, etc.)
        for m in CONDITIONAL_INDICATIVE_SUFFIXES.finditer(sentence):
            w = m.group(0)
            if w.lower() not in NON_VERB_SUFFIX_EXCLUSIONS:
                detected.append(w)

        # 7. Pretéritos irregulares de alta frequência (disse, fez, teve, etc.)
        for m in PAST_IRREGULAR_VERBS.finditer(sentence):
            detected.append(m.group(0))

        # 8. Verbos declarativos frequentes no Presente do Indicativo
        for m in PRESENT_DECLARATIVE_VERBS.finditer(sentence):
            detected.append(m.group(0))

        return list(dict.fromkeys(detected))

    def detect_subject_and_entities(self, sentence: str) -> tuple[bool, list[str]]:
        """
        Identifica estrutura sintática com sujeito e entidades nominais.
        """
        # Sintagma nominal sujeito no início da frase (Artigo/Demonstrativo/Numeral + Substantivo)
        has_subject_np = bool(
            re.search(
                r"^(?:o|a|os|as|um|uma|uns|umas|este|esta|estes|estas|esse|essa|dois|três|quatro|\d+)\s+"
                r"[a-zA-ZáéíóúâêîôûãõçÁÉÍÓÚÂÊÎÔÛÃÕÇ]+",
                sentence,
                re.IGNORECASE,
            )
        )

        entities: list[str] = []

        # Nomes próprios no meio da frase (capitalização indicando entidade ou pessoa)
        tokens = sentence.split()[1:]  # ignora o primeiro token para evitar maiúscula de abertura
        for token in tokens:
            clean_tok = re.sub(r"\W+", "", token)
            if clean_tok.istitle() and len(clean_tok) >= 3 and clean_tok.lower() not in NON_VERB_SUFFIX_EXCLUSIONS:
                entities.append(clean_tok)

        # Entidades de órgãos públicos, termos médicos e políticos
        for m in CORE_ENTITIES.finditer(sentence):
            entities.append(m.group(0))

        return has_subject_np, list(dict.fromkeys(entities))

    def score_sentence(self, sentence: str) -> tuple[float, bool, dict[str, Any]]:
        """
        Calcula o escore adaptativo de factualidade / check-worthiness de uma sentença.
        """
        verbs = self.detect_verbs(sentence)
        has_subject_np, entities = self.detect_subject_and_entities(sentence)
        noise_matches = NOISE_PATTERNS.findall(sentence)

        n_verbs = len(verbs)
        n_entities = len(entities)
        has_noise = len(noise_matches) > 0

        score = 0.0

        # Verbos no indicativo (passado/presente/futuro)
        if n_verbs > 0:
            score += 0.40 * min(n_verbs, 2)

        # Bônus para voz passiva analítica ou locuções de ação
        if PASSIVE_VOICE_PATTERN.search(sentence) or VERBAL_PERIPHRASIS_PATTERN.search(sentence):
            score += 0.15

        # Estrutura com sintagma nominal sujeito explícito
        if has_subject_np:
            score += 0.15

        # Entidades nominais ou nomes próprios identificados
        if n_entities > 0:
            score += 0.20 * min(n_entities, 2)

        # Especificadores de dados: números, porcentagens, valores monetários, datas
        if re.search(r"\b(?:\d+|%|R\$|milhões|bilhões|ontem|hoje|neste ano|em \d{4})\b", sentence, re.IGNORECASE):
            score += 0.15

        # Penalidade expressiva por ruído (saudações, imperativos de compartilhamento, opiniões)
        if has_noise:
            score -= 0.35

        # Critério adaptativo para sentença factual:
        # Requer verbo finito no indicativo + pontuação mínima
        is_factual = (score >= 0.35) and (n_verbs > 0)

        meta = {
            "verbs": verbs,
            "has_subject_np": has_subject_np,
            "entities": entities,
            "noise": noise_matches,
            "score": round(max(score, 0.0), 4),
        }
        return score, is_factual, meta

    def extract_claims(self, text: str) -> list[str]:
        """Extrai todas as sentenças classificadas com teor factual."""
        cleaned = self.clean_text(text)
        sentences = self.segment_sentences(cleaned)
        claims: list[tuple[float, str]] = []

        for s in sentences:
            score, is_factual, _ = self.score_sentence(s)
            if is_factual:
                claims.append((score, s))

        claims.sort(key=lambda x: x[0], reverse=True)
        return [c[1] for c in claims]

    def extract_primary_claim(self, text: str) -> str | None:
        """Retorna a sentença factual com maior pontuação de check-worthiness."""
        claims = self.extract_claims(text)
        return claims[0] if claims else None

    async def analyze(self, text: str, urls: list[str]) -> AnalyzerResult:
        """
        Executa a extração adaptativa de alegações factuais e isolamento de ruído.
        NÃO emite veredito (verdict=None) nem confiança (confidence=0.0).
        """
        cleaned = self.clean_text(text)
        sentences = self.segment_sentences(cleaned)

        factual_claims: list[tuple[float, str]] = []
        discarded_sentences: list[dict[str, Any]] = []

        for s in sentences:
            score, is_factual, meta = self.score_sentence(s)
            if is_factual:
                factual_claims.append((score, s))
            else:
                discarded_sentences.append({
                    "sentence": s,
                    "reason": "ruído/conversacional" if meta["noise"] else "baixo teor factual",
                    "score": meta["score"],
                })

        factual_claims.sort(key=lambda x: x[0], reverse=True)
        extracted_claims = [c[1] for c in factual_claims]
        primary_claim = extracted_claims[0] if extracted_claims else None
        best_score = factual_claims[0][0] if factual_claims else 0.0

        reasons: list[str] = []
        if primary_claim:
            reasons.append(f"Alegação factual isolada: \"{primary_claim}\"")
            if discarded_sentences:
                reasons.append(
                    f"Filtrados {len(discarded_sentences)} fragmento(s) de ruído, saudações ou apelos de compartilhamento."
                )
        else:
            reasons.append(
                "Nenhuma alegação factual assertiva identificada no texto (conteúdo prioritariamente opinativo ou conversacional)."
            )

        summary = (
            f"Extração concluída: {len(extracted_claims)} sentença(s) com teor factual identificada(s)."
            if extracted_claims
            else "Extração concluída: texto sem sentenças factuais claras."
        )

        return AnalyzerResult(
            analyzer_name="claim_extractor",
            verdict=None,
            confidence=0.0,
            claim=primary_claim,
            summary=summary,
            reasons=reasons,
            sources=["Extrator de Alegações FactChkBR (Motor Morfológico Adaptativo)"],
            raw_details={
                "extracted_claims": extracted_claims,
                "extracted_claim": primary_claim or "",
                "primary_claim": primary_claim,
                "discarded_sentences": discarded_sentences,
                "total_sentences": len(sentences),
                "factual_sentences_count": len(extracted_claims),
                "claims_found": len(extracted_claims),
                "check_worthiness_score": round(max(best_score, 0.0), 4),
            },
        )

