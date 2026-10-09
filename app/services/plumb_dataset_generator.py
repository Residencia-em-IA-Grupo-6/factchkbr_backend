"""
Gerador de Dataset de Treinamento via Classificação em Lote com Plumb-4B.

Responsável por:
1. Condensar matérias/notícias longas em alegações atômicas curtas e denotativas com zero gargalo (CPU).
2. Classificar o tema da alegação (Saúde, Política, Entretenimento, Esportes, Economia, Outros) via Plumb-4B.
3. Emitir veredito binário estrito (V ou F) com probabilidades calibradas via Plumb-4B.
4. Suportar streaming/checkpointing incremental para grandes volumes em CSV, TSV e JSONL.
"""

import csv
import json
import logging
import os
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Generator, Iterable

from pydantic import BaseModel, Field

from app.services.plumb_classifier import (
    PlumbBinaryResult,
    PlumbTopicClassifier,
    PlumbTopicResult,
    get_plumb_classifier,
)

logger = logging.getLogger("factchkbr.services.plumb_dataset_generator")


# ==============================================================================
# 1. CONDENSADOR DE NOTÍCIAS EM ALEGAÇÕES CURTAS (CPU / ULTRA-RÁPIDO)
# ==============================================================================

RE_HTML_TAGS = re.compile(r"<[^>]+>")
RE_URLS = re.compile(r"https?://\S+|www\.\S+")
RE_JOURNALISTIC_CREDITS = re.compile(
    r"^(?:(?:Foto|Imagem|Crédito|Fonte|Por|Da Redação|Agência|Reportagem|Publicado em)[^:\n]*:?\s*)+",
    re.IGNORECASE,
)
RE_CLEAN_HEADER = re.compile(
    r"^(?:(?:URGENTE|BOMBA|ALERTA|ATENÇÃO BRASIL|CONFIRA|VEJA|LEIA TAMBÉM)[!.:\s-]*)+",
    re.IGNORECASE,
)
RE_EMOJIS_PUNCT = re.compile(r"[🚨⚠️💣🔥🛑📢👀⚡🇧🇷❌‼️⁉️]+")
RE_MULTIPLE_SPACES = re.compile(r"\s+")
RE_MULTIPLE_PUNCT = re.compile(r"([!?.]){2,}")


class ClaimCondenser:
    """
    Extrator e condensador determinístico de alegações curtas a partir de notícias completas.
    Opera exclusivamente em CPU com expressões regulares e spaCy, sem travar GPU/Metal.
    """
    _nlp: Any = None

    def __init__(self, spacy_model: str = "pt_core_news_sm") -> None:
        self.spacy_model = spacy_model

    def _get_spacy(self) -> Any:
        if ClaimCondenser._nlp is None:
            import spacy
            try:
                ClaimCondenser._nlp = spacy.load(self.spacy_model, exclude=["lemmatizer", "ner"])
            except Exception:
                # Tenta modelos alternativos se disponíveis
                for alt in ("pt_core_news_md", "pt_core_news_lg"):
                    try:
                        ClaimCondenser._nlp = spacy.load(alt, exclude=["lemmatizer", "ner"])
                        break
                    except Exception:
                        pass
        return ClaimCondenser._nlp

    @staticmethod
    def clean_raw_text(text: str) -> str:
        """Higieniza o texto bruto de uma notícia removendo marcações e ruídos."""
        if not text:
            return ""
        t = RE_HTML_TAGS.sub(" ", text)
        t = RE_URLS.sub(" ", t)
        t = RE_EMOJIS_PUNCT.sub(" ", t)
        t = RE_MULTIPLE_PUNCT.sub(r"\1", t)
        t = RE_JOURNALISTIC_CREDITS.sub("", t.strip())
        t = RE_CLEAN_HEADER.sub("", t.strip())
        t = RE_MULTIPLE_SPACES.sub(" ", t).strip()
        return t

    def extract_lead_claim(self, text: str, title: str | None = None) -> str:
        """
        Extrai o lide principal (a afirmação mais factual e representativa) da notícia.
        Se o título for informativo e factual, pode ser aproveitado como base.
        """
        clean_title = self.clean_raw_text(title or "").strip()
        clean_body = self.clean_raw_text(text).strip()

        # Se o título já for uma oração factual substantiva completa (>= 4 palavras)
        if clean_title and len(clean_title.split()) >= 4:
            # Se for conciso e direto (<= 140 chars), é o melhor candidato
            if len(clean_title) <= 140 and not clean_title.endswith("?"):
                return clean_title

        if not clean_body:
            return clean_title or ""

        # Se o corpo da notícia já for uma frase ou parágrafo curto e conciso (<= 250 chars)
        if len(clean_body) <= 250 and len(clean_body.split()) >= 4:
            return clean_body

        # Divide o corpo em sentenças utilizando quebras de pontuação terminal legítimas
        raw_sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", clean_body) if s.strip()]
        for s_str in raw_sentences:
            if len(s_str) < 25 or len(s_str.split()) < 4 or s_str.endswith("?"):
                continue
            if len(s_str) <= 250:
                return s_str

        # Se a primeira sentença for mais longa, tenta truncar no último espaço antes de 220
        first_s = raw_sentences[0] if raw_sentences else clean_body
        if len(first_s) > 220:
            return first_s[:220].rsplit(" ", 1)[0] + "."
        return first_s

    def extract_claims(
        self,
        text: str,
        title: str | None = None,
        max_claims: int = 1,
        strategy: str = "lead",
    ) -> list[str]:
        """
        Extrai N alegações candidatas a partir da notícia de acordo com a estratégia:
        - 'lead': Retorna o lide principal informativo.
        - 'spacy': Segmenta e retorna até max_claims sentenças factuais completas.
        - 'title_lead': Combina título e lide em proposições separadas.
        - 'full': Mantém o texto limpo se for curto, ou os primeiros 200 caracteres.
        """
        clean_title = self.clean_raw_text(title or "").strip()
        clean_body = self.clean_raw_text(text).strip()

        if strategy == "lead":
            lead = self.extract_lead_claim(clean_body, title=clean_title)
            return [lead] if lead else []

        if strategy == "full":
            src = clean_title if (clean_title and len(clean_title) > len(clean_body)) else clean_body
            return [src[:250]] if src else []

        claims: list[str] = []
        if strategy in ("spacy", "sentences"):
            nlp = self._get_spacy()
            corpus = f"{clean_title}. {clean_body}" if clean_title else clean_body
            if nlp is not None:
                doc = nlp(corpus[:2500])
                for sent in doc.sents:
                    s_str = sent.text.strip()
                    if len(s_str) < 25 or len(s_str.split()) < 4 or s_str.endswith("?"):
                        continue
                    has_verb = any(t.pos_ in ("VERB", "AUX") for t in sent)
                    if has_verb and s_str not in claims:
                        claims.append(s_str[:220])
                        if len(claims) >= max_claims:
                            break
            else:
                sentences = re.split(r"(?<=[.!?])\s+", corpus)
                for s in sentences:
                    s_str = s.strip()
                    if len(s_str) >= 25 and len(s_str.split()) >= 4 and not s_str.endswith("?"):
                        if s_str not in claims:
                            claims.append(s_str[:220])
                            if len(claims) >= max_claims:
                                break

        if not claims:
            lead = self.extract_lead_claim(clean_body, title=clean_title)
            if lead:
                claims.append(lead)

        return claims[:max_claims]


# ==============================================================================
# 2. ESQUEMA DE DADOS DE SAÍDA (DATASET RECORD)
# ==============================================================================

class PlumbDatasetRecord(BaseModel):
    """Registro individual classificado para o dataset de treino."""
    id: str | int = Field(..., description="Identificador da linha/notícia")
    noticia_original: str = Field(..., description="Texto ou trecho original da matéria")
    alegacao: str = Field(..., description="Alegação curta condensada")
    tema: str = Field(..., description="Tema classificado pelo Plumb-4B (Saúde, Política, etc.)")
    tema_confianca: float = Field(..., description="Grau de confiança do tema (0.0 a 1.0)")
    veredito: str = Field(..., description="Veredito binário (V ou F)")
    veredito_confianca: float = Field(..., description="Grau de certeza do veredito (0.0 a 1.0)")
    prob_v: float = Field(..., description="Probabilidade de ser Verdadeiro (V)")
    prob_f: float = Field(..., description="Probabilidade de ser Falso (F)")
    probabilidades_tema_json: str = Field(..., description="JSON com probabilidades por tema")
    tempo_processamento_ms: float = Field(default=0.0, description="Tempo total de processamento em ms")
    metadados_extras: dict[str, Any] = Field(default_factory=dict, description="Outras colunas preservadas")


# ==============================================================================
# 3. GERADOR DE DATASET EM LOTE (BATCH PROCESSOR)
# ==============================================================================

class PlumbDatasetGenerator:
    """
    Motor de processamento em lote que consome notícias de TSV/CSV e produz
    datasets enriquecidos com alegações curtas, tópicos e vereditos V/F pelo Plumb-4B.
    """

    def __init__(
        self,
        classifier: PlumbTopicClassifier | None = None,
        condenser: ClaimCondenser | None = None,
    ) -> None:
        self.classifier = classifier or get_plumb_classifier()
        self.condenser = condenser or ClaimCondenser()

    @staticmethod
    def detect_delimiter(file_path: str | Path, sample_bytes: int = 8192) -> str:
        """Detecta automaticamente se o arquivo é TSV, CSV ou usa ponto-e-vírgula."""
        p = Path(file_path)
        if p.suffix.lower() == ".tsv":
            return "\t"
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                sample = f.read(sample_bytes)
                sniffer = csv.Sniffer()
                dialect = sniffer.sniff(sample, delimiters=[",", "\t", ";", "|"])
                return dialect.delimiter
        except Exception:
            # Fallback seguro
            return "\t" if p.suffix.lower() == ".tsv" else ","

    @staticmethod
    def find_text_column(headers: list[str], custom_col: str | None = None) -> str:
        """Localiza a coluna que contém o texto da notícia."""
        if custom_col and custom_col in headers:
            return custom_col

        candidates = [
            "texto", "text", "noticia", "notícia", "content", "body", "materia",
            "matéria", "conteudo", "conteúdo", "article", "full_text", "news",
            "title", "titulo", "título", "manchete", "claim", "alegacao", "alegação",
        ]
        lower_headers = {h.lower().strip(): h for h in headers}

        # 1. Correspondência exata
        for cand in candidates:
            if cand in lower_headers:
                return lower_headers[cand]

        # 2. Correspondência parcial (ignorando colunas de ID como noticia_id, id_texto)
        id_suffixes = ("_id", "id_", "codigo", "_cod", "pk")
        non_id_headers = [
            h for h in headers
            if not any(h.lower().strip().endswith(s) or h.lower().strip().startswith(s) for s in id_suffixes)
            and h.lower().strip() != "id"
        ]

        for cand in candidates:
            for h in non_id_headers:
                if cand in h.lower():
                    return h

        # 3. Fallback: primeira coluna não-id ou primeira coluna disponível
        return non_id_headers[0] if non_id_headers else (headers[0] if headers else "text")


    def process_item(
        self,
        item_id: str | int,
        raw_text: str,
        title: str | None = None,
        strategy: str = "lead",
        max_claims: int = 1,
        extra_meta: dict[str, Any] | None = None,
    ) -> list[PlumbDatasetRecord]:
        """
        Processa uma única notícia:
        1. Condensa em 1..N alegações curtas (CPU).
        2. Para cada alegação, roda Plumb-4B para Tema e Veredito (V/F).
        """
        t0 = time.perf_counter()
        claims = self.condenser.extract_claims(
            text=raw_text,
            title=title,
            max_claims=max_claims,
            strategy=strategy,
        )

        if not claims:
            # Garante que haja ao menos uma alegação para avaliar
            claims = [ClaimCondenser.clean_raw_text(raw_text)[:200]]

        records: list[PlumbDatasetRecord] = []
        for claim in claims:
            t_claim_start = time.perf_counter()

            # 1. Classificação do Tema
            topic_res = self.classifier.classify(claim)
            topic = topic_res.topic if topic_res else "Outros"
            topic_conf = topic_res.confidence if topic_res else 0.50
            topic_probs = topic_res.probabilities if topic_res else {}

            # 2. Veredito Binário (V ou F)
            # Passa o título/contexto da matéria para calibrar a decisão se disponível
            context_hint = title if (title and title != claim) else None
            binary_res = self.classifier.evaluate_claim_binary(claim, context=context_hint)
            verdict = binary_res.verdict if binary_res else "F"
            verdict_conf = binary_res.confidence if binary_res else 0.50
            p_v = binary_res.prob_true if binary_res else 0.50
            p_f = binary_res.prob_false if binary_res else 0.50

            dur_ms = round((time.perf_counter() - t_claim_start) * 1000.0, 2)

            rec = PlumbDatasetRecord(
                id=item_id,
                noticia_original=raw_text[:1000],
                alegacao=claim,
                tema=topic,
                tema_confianca=topic_conf,
                veredito=verdict,
                veredito_confianca=verdict_conf,
                prob_v=p_v,
                prob_f=p_f,
                probabilidades_tema_json=json.dumps(topic_probs, ensure_ascii=False),
                tempo_processamento_ms=dur_ms,
                metadados_extras=extra_meta or {},
            )
            records.append(rec)

        return records
