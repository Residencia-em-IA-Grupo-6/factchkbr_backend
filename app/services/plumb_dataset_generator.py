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
    PlumbFusedResult,
    PlumbTopicClassifier,
    PlumbTopicResult,
    get_plumb_classifier,
    get_plumb_classifier_engine,
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
RE_REPORTING = re.compile(
    r"^(?:(?:mensagens?|publicaç(?:ão|ões)|vídeos?|áudios?|boatos?|postagens?|posts?|usuários?|conteúdos?|textos?)"
    r"(?:\s+(?:compartilhados?|viralizados?|divulgados?))?"
    r"(?:\s+(?:nas|em|pelas)\s+redes\s+sociais|\s+(?:na|pela)\s+internet|\s+(?:no|pelo)\s+whatsapp|\s+online|\s+no\s+telegram)?"
    r"\s+(?:afirmam?|dizem?|alegam?|garantem?|mostram?|prometem?|indicam?|sustentam?|divulgam?)\s+(?:que\s+|de\s+que\s+|uma\s+|a\s+)?"
    r"|(?:circula|compartilhado|viralizou|espalhou-se|corre)\s+(?:(?:nas|em)\s+redes|(?:no|pelo)\s+whatsapp|(?:no|pelo)\s+facebook|(?:no|pelo)\s+telegram|na\s+internet)?\s*(?:que|boato de que|afirmação de que)?\s*"
    r"|(?:segundo|de acordo com|conforme)\s+[^,.:]+,\s*)",
    re.IGNORECASE,
)


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
                nlp = spacy.load(self.spacy_model, exclude=["lemmatizer", "ner", "parser"])
                nlp.add_pipe("sentencizer")
                ClaimCondenser._nlp = nlp
            except Exception:
                # Tenta modelos alternativos se disponíveis
                for alt in ("pt_core_news_md", "pt_core_news_lg"):
                    try:
                        nlp = spacy.load(alt, exclude=["lemmatizer", "ner", "parser"])
                        nlp.add_pipe("sentencizer")
                        ClaimCondenser._nlp = nlp
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

    @staticmethod
    def clean_claim_statement(claim: str) -> str:
        """Remove orações atributivas de discurso indireto para isolar o fato falseável."""
        c = RE_REPORTING.sub("", claim.strip()).strip()
        if c and c[0].islower():
            c = c[0].upper() + c[1:]
        return c

    def extract_lead_claim(self, text: str, title: str | None = None) -> str:
        """
        Extrai o lide principal (a afirmação mais factual e representativa) da notícia,
        limpando enquadramentos de discurso indireto.
        """
        clean_title = self.clean_claim_statement(self.clean_raw_text(title or "").strip())
        clean_body = self.clean_raw_text(text).strip()

        # Se o título já for uma oração factual substantiva completa (>= 4 palavras)
        if clean_title and len(clean_title.split()) >= 4:
            # Se for conciso e direto (<= 140 chars), é o melhor candidato
            if len(clean_title) <= 140 and not clean_title.endswith("?"):
                return clean_title

        if not clean_body:
            return clean_title or ""

        # Divide o corpo em sentenças utilizando quebras de pontuação terminal legítimas
        raw_sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", clean_body) if s.strip()]
        for s_str in raw_sentences:
            s_clean = self.clean_claim_statement(s_str)
            if len(s_clean) < 20 or len(s_clean.split()) < 4 or s_clean.endswith("?"):
                continue
            if len(s_clean) <= 250:
                return s_clean

        # Se a primeira sentença for mais longa, tenta truncar no último espaço antes de 220
        first_s = self.clean_claim_statement(raw_sentences[0] if raw_sentences else clean_body)
        if len(first_s) > 220:
            return first_s[:220].rsplit(" ", 1)[0] + "."
        return first_s

    def extract_claims(
        self,
        text: str,
        title: str | None = None,
        max_claims: int = 5,
        strategy: str = "spacy",
    ) -> list[str]:
        """
        Extrai N alegações candidatas a partir da notícia de acordo com a estratégia:
        - 'spacy': Segmenta a matéria em múltiplas sentenças declarativas factuais, sem discurso indireto.
        - 'lead': Retorna o lide principal informativo limpo.
        - 'full': Mantém o texto limpo se for curto, ou os primeiros 200 caracteres.
        """
        clean_title = self.clean_claim_statement(self.clean_raw_text(title or "").strip())
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
                doc = nlp(corpus[:3000])
                for sent in doc.sents:
                    s_str = self.clean_claim_statement(sent.text.strip())
                    if len(s_str) < 20 or len(s_str.split()) < 4 or s_str.endswith("?"):
                        continue
                    has_verb = any(t.pos_ in ("VERB", "AUX") for t in sent)
                    if has_verb and s_str not in claims:
                        claims.append(s_str[:220])
                        if len(claims) >= max_claims:
                            break
            else:
                sentences = re.split(r"(?<=[.!?])\s+", corpus)
                for s in sentences:
                    s_str = self.clean_claim_statement(s.strip())
                    if len(s_str) >= 20 and len(s_str.split()) >= 4 and not s_str.endswith("?"):
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
        classifier: Any = None,
        condenser: ClaimCondenser | None = None,
        engine: str = "auto",
        fused: bool = True,
        target_topics: list[str] | None = None,
    ) -> None:
        self.engine = engine
        self.classifier = classifier or get_plumb_classifier_engine(engine)
        self.condenser = condenser or ClaimCondenser()
        self.fused = fused
        self.target_topics = [t.lower().strip() for t in (target_topics or [])]

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
        strategy: str = "spacy",
        max_claims: int = 5,
        extra_meta: dict[str, Any] | None = None,
    ) -> list[PlumbDatasetRecord]:
        """
        Processa uma notícia completa:
        1. Decompõe o texto em 1..N alegações factuais atômicas (neural via Plumb-4B ou sintática via spaCy).
        2. Para cada alegação extraída, roda a inferência de Tema + Veredito (V/F) pelo Plumb-4B.
        3. Se target_topics estiver ativo, aplica Early Exit para tópicos irrelevantes.
        """
        # Extração de alegações: neural via Plumb-4B ou sintática via spaCy
        if strategy == "plumb" and hasattr(self.classifier, "decompose_news"):
            claims = self.classifier.decompose_news(raw_text, max_claims=max_claims)
        else:
            claims = self.condenser.extract_claims(
                text=raw_text,
                title=title,
                max_claims=max_claims,
                strategy=strategy,
            )

        if not claims:
            claims = [ClaimCondenser.clean_raw_text(raw_text)[:200]]

        records: list[PlumbDatasetRecord] = []
        for idx, claim in enumerate(claims, start=1):
            sub_id = f"{item_id}_{idx}" if len(claims) > 1 else item_id
            t_claim_start = time.perf_counter()

            # MODO 1: Fused Decision (1 único forward pass conjunto)
            if self.fused and hasattr(self.classifier, "decide_fused"):
                fused_res: PlumbFusedResult | None = self.classifier.decide_fused(claim)
                if fused_res is not None:
                    topic = fused_res.topic
                    topic_conf = fused_res.topic_confidence
                    verdict = fused_res.verdict
                    verdict_conf = fused_res.verdict_confidence
                    p_v = fused_res.prob_v
                    p_f = fused_res.prob_f
                    topic_probs = fused_res.topic_probabilities

                    # Filtragem Condicional: se fora dos target_topics, marca neutro
                    if self.target_topics and topic.lower() not in self.target_topics:
                        verdict = "INCONCLUSIVO"
                        verdict_conf = 0.50

                    dur_ms = round((time.perf_counter() - t_claim_start) * 1000.0, 2)
                    records.append(
                        PlumbDatasetRecord(
                            id=sub_id,
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
                    )
                    continue

            # MODO 2: Two-Pass com Early Exit
            # Passo 2.1: Tema
            topic_res = self.classifier.classify(claim)
            topic = topic_res.topic if topic_res else "Outros"
            topic_conf = topic_res.confidence if topic_res else 0.50
            topic_probs = topic_res.probabilities if topic_res else {}

            # Filtragem Condicional / Early Exit: se o tema estiver fora dos alvos, pula o 2º forward pass!
            if self.target_topics and topic.lower() not in self.target_topics:
                dur_ms = round((time.perf_counter() - t_claim_start) * 1000.0, 2)
                records.append(
                    PlumbDatasetRecord(
                        id=sub_id,
                        noticia_original=raw_text[:1000],
                        alegacao=claim,
                        tema=topic,
                        tema_confianca=topic_conf,
                        veredito="OUT_OF_SCOPE",
                        veredito_confianca=topic_conf,
                        prob_v=0.50,
                        prob_f=0.50,
                        probabilidades_tema_json=json.dumps(topic_probs, ensure_ascii=False),
                        tempo_processamento_ms=dur_ms,
                        metadados_extras=extra_meta or {},
                    )
                )
                continue

            # Passo 2.2: Veredito Binário V/F
            context_hint = title if (title and title != claim) else None
            binary_res = self.classifier.evaluate_claim_binary(claim, context=context_hint)
            verdict = binary_res.verdict if binary_res else "F"
            verdict_conf = binary_res.confidence if binary_res else 0.50
            p_v = binary_res.prob_true if binary_res else 0.50
            p_f = binary_res.prob_false if binary_res else 0.50

            dur_ms = round((time.perf_counter() - t_claim_start) * 1000.0, 2)
            records.append(
                PlumbDatasetRecord(
                    id=sub_id,
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
            )

        return records

    def process_batch(
        self,
        batch: list[dict[str, Any]],
        strategy: str = "spacy",
        max_claims: int = 5,
    ) -> list[PlumbDatasetRecord]:
        """
        Processa um lote de itens agrupados utilizando batching paralelo no motor neural
        quando disponível (Estratégia 1: Batching Paralelo).
        """
        if not batch:
            return []

        # Se o classificador suportar batch_decide_fused e estiver em modo fused:
        if self.fused and hasattr(self.classifier, "batch_decide_fused"):
            # 1. Extração de alegações para todos os itens do lote
            extracted_items: list[tuple[dict[str, Any], str, str]] = []  # (item, claim, sub_id)
            for item in batch:
                raw_text = item.get("text", "")
                title = item.get("title")
                if strategy == "plumb" and hasattr(self.classifier, "decompose_news"):
                    claims = self.classifier.decompose_news(raw_text, max_claims=max_claims)
                else:
                    claims = self.condenser.extract_claims(
                        text=raw_text,
                        title=title,
                        max_claims=max_claims,
                        strategy=strategy,
                    )
                if not claims:
                    claims = [ClaimCondenser.clean_raw_text(raw_text)[:200]]
                for sub_idx, c in enumerate(claims, start=1):
                    sub_id = f"{item['id']}_{sub_idx}" if len(claims) > 1 else str(item["id"])
                    extracted_items.append((item, c, sub_id))

            if not extracted_items:
                return []

            claims_list = [c for _, c, _ in extracted_items]
            t0 = time.perf_counter()
            fused_results = self.classifier.batch_decide_fused(claims_list)
            dur_total_ms = (time.perf_counter() - t0) * 1000.0
            dur_per_claim_ms = round(dur_total_ms / max(len(claims_list), 1), 2)

            batch_records: list[PlumbDatasetRecord] = []
            for (item, claim, sub_id), fused_res in zip(extracted_items, fused_results):
                if fused_res is None:
                    # Fallback individual caso algum item específico falhe
                    fallback_recs = self.process_item(
                        item_id=sub_id,
                        raw_text=item["text"],
                        title=item.get("title"),
                        strategy=strategy,
                        max_claims=max_claims,
                        extra_meta=item.get("extra_meta"),
                    )
                    batch_records.extend(fallback_recs)
                    continue

                topic = fused_res.topic
                topic_conf = fused_res.topic_confidence
                verdict = fused_res.verdict
                verdict_conf = fused_res.verdict_confidence
                p_v = fused_res.prob_v
                p_f = fused_res.prob_f
                topic_probs = fused_res.topic_probabilities

                if self.target_topics and topic.lower() not in self.target_topics:
                    verdict = "INCONCLUSIVO"
                    verdict_conf = 0.50

                batch_records.append(
                    PlumbDatasetRecord(
                        id=sub_id,
                        noticia_original=item["text"][:1000],
                        alegacao=claim,
                        tema=topic,
                        tema_confianca=topic_conf,
                        veredito=verdict,
                        veredito_confianca=verdict_conf,
                        prob_v=p_v,
                        prob_f=p_f,
                        probabilidades_tema_json=json.dumps(topic_probs, ensure_ascii=False),
                        tempo_processamento_ms=dur_per_claim_ms,
                        metadados_extras=item.get("extra_meta") or {},
                    )
                )
            return batch_records

        # Fallback sequencial se o classificador não suportar batching ou for modo two-pass
        all_records: list[PlumbDatasetRecord] = []
        for item in batch:
            recs = self.process_item(
                item_id=item["id"],
                raw_text=item["text"],
                title=item.get("title"),
                strategy=strategy,
                max_claims=max_claims,
                extra_meta=item.get("extra_meta"),
            )
            all_records.extend(recs)
        return all_records

