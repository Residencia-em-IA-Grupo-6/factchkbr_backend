"""
Serviço de classificação de tópicos baseado no modelo Plumb-4B (crh225/plumb-4b).
Identifica o domínio temático principal de textos (Saúde, Política, Entretenimento,
Esportes, Economia, Outros) através de uma única passagem de inferência calibrada com JevK5.
"""
import asyncio
import logging
from typing import Any
from pydantic import BaseModel, Field

from app.schemas.analysis import Verdict

logger = logging.getLogger("factchkbr.services.plumb_classifier")

TOPIC_MAP: dict[str, str] = {
    "health": "Saúde",
    "politics": "Política",
    "entertainment": "Entretenimento",
    "sports": "Esportes",
    "economy": "Economia",
    "other": "Outros",
}

CRITERIA: dict[str, str] = {
    "health": "health, medicine, pharmaceuticals, treatments, weight loss, slimming, remedies, wellness, diseases, cancer, vaccines",
    "politics": "politics, elections, politicians, government, voting, congress, public policy",
    "entertainment": "entertainment, celebrities, movies, cinema, music, culture",
    "sports": "sports, football, soccer, matches, athletes",
    "economy": "economy, banks, interest rates, inflation, investments, financial market",
    "other": "greetings, casual chat, personal messages, religious blessings, other miscellaneous topics",
}

CLAIM_CRITERIA: dict[str, str] = {
    "fake": "false, debunked by fact-checkers, unauthorized or fraudulent medical product, contradicted by health authorities, or fabricated miracle claim without official evidence",
    "verdadeiro": "factually accurate, confirmed by official records, regulatory approval, or reputable journalism",
    "suspeito": "misleading, partially true, exaggerated claims, omitted risks, or conflicting facts",
    "inconclusivo": "insufficient evidence to verify or refute, ongoing investigation, or lack of conclusive data",
}

BINARY_CLAIM_CRITERIA: dict[str, str] = {
    "V": "factually true, verifiable reality, scientifically or historically accurate, real event, corroborated by facts",
    "F": "factually false, fake news, fabricated claim, pseudoscience, debunked myth, hoax, misleading or baseless rumor",
}

FUSED_CRITERIA: dict[str, str] = {
    "saude_v": "health, medicine or pharmaceuticals: factually true or scientifically confirmed statement",
    "saude_f": "health, medicine or pharmaceuticals: false claim, myth, hoax, fraudulent remedy or medical scam",
    "politica_v": "politics, elections or government: factually true or verified event",
    "politica_f": "politics, elections or government: fake news, false political claim or fabricated statement",
    "economia_v": "economy or finance: verified accurate economic event or data",
    "economia_f": "economy or finance: false claim, financial scam or fake investment",
    "esportes_v": "sports or athletic competition: verified match result or real sport event",
    "esportes_f": "sports or athletic competition: false claim, fake score or fake sport rumor",
    "entretenimento": "entertainment, celebrities, cinema, culture or gossip",
    "outros": "other miscellaneous daily topics or general conversation",
}


class PlumbTopicResult(BaseModel):
    """Resultado da classificação de tópico gerado pelo Plumb-4B."""
    topic: str = Field(..., description="Tema classificado em português: Saúde, Política, Entretenimento, Esportes, Economia, Outros")
    raw_topic: str = Field(..., description="Chave original predita pelo modelo (ex: health, politics, etc.)")
    confidence: float = Field(..., description="Confiança na predição do tópico dominante (0.0 a 1.0)")
    probabilities: dict[str, float] = Field(default_factory=dict, description="Distribuição de probabilidades normalizada por tema em português")
    is_health: bool = Field(..., description="True se o tema for classificado como Saúde")
    health_probability: float = Field(..., description="Probabilidade atribuída ao tema Saúde")


class PlumbClaimResult(BaseModel):
    """Resultado da checagem epistêmica de uma alegação factual atômica pelo Plumb-4B."""
    statement: str = Field(..., description="Alegação factual atômica avaliada")
    verdict: Verdict = Field(..., description="Veredito epistêmico emitido pelo Plumb-4B")
    confidence: float = Field(..., description="Grau de certeza da decisão (0.0 a 1.0)")
    probabilities: dict[str, float] = Field(default_factory=dict, description="Distribuição de probabilidades por veredito")
    rationale_hint: str = Field(default="", description="Pista ou resumo do critério orientador da decisão")


class PlumbBinaryResult(BaseModel):
    """Resultado da checagem binária (V ou F) emitida pelo Plumb-4B."""
    statement: str = Field(..., description="Alegação factual avaliada")
    verdict: str = Field(..., description="Veredito binário emitido pelo Plumb-4B: 'V' ou 'F'")
    confidence: float = Field(..., description="Grau de certeza da decisão (0.0 a 1.0)")
    probabilities: dict[str, float] = Field(default_factory=dict, description="Distribuição de probabilidades entre V e F")
    prob_true: float = Field(..., description="Probabilidade de ser Verdadeiro (V)")
    prob_false: float = Field(..., description="Probabilidade de ser Falso (F)")


class PlumbFusedResult(BaseModel):
    """Resultado da classificação unificada (Tema + Veredito V/F) em 1 único forward pass."""
    statement: str = Field(..., description="Alegação factual avaliada")
    topic: str = Field(..., description="Tema dominante: Saúde, Política, Entretenimento, Esportes, Economia, Outros")
    topic_confidence: float = Field(..., description="Confiança na predição do tema (0.0 a 1.0)")
    verdict: str = Field(..., description="Veredito binário dominante: 'V' ou 'F'")
    verdict_confidence: float = Field(..., description="Confiança no veredito binário (0.0 a 1.0)")
    prob_v: float = Field(..., description="Probabilidade normalizada de ser Verdadeiro (V)")
    prob_f: float = Field(..., description="Probabilidade normalizada de ser Falso (F)")
    topic_probabilities: dict[str, float] = Field(default_factory=dict, description="Probabilidades por tema")
    raw_probabilities: dict[str, float] = Field(default_factory=dict, description="Probabilidades conjuntas brutas")
    raw_choice: str = Field(default="outros", description="Chave combinada selecionada pelo modelo")




class PlumbTopicClassifier:
    """
    Classificador temático baseado no Plumb-4B (singleton lazy-loaded via JevK5).
    Utiliza o checkpoint crh225/plumb-4b com cabeça de decisão rápida em MPS/CPU.
    """
    _agent: Any = None
    _load_attempted: bool = False

    def __init__(
        self,
        model_name: str = "crh225/plumb-4b",
        device: str = "mps",
        preload: bool = False,
    ) -> None:
        self.model_name = model_name
        self.device = device
        if preload:
            self._ensure_loaded()

    def _ensure_loaded(self) -> Any:
        if PlumbTopicClassifier._agent is not None:
            return PlumbTopicClassifier._agent
        if PlumbTopicClassifier._load_attempted and PlumbTopicClassifier._agent is None:
            return None

        PlumbTopicClassifier._load_attempted = True
        try:
            import torch
            from jevk5 import JevK5

            dev = self.device
            if dev == "mps" and not torch.backends.mps.is_available():
                logger.warning("MPS não disponível no ambiente, utilizando CPU para Plumb-4B.")
                dev = "cpu"

            logger.info("Carregando modelo Plumb-4B (%s, device=%s, dtype=bfloat16, graphs=False)...", self.model_name, dev)
            PlumbTopicClassifier._agent = JevK5(
                source=self.model_name,
                device=dev,
                dtype=torch.bfloat16,
                graphs=False,
            )
            logger.info("Modelo Plumb-4B carregado com sucesso.")
            return PlumbTopicClassifier._agent
        except Exception as exc:
            logger.warning("Não foi possível carregar o modelo Plumb-4B (%s): %s", type(exc).__name__, exc)
            PlumbTopicClassifier._agent = None
            return None

    def classify(self, text: str) -> PlumbTopicResult | None:
        """
        Classifica sincronamente o texto em um dos domínios temáticos.
        Retorna PlumbTopicResult ou None se o modelo estiver indisponível.
        """
        agent = self._ensure_loaded()
        if agent is None:
            return None

        clean_text = text.strip()
        if not clean_text:
            return None

        truncated_text = clean_text[:2000]

        question = {
            "type": "choice",
            "instructions": "Classify the main topic of the text into exactly one category.",
            "criteria": CRITERIA,
        }

        try:
            res = agent.decide(truncated_text, question)
            choice = str(res.get("choice", "other"))
            raw_probs = res.get("probabilities", {})
            conf = float(res.get("confidence", 0.0))

            probs_pt: dict[str, float] = {}
            for eng_k, pt_k in TOPIC_MAP.items():
                probs_pt[pt_k] = round(float(raw_probs.get(eng_k, 0.0)), 4)

            topic_pt = TOPIC_MAP.get(choice, "Outros")
            health_prob = round(float(raw_probs.get("health", 0.0)), 4)

            return PlumbTopicResult(
                topic=topic_pt,
                raw_topic=choice,
                confidence=round(float(conf), 4),
                probabilities=probs_pt,
                is_health=(choice == "health"),
                health_probability=health_prob,
            )
        except Exception as exc:
            logger.warning("Falha na inferência do Plumb-4B (%s): %s", type(exc).__name__, exc)
            return None

    async def classify_async(self, text: str) -> PlumbTopicResult | None:
        """
        Executa a classificação de forma assíncrona em threadpool (sem travar o event loop).
        """
        return await asyncio.to_thread(self.classify, text)

    def evaluate_claim(
        self,
        claim: str,
        evidences: list[Any] | None = None,
        context: dict[str, Any] | None = None,
    ) -> PlumbClaimResult | None:
        """
        Avalia sincronamente a veracidade de uma proposição factual confrontando com evidências.
        Utiliza o motor de inferência calibrado do Plumb-4B via JevK5.
        """
        agent = self._ensure_loaded()
        if agent is None:
            return None

        clean_claim = claim.strip()
        if not clean_claim:
            return None

        simplified_evidences: list[dict[str, Any]] = []
        for e in (evidences or [])[:5]:
            if isinstance(e, dict):
                src = e.get("source_name") or e.get("source") or "Fonte"
                title = e.get("title", "")
                snippet = e.get("snippet") or e.get("finding") or ""
                rating = e.get("rating") or e.get("stance") or ""
                simplified_evidences.append({
                    "source": src,
                    "title": title,
                    "finding": snippet or title,
                    "rating": rating,
                })
            elif hasattr(e, "model_dump"):
                d = e.model_dump()
                simplified_evidences.append({
                    "source": d.get("source_name", "Fonte"),
                    "title": d.get("title", ""),
                    "finding": d.get("snippet") or d.get("title", ""),
                    "rating": d.get("rating") or d.get("stance", ""),
                })

        import json
        state = {
            "target_claim": clean_claim,
            "evidences": simplified_evidences,
            "context": context or {},
        }

        question = {
            "type": "choice",
            "instructions": (
                "Evaluate the factual truthfulness of the target claim strictly against the provided evidence, "
                "scientific reality, official health regulations (Anvisa/WHO/MS), and the legal burden of proof. "
                "Claims of unverified treatments, unauthorized miracle products, or fabricated endorsements without official backing must be classified as fake."
            ),
            "criteria": CLAIM_CRITERIA,
        }

        try:
            state_str = json.dumps(state, ensure_ascii=False)
            res = agent.decide(state_str, question)
            choice = str(res.get("choice", "inconclusivo")).lower()
            raw_probs = res.get("probabilities", {})
            conf = float(res.get("confidence", 0.70))

            verdict_map = {
                "fake": Verdict.FAKE,
                "verdadeiro": Verdict.VERDADEIRO,
                "suspeito": Verdict.SUSPEITO,
                "inconclusivo": Verdict.INCONCLUSIVO,
            }
            verdict = verdict_map.get(choice, Verdict.INCONCLUSIVO)

            probs = {k: round(float(v), 4) for k, v in raw_probs.items()}

            return PlumbClaimResult(
                statement=clean_claim,
                verdict=verdict,
                confidence=round(conf, 4),
                probabilities=probs,
                rationale_hint=f"Decisão Plumb-4B ({choice}, conf: {conf:.2f}) com {len(simplified_evidences)} evidência(s).",
            )
        except Exception as exc:
            logger.warning("Falha na avaliação de alegação do Plumb-4B (%s): %s", type(exc).__name__, exc)
            return None

    async def evaluate_claim_async(
        self,
        claim: str,
        evidences: list[Any] | None = None,
        context: dict[str, Any] | None = None,
    ) -> PlumbClaimResult | None:
        """
        Executa a avaliação de alegação de forma assíncrona em threadpool (sem bloquear o event loop).
        """
        return await asyncio.to_thread(self.evaluate_claim, claim, evidences, context)

    def evaluate_claim_binary(
        self,
        claim: str,
        context: str | None = None,
    ) -> PlumbBinaryResult | None:
        """
        Avalia a veracidade factual de forma estritamente binária (V ou F) com mínima latência.
        Utiliza 1 única passagem de inferência calibrada pelo JevK5.
        """
        agent = self._ensure_loaded()
        if agent is None:
            return None

        clean_claim = claim.strip()
        if not clean_claim:
            return None

        input_text = clean_claim if not context else f"{clean_claim}\nContexto: {context.strip()[:600]}"
        truncated_text = input_text[:2000]

        question = {
            "type": "choice",
            "instructions": (
                "Evaluate the factual truthfulness of the target statement. "
                "Is it factually true (V) or factually false/fake (F)?"
            ),
            "criteria": BINARY_CLAIM_CRITERIA,
        }

        try:
            res = agent.decide(truncated_text, question)
            choice = str(res.get("choice", "F")).upper()
            if choice not in ("V", "F"):
                choice = "F" if choice in ("FAKE", "FALSE", "0") else "V"

            raw_probs = res.get("probabilities", {})
            conf = float(res.get("confidence", 0.50))

            p_v = round(float(raw_probs.get("V", 0.50)), 4)
            p_f = round(float(raw_probs.get("F", 0.50)), 4)

            return PlumbBinaryResult(
                statement=clean_claim,
                verdict=choice,
                confidence=round(conf, 4),
                probabilities={"V": p_v, "F": p_f},
                prob_true=p_v,
                prob_false=p_f,
            )
        except Exception as exc:
            logger.warning("Falha na avaliação binária do Plumb-4B (%s): %s", type(exc).__name__, exc)
            return None

    async def evaluate_claim_binary_async(
        self,
        claim: str,
        context: str | None = None,
    ) -> PlumbBinaryResult | None:
        """
        Executa a avaliação binária de forma assíncrona em threadpool (sem bloquear o event loop).
        """
        return await asyncio.to_thread(self.evaluate_claim_binary, claim, context)

    def decide_fused(self, claim: str) -> PlumbFusedResult | None:
        """
        Classifica Tema e Veredito (V/F) em UM ÚNICO forward pass no JevK5.
        Reduz pela metade a latência de inferência por alegação.
        """
        agent = self._ensure_loaded()
        if agent is None:
            return None

        clean_claim = claim.strip()
        if not clean_claim:
            return None

        question = {
            "type": "choice",
            "instructions": (
                "Classify both the main subject domain and the factual truthfulness (true or false/fake) of the statement."
            ),
            "criteria": FUSED_CRITERIA,
        }

        try:
            res = agent.decide(clean_claim[:2000], question)
            raw_probs = res.get("probabilities", {})
            return parse_fused_probabilities(raw_probs, clean_claim)
        except Exception as exc:
            logger.warning("Falha na decisão unificada do Plumb-4B (%s): %s", type(exc).__name__, exc)
            return None

    def batch_decide_fused(self, claims: list[str]) -> list[PlumbFusedResult | None]:
        """
        Executa classificação unificada (Tema + Veredito V/F) em lote (batching paralelo).
        Processa múltiplos textos em um único forward pass no tensor do modelo,
        aumentando expressivamente o throughput de inferência.
        """
        if not claims:
            return []

        agent = self._ensure_loaded()
        if agent is None:
            return [None] * len(claims)

        from jevk5.runtime import decision_options

        question = {
            "type": "choice",
            "instructions": (
                "Classify both the main subject domain and the factual truthfulness (true or false/fake) of the statement."
            ),
            "criteria": FUSED_CRITERIA,
        }
        options = decision_options(question)
        desc_list = [opt[1] for opt in options]

        cleaned_claims = [c.strip()[:2000] for c in claims]
        try:
            import torch
            encoded_list = [agent.encode(c, question["instructions"], desc_list) for c in cleaned_claims]
            max_len = max(len(ids) for ids in encoded_list)
            b_size = len(encoded_list)

            padded = torch.zeros((b_size, max_len), dtype=torch.long, device=agent.device)
            last = torch.tensor([len(ids) - 1 for ids in encoded_list], device=agent.device)
            for i, ids in enumerate(encoded_list):
                padded[i, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=agent.device)

            with torch.inference_mode():
                hidden = agent.model.model(input_ids=padded, use_cache=False).last_hidden_state
                slot_logits = (hidden[torch.arange(b_size, device=agent.device), last] @ agent.slot_weight.T)[:, :len(options)]
                probs_tensor = torch.softmax(slot_logits.float() / agent.temperature, dim=-1).cpu().float().numpy()

            results: list[PlumbFusedResult | None] = []
            for i, c in enumerate(claims):
                p_row = probs_tensor[i]
                raw_probs = {key: float(v) for (key, _), v in zip(options, p_row, strict=True)}
                results.append(parse_fused_probabilities(raw_probs, c))
            return results
        except Exception as exc:
            logger.warning("Falha no batch_decide_fused (%s): %s. Executando fallback sequencial.", type(exc).__name__, exc)
            return [self.decide_fused(c) for c in claims]

    def batch_evaluate_claim_binary(
        self,
        claims: list[str],
        contexts: list[str | None] | None = None,
    ) -> list[PlumbBinaryResult | None]:
        """
        Avalia veracidade factual (V ou F) em lote (batching paralelo).
        """
        if not claims:
            return []

        agent = self._ensure_loaded()
        if agent is None:
            return [None] * len(claims)

        from jevk5.runtime import decision_options

        question = {
            "type": "choice",
            "instructions": (
                "Evaluate the factual truthfulness of the target statement. "
                "Is it factually true (V) or factually false/fake (F)?"
            ),
            "criteria": BINARY_CLAIM_CRITERIA,
        }
        options = decision_options(question)
        desc_list = [opt[1] for opt in options]

        contexts_list = contexts or ([None] * len(claims))
        inputs = [
            c if not ctx else f"{c.strip()}\nContexto: {str(ctx).strip()[:500]}"
            for c, ctx in zip(claims, contexts_list)
        ]
        cleaned_inputs = [inp.strip()[:2000] for inp in inputs]

        try:
            import torch
            encoded_list = [agent.encode(inp, question["instructions"], desc_list) for inp in cleaned_inputs]
            max_len = max(len(ids) for ids in encoded_list)
            b_size = len(encoded_list)

            padded = torch.zeros((b_size, max_len), dtype=torch.long, device=agent.device)
            last = torch.tensor([len(ids) - 1 for ids in encoded_list], device=agent.device)
            for i, ids in enumerate(encoded_list):
                padded[i, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=agent.device)

            with torch.inference_mode():
                hidden = agent.model.model(input_ids=padded, use_cache=False).last_hidden_state
                slot_logits = (hidden[torch.arange(b_size, device=agent.device), last] @ agent.slot_weight.T)[:, :len(options)]
                probs_tensor = torch.softmax(slot_logits.float() / agent.temperature, dim=-1).cpu().float().numpy()

            results_bin: list[PlumbBinaryResult | None] = []
            for i, c in enumerate(claims):
                p_row = probs_tensor[i]
                raw_probs = {key: float(v) for (key, _), v in zip(options, p_row, strict=True)}
                p_v = round(float(raw_probs.get("V", 0.50)), 4)
                p_f = round(float(raw_probs.get("F", 0.50)), 4)
                verdict = "V" if p_v >= p_f else "F"
                results_bin.append(PlumbBinaryResult(
                    statement=c,
                    verdict=verdict,
                    confidence=max(p_v, p_f),
                    probabilities={"V": p_v, "F": p_f},
                    prob_true=p_v,
                    prob_false=p_f,
                ))
            return results_bin
        except Exception as exc:
            logger.warning("Falha no batch_evaluate_claim_binary (%s): %s. Executando fallback sequencial.", type(exc).__name__, exc)
            return [self.evaluate_claim_binary(c, ctx) for c, ctx in zip(claims, contexts_list)]



def parse_fused_probabilities(raw_probs: dict[str, float], statement: str) -> PlumbFusedResult:
    """Calcula probabilidades marginais de tópicos e veredito a partir da distribuição conjunta."""
    topic_probs = {
        "Saúde": float(raw_probs.get("saude_v", 0.0) + raw_probs.get("saude_f", 0.0)),
        "Política": float(raw_probs.get("politica_v", 0.0) + raw_probs.get("politica_f", 0.0)),
        "Economia": float(raw_probs.get("economia_v", 0.0) + raw_probs.get("economia_f", 0.0)),
        "Esportes": float(raw_probs.get("esportes_v", 0.0) + raw_probs.get("esportes_f", 0.0)),
        "Entretenimento": float(raw_probs.get("entretenimento", 0.0)),
        "Outros": float(raw_probs.get("outros", 0.0)),
    }
    s_top = sum(topic_probs.values()) or 1.0
    topic_probs = {k: round(v / s_top, 4) for k, v in topic_probs.items()}
    best_topic = max(topic_probs, key=topic_probs.get)
    topic_conf = topic_probs[best_topic]

    v_sum = float(
        raw_probs.get("saude_v", 0.0)
        + raw_probs.get("politica_v", 0.0)
        + raw_probs.get("economia_v", 0.0)
        + raw_probs.get("esportes_v", 0.0)
    )
    f_sum = float(
        raw_probs.get("saude_f", 0.0)
        + raw_probs.get("politica_f", 0.0)
        + raw_probs.get("economia_f", 0.0)
        + raw_probs.get("esportes_f", 0.0)
    )

    if (v_sum + f_sum) > 0.0001:
        norm_v = round(v_sum / (v_sum + f_sum), 4)
        norm_f = round(f_sum / (v_sum + f_sum), 4)
    else:
        norm_v, norm_f = 0.50, 0.50

    best_verdict = "V" if norm_v >= norm_f else "F"
    verdict_conf = max(norm_v, norm_f)
    best_raw = max(raw_probs, key=raw_probs.get) if raw_probs else "outros"

    return PlumbFusedResult(
        statement=statement,
        topic=best_topic,
        topic_confidence=topic_conf,
        verdict=best_verdict,
        verdict_confidence=verdict_conf,
        prob_v=norm_v,
        prob_f=norm_f,
        topic_probabilities=topic_probs,
        raw_probabilities={k: round(float(v), 4) for k, v in raw_probs.items()},
        raw_choice=best_raw,
    )


class MLXPlumbClassifier:
    """
    Classificador Plumb-4B de alta performance executado nativamente via Apple MLX.
    Utiliza kernels compilados em Metal e memória unificada, com latência ~200ms por inferência.
    """
    _model: Any = None
    _tokenizer: Any = None
    _load_attempted: bool = False

    def __init__(self, model_name: str = "crh225/plumb-4b") -> None:
        self.model_name = model_name

    def _ensure_loaded(self) -> bool:
        if MLXPlumbClassifier._model is not None:
            return True
        if MLXPlumbClassifier._load_attempted and MLXPlumbClassifier._model is None:
            return False

        MLXPlumbClassifier._load_attempted = True
        try:
            import mlx.core as mx
            import mlx_lm
            import mlx_lm.utils

            mlx_lm.utils.MODEL_REMAPPING["qwen3_5_text"] = "qwen3_5"
            logger.info("Carregando modelo Plumb-4B no motor Apple MLX Metal (%s)...", self.model_name)
            m, t = mlx_lm.load(self.model_name)
            MLXPlumbClassifier._model = m
            MLXPlumbClassifier._tokenizer = t
            logger.info("Modelo Plumb-4B carregado no Apple MLX com sucesso.")
            return True
        except Exception as exc:
            logger.warning("Falha ao inicializar MLXPlumbClassifier (%s): %s", type(exc).__name__, exc)
            MLXPlumbClassifier._model = None
            return False

    def _decide_choice(self, text: str, instructions: str, criteria: dict[str, str]) -> dict[str, Any] | None:
        """Executa forward pass no MLX e extrai softmax calibrado nos tokens das opções."""
        if getattr(self, "_disabled", False):
            fallback = get_plumb_classifier()
            agent = fallback._ensure_loaded()
            if agent is None:
                return None
            return agent.decide(text.strip()[:2000], {"type": "choice", "instructions": instructions, "criteria": criteria})

        if not self._ensure_loaded():
            return None

        try:
            import mlx.core as mx
            keys = list(criteria.keys())
            options_text = "\n".join(f"{chr(65+i)}: {desc}" for i, desc in enumerate(criteria.values()))
            prompt_content = f"{instructions}\n\nTexto: \"{text.strip()[:2000]}\"\n\nOpções:\n{options_text}\nResposta:"
            prompt = self._tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt_content}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )

            tokens = mx.array(self._tokenizer.encode(prompt, add_special_tokens=False))
            letters = [chr(65 + i) for i in range(len(keys))]
            slot_ids = [self._tokenizer.encode(l, add_special_tokens=False)[0] for l in letters]

            logits = self._model(tokens[None])
            slot_logits = logits[0, -1, slot_ids]
            probs = mx.softmax(slot_logits)
            mx.eval(probs)

            probs_dict = {key: float(p) for key, p in zip(keys, probs)}
            best_key = max(probs_dict, key=probs_dict.get)
            return {
                "choice": best_key,
                "confidence": probs_dict[best_key],
                "probabilities": probs_dict,
            }
        except Exception as exc:
            logger.warning("Falha durante inferência no MLX (%s): %s. Recorrendo ao JevK5.", type(exc).__name__, exc)
            self._disabled = True
            fallback = get_plumb_classifier()
            agent = fallback._ensure_loaded()
            if agent is None:
                return None
            return agent.decide(text.strip()[:2000], {"type": "choice", "instructions": instructions, "criteria": criteria})

    def classify(self, text: str) -> PlumbTopicResult | None:
        res = self._decide_choice(text, "Classify the main topic of the text into exactly one category.", CRITERIA)
        if not res:
            return None
        choice = str(res["choice"])
        raw_probs = res["probabilities"]
        conf = float(res["confidence"])
        probs_pt = {pt_k: round(float(raw_probs.get(eng_k, 0.0)), 4) for eng_k, pt_k in TOPIC_MAP.items()}
        return PlumbTopicResult(
            topic=TOPIC_MAP.get(choice, "Outros"),
            raw_topic=choice,
            confidence=round(conf, 4),
            probabilities=probs_pt,
            is_health=(choice == "health"),
            health_probability=round(float(raw_probs.get("health", 0.0)), 4),
        )

    def evaluate_claim_binary(self, claim: str, context: str | None = None) -> PlumbBinaryResult | None:
        inp = claim if not context else f"{claim}\nContexto: {context.strip()[:500]}"
        res = self._decide_choice(
            inp,
            "Evaluate the factual truthfulness of the target statement. Is it true (V) or false/fake (F)?",
            BINARY_CLAIM_CRITERIA,
        )
        if not res:
            return None
        choice = str(res["choice"]).upper()
        if choice not in ("V", "F"):
            choice = "F" if choice in ("FAKE", "FALSE", "0") else "V"
        raw_probs = res["probabilities"]
        p_v = round(float(raw_probs.get("V", 0.50)), 4)
        p_f = round(float(raw_probs.get("F", 0.50)), 4)
        return PlumbBinaryResult(
            statement=claim,
            verdict=choice,
            confidence=round(float(res["confidence"]), 4),
            probabilities={"V": p_v, "F": p_f},
            prob_true=p_v,
            prob_false=p_f,
        )

    def decide_fused(self, claim: str) -> PlumbFusedResult | None:
        res = self._decide_choice(
            claim,
            "Classify both the main subject domain and the factual truthfulness (true or false/fake) of the statement.",
            FUSED_CRITERIA,
        )
        if not res:
            return None
        return parse_fused_probabilities(res["probabilities"], claim)

    def batch_decide_fused(self, claims: list[str]) -> list[PlumbFusedResult | None]:
        """Classifica lote no MLX com decisão unificada (ou batching tensor paralelo via JevK5)."""
        if getattr(self, "_disabled", False):
            return get_plumb_classifier().batch_decide_fused(claims)
        res = []
        for c in claims:
            r = self.decide_fused(c)
            if getattr(self, "_disabled", False):
                # Se desativou no meio, processa o restante via JevK5 batching
                remaining = claims[len(res):]
                res.extend(get_plumb_classifier().batch_decide_fused(remaining))
                break
            res.append(r)
        return res

    def batch_evaluate_claim_binary(
        self,
        claims: list[str],
        contexts: list[str | None] | None = None,
    ) -> list[PlumbBinaryResult | None]:
        """Avalia lote no MLX com veredito binário."""
        if getattr(self, "_disabled", False):
            return get_plumb_classifier().batch_evaluate_claim_binary(claims, contexts)
        contexts_list = contexts or ([None] * len(claims))
        return [self.evaluate_claim_binary(c, ctx) for c, ctx in zip(claims, contexts_list)]



# Helper singletons
_default_classifier: PlumbTopicClassifier | None = None
_default_mlx_classifier: MLXPlumbClassifier | None = None


def get_plumb_classifier() -> PlumbTopicClassifier:
    """Retorna a instância singleton do classificador Plumb-4B via JevK5/PyTorch."""
    global _default_classifier
    if _default_classifier is None:
        _default_classifier = PlumbTopicClassifier()
    return _default_classifier


def get_plumb_classifier_engine(engine: str = "auto") -> Any:
    """
    Retorna o classificador Plumb-4B na engine solicitada ('mlx', 'jevk5' ou 'auto').
    No modo 'auto', tenta o Apple MLX primeiro (~200ms) e recorre ao JevK5 caso o MLX falhe.
    """
    global _default_mlx_classifier, _default_classifier
    eng = engine.lower().strip()

    if eng == "mlx":
        if _default_mlx_classifier is None:
            _default_mlx_classifier = MLXPlumbClassifier()
        if _default_mlx_classifier._ensure_loaded():
            return _default_mlx_classifier
        logger.warning("Engine MLX solicitada mas indisponível. Recorrendo ao JevK5.")
        return get_plumb_classifier()

    if eng == "auto":
        try:
            if _default_mlx_classifier is None:
                _default_mlx_classifier = MLXPlumbClassifier()
            if _default_mlx_classifier._ensure_loaded():
                return _default_mlx_classifier
        except Exception:
            pass
        return get_plumb_classifier()

    return get_plumb_classifier()

