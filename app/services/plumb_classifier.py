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


# Helper singleton
_default_classifier: PlumbTopicClassifier | None = None


def get_plumb_classifier() -> PlumbTopicClassifier:
    """Retorna a instância singleton do classificador Plumb-4B."""
    global _default_classifier
    if _default_classifier is None:
        _default_classifier = PlumbTopicClassifier()
    return _default_classifier
