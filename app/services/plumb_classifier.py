"""
Serviço de classificação de tópicos baseado no modelo Plumb-4B (crh225/plumb-4b).
Identifica o domínio temático principal de textos (Saúde, Política, Entretenimento,
Esportes, Economia, Outros) através de uma única passagem de inferência calibrada com JevK5.
"""
import asyncio
import logging
from typing import Any
from pydantic import BaseModel, Field

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


class PlumbTopicResult(BaseModel):
    """Resultado da classificação de tópico gerado pelo Plumb-4B."""
    topic: str = Field(..., description="Tema classificado em português: Saúde, Política, Entretenimento, Esportes, Economia, Outros")
    raw_topic: str = Field(..., description="Chave original predita pelo modelo (ex: health, politics, etc.)")
    confidence: float = Field(..., description="Confiança na predição do tópico dominante (0.0 a 1.0)")
    probabilities: dict[str, float] = Field(default_factory=dict, description="Distribuição de probabilidades normalizada por tema em português")
    is_health: bool = Field(..., description="True se o tema for classificado como Saúde")
    health_probability: float = Field(..., description="Probabilidade atribuída ao tema Saúde")


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


# Helper singleton
_default_classifier: PlumbTopicClassifier | None = None


def get_plumb_classifier() -> PlumbTopicClassifier:
    """Retorna a instância singleton do classificador Plumb-4B."""
    global _default_classifier
    if _default_classifier is None:
        _default_classifier = PlumbTopicClassifier()
    return _default_classifier
