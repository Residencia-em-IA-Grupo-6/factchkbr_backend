"""
Serviço de classificação de tópicos baseado no modelo Laya (convaiinnovations/laya).
Identifica o domínio temático principal de textos (Saúde, Política, Entretenimento,
Esportes, Economia, Outros) através de uma única passagem de inferência não-autoregressiva.
"""
import asyncio
import logging
from typing import Any
from pydantic import BaseModel, Field

logger = logging.getLogger("factchkbr.services.laya_classifier")

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


class LayaTopicResult(BaseModel):
    """Resultado da classificação de tópico gerado pelo Laya."""
    topic: str = Field(..., description="Tema classificado em português: Saúde, Política, Entretenimento, Esportes, Economia, Outros")
    raw_topic: str = Field(..., description="Chave original predita pelo modelo (ex: health, politics, etc.)")
    confidence: float = Field(..., description="Confiança na predição do tópico dominante (0.0 a 1.0)")
    probabilities: dict[str, float] = Field(default_factory=dict, description="Distribuição de probabilidades normalizada por tema em português")
    is_health: bool = Field(..., description="True se o tema for classificado como Saúde")
    health_probability: float = Field(..., description="Probabilidade atribuída ao tema Saúde")


class LayaTopicClassifier:
    """
    Classificador temático baseado no Laya (singleton lazy-loaded).
    Utiliza o checkpoint multilíngue convaiinnovations/laya com cabeça de decisão rápida.
    """
    _agent: Any = None
    _load_attempted: bool = False

    def __init__(
        self,
        model_name: str = "convaiinnovations/laya",
        subfolder: str = "multilingual",
        preload: bool = False,
    ) -> None:
        self.model_name = model_name
        self.subfolder = subfolder
        if preload:
            self._ensure_loaded()

    def _ensure_loaded(self) -> Any:
        if LayaTopicClassifier._agent is not None:
            return LayaTopicClassifier._agent
        if LayaTopicClassifier._load_attempted and LayaTopicClassifier._agent is None:
            return None

        LayaTopicClassifier._load_attempted = True
        try:
            import laya
            logger.info("Carregando modelo Laya (%s, subfolder=%s)...", self.model_name, self.subfolder)
            LayaTopicClassifier._agent = laya.load(self.model_name, subfolder=self.subfolder)
            logger.info("Modelo Laya carregado com sucesso.")
            return LayaTopicClassifier._agent
        except Exception as exc:
            logger.warning("Não foi possível carregar o modelo Laya (%s): %s", type(exc).__name__, exc)
            LayaTopicClassifier._agent = None
            return None

    def classify(self, text: str) -> LayaTopicResult | None:
        """
        Classifica sincronamente o texto em um dos domínios temáticos.
        Retorna LayaTopicResult ou None se o modelo estiver indisponível.
        """
        agent = self._ensure_loaded()
        if agent is None:
            return None

        clean_text = text.strip()
        if not clean_text:
            return None

        truncated_text = clean_text[:2000]

        q = {
            "topic": {
                "type": "choice",
                "instructions": "Classify the main topic of the text into exactly one category.",
                "criteria": CRITERIA,
            }
        }

        try:
            res = agent.decide(truncated_text, questions=q, return_details=True)
            choice = res.values["topic"]["choice"]
            raw_probs = res.probabilities.get("topic", {})
            conf = raw_probs.get(choice, 0.0)

            probs_pt: dict[str, float] = {}
            for eng_k, pt_k in TOPIC_MAP.items():
                probs_pt[pt_k] = round(float(raw_probs.get(eng_k, 0.0)), 4)

            topic_pt = TOPIC_MAP.get(choice, "Outros")
            health_prob = round(float(raw_probs.get("health", 0.0)), 4)

            return LayaTopicResult(
                topic=topic_pt,
                raw_topic=choice,
                confidence=round(float(conf), 4),
                probabilities=probs_pt,
                is_health=(choice == "health"),
                health_probability=health_prob,
            )
        except Exception as exc:
            logger.warning("Falha na inferência do Laya (%s): %s", type(exc).__name__, exc)
            return None

    async def classify_async(self, text: str) -> LayaTopicResult | None:
        """
        Executa a classificação de forma assíncrona em threadpool (sem travar o event loop).
        """
        return await asyncio.to_thread(self.classify, text)


# Helper singleton
_default_classifier: LayaTopicClassifier | None = None


def get_laya_classifier() -> LayaTopicClassifier:
    """Retorna a instância singleton do classificador Laya."""
    global _default_classifier
    if _default_classifier is None:
        _default_classifier = LayaTopicClassifier()
    return _default_classifier
