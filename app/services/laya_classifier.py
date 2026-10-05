"""
Módulo de compatibilidade retroativa para Laya, redirecionando para Plumb-4B.
"""
from app.services.plumb_classifier import (
    CRITERIA,
    TOPIC_MAP,
    PlumbTopicClassifier as LayaTopicClassifier,
    PlumbTopicResult as LayaTopicResult,
    get_plumb_classifier as get_laya_classifier,
)

__all__ = [
    "CRITERIA",
    "TOPIC_MAP",
    "LayaTopicClassifier",
    "LayaTopicResult",
    "get_laya_classifier",
]
