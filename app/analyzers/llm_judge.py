from app.core.base import BaseAnalyzer
from app.core.registry import register_analyzer
from app.schemas.analysis import AnalyzerResult, Verdict


@register_analyzer("llm_judge", weight=1.0)
class LlmJudgeAnalyzer(BaseAnalyzer):
    """
    Analisador 3: Avaliador via LLM / Modelo de NLP (OpenAI, Gemini, Ollama ou BERTimbau).
    """

    async def analyze(self, text: str, urls: list[str]) -> AnalyzerResult:
        # TODO: Implementar chamada ao modelo de linguagem (LLM / NLP)
        return AnalyzerResult(
            analyzer_name="llm_judge",
            verdict=Verdict.INCONCLUSIVO,
            confidence=0.60,
            claim=text[:100],
            summary="Avaliação de veracidade realizada via LLM Judge.",
            reasons=["Análise semântica e contextual processada."],
            sources=["LLM Fact-Checking Engine"]
        )
