import json
import logging
import httpx

from app.config import get_settings
from app.core.base import BaseAnalyzer
from app.core.registry import register_analyzer
from app.schemas.analysis import AnalyzerResult, Verdict

logger = logging.getLogger("factchkbr.analyzers.llm_judge")


@register_analyzer("llm_judge", weight=1.0)
class LlmJudgeAnalyzer(BaseAnalyzer):
    """
    Analisador 3: Avaliador via LLM / Modelo de NLP (Ollama local qwen3.5:9b ou OpenAI).
    """

    def __init__(self) -> None:
        self.settings = get_settings()

    async def analyze(self, text: str, urls: list[str]) -> AnalyzerResult:
        """Executa avaliação factual via LLM com formato estruturado JSON."""
        return await self.analyze_with_context(text, urls)

    async def analyze_with_context(
        self,
        text: str,
        urls: list[str],
        evidences: list[dict[str, Any]] | None = None,
        heuristic_features: dict[str, Any] | None = None,
    ) -> AnalyzerResult:
        """
        Executa avaliação factual via LLM embasada nas evidências externas recuperadas
        e nos indicadores linguísticos/sensacionalistas da mensagem original.
        """
        endpoint = self.settings.get_llm_endpoint()
        model = self.settings.get_llm_model()
        headers = self.settings.get_llm_headers()
        provider = self.settings.LLM_PROVIDER

        if provider.lower() == "openai" and not self.settings.OPENAI_API_KEY:
            return self._fallback_result(text)

        system_prompt = (
            "Você é um perito em verificação de fatos e desinformação no Brasil.\n"
            "Avalie a alegação recebida confrontando-a com as evidências de checadores e notícias coletadas.\n"
            "Retorne RIGOROSAMENTE apenas um JSON no formato:\n"
            "{\n"
            '  "verdict": "VERDADEIRO" | "FAKE" | "SUSPEITO" | "INCONCLUSIVO",\n'
            '  "confidence": 0.0 a 1.0,\n'
            '  "summary": "Resumo explicativo e conciso de 1 a 2 parágrafos",\n'
            '  "reasons": ["Motivo 1", "Motivo 2"],\n'
            '  "sources": ["Nome do veículo ou fonte checada"]\n'
            "}"
        )

        user_content = f"Alegação a ser verificada: \"{text}\"\n\n"
        if evidences:
            user_content += "Evidências e matérias encontradas por checadores e veículos confiáveis:\n"
            for ev in evidences[:5]:
                title = ev.get("title", "")
                src = ev.get("source_name", "Fonte")
                rating = ev.get("rating")
                rating_str = f" [Classificação: {rating}]" if rating else ""
                user_content += f"- {src}: \"{title}\"{rating_str}\n"
            user_content += "\n"

        if heuristic_features and heuristic_features.get("composite_sensationalism_score", 0) > 0.50:
            score = heuristic_features["composite_sensationalism_score"]
            user_content += f"Nota de alerta: O texto original possui sinais expressivos de sensacionalismo/apelo (índice: {score:.2f}).\n\n"

        user_content += "Avalie as evidências e emita o veredito final com justificativa fundamentada."

        try:
            timeout = 35.0 if provider.lower() == "ollama" else 15.0
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(
                    endpoint,
                    headers=headers,
                    json={
                        "model": model,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_content},
                        ],
                        "response_format": {"type": "json_object"},
                        "temperature": 0.1,
                    },
                )
                if resp.status_code == 200:
                    data = resp.json()
                    content = data["choices"][0]["message"]["content"]
                    parsed = json.loads(content)

                    # Mapeia veredito com validação
                    v_raw = parsed.get("verdict", "INCONCLUSIVO").upper()
                    verdict = Verdict[v_raw] if v_raw in Verdict.__members__ else Verdict.INCONCLUSIVO

                    confidence = float(parsed.get("confidence", 0.60))
                    confidence = max(0.0, min(1.0, confidence))

                    return AnalyzerResult(
                        analyzer_name="llm_judge",
                        verdict=verdict,
                        confidence=confidence,
                        claim=text[:120],
                        summary=parsed.get("summary", "Avaliação realizada via modelo de linguagem."),
                        reasons=parsed.get("reasons", ["Avaliação contextual por LLM"]),
                        sources=parsed.get("sources", [f"LLM Judge ({model})"]),
                        raw_details={"model": model, "provider": provider},
                    )
        except Exception as e:
            logger.debug("LLM Judge (%s: %s) indisponível (%s). Usando retorno padrão.", provider, model, e)

        return self._fallback_result(text)

    def _fallback_result(self, text: str) -> AnalyzerResult:
        """Resultado padrão caso o provedor LLM esteja indisponível."""
        return AnalyzerResult(
            analyzer_name="llm_judge",
            verdict=Verdict.INCONCLUSIVO,
            confidence=0.50,
            claim=text[:120],
            summary="Avaliador LLM offline ou sem conexão com o servidor local do Ollama.",
            reasons=["Servidor LLM não respondeu à requisição de julgamento."],
            sources=[f"Ollama local ({self.settings.OLLAMA_MODEL})"],
        )

