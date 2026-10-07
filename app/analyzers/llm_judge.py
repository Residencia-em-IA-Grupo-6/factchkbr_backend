import json
import logging
import re
from typing import Any
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
        sub_claims: list[dict[str, Any]] | None = None,
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
            "/no_think\n"
            "Você é o redator sênior e analista editorial do FactChkBR, perito em verificação de fatos e desinformação no Brasil.\n"
            "O motor neural de decisão epistêmica (Plumb-4B) é o responsável por determinar matematicamente a veracidade de cada alegação individual.\n"
            "SUA MISSÃO EXCLUSIVA É JORNALÍSTICA E EDITORIAL: Redigir uma síntese explicativa e pedagógica ('summary'), enumerar as razões fáticas ('reasons') e elaborar justificativas claras para cada alegação em 'claims_evaluation' com base nas evidências.\n\n"
            "DIRETRIZES FUNDAMENTAIS:\n"
            "1. FIDELIDADE AOS VEREDITOS:\n"
            "   - Mantenha rigorosa fidelidade aos vereditos determinados pelo modelo de decisão para cada alegação.\n"
            "   - Não inverta nem altere os vereditos definidos.\n\n"
            "2. ÔNUS DA PROVA E ALEGAÇÕES NÃO ENCONTRADAS:\n"
            "   - Em checagem de fatos sobre alegações extraordinárias, promessas em saúde, curas ou tratamentos milagrosos, simulações de diálogos/entrevistas com figuras públicas ou anúncios comerciais, a afirmação exige sustentação fática verificável.\n"
            "   - Quando NÃO forem encontradas matérias, comunicados oficiais ou checagens confirmando o fato alegado ('não encontrado' / ausência de respaldo em fontes confiáveis), explique claramente que a afirmação configura boato sem sustentação fática ou provavelmente falso.\n"
            "   - Diálogos forjados, simulação de apresentadores ou endossos de celebridades para produtos ou métodos médicos sem registro jornalístico são formatos típicos de golpes comerciais (scams) e devem ser justificados como tal.\n\n"
            "3. CLAREZA EDITORIAL E COERÊNCIA:\n"
            "   - Certifique-se de que a explicação em 'summary', 'reasons' e 'claims_evaluation' justifique plenamente a classificação atribuída.\n"
            "   - Se a alegação recebida for desmentida ou refutada pelas fontes, o veredito é FAKE.\n"
            "   - Se for confirmada por fontes oficiais/jornalismo de referência, o veredito é VERDADEIRO.\n"
            "   - Se trouxer mistura de fatos reais com alegações sem respaldo, o veredito geral é SUSPEITO.\n\n"
            "FORMATO DE RESPOSTA:\n"
            "Retorne RIGOROSAMENTE apenas um JSON no formato:\n"
            "{\n"
            '  "verdict": "VERDADEIRO" | "FAKE" | "SUSPEITO" | "INCONCLUSIVO",\n'
            '  "confidence": 0.0 a 1.0,\n'
            '  "summary": "Resumo explicativo detalhado e conciso de 1 a 2 parágrafos",\n'
            '  "reasons": ["Motivo 1", "Motivo 2"],\n'
            '  "sources": ["Nome do veículo ou fonte checada"],\n'
            '  "claims_evaluation": [\n'
            '    {\n'
            '      "statement": "texto da alegação avaliada",\n'
            '      "verdict": "VERDADEIRO" | "FAKE" | "SUSPEITO" | "INCONCLUSIVO",\n'
            '      "confidence": 0.0 a 1.0,\n'
            '      "justification": "Explicação pontual do porquê esta alegação recebeu este veredito com base nas evidências"\n'
            '    }\n'
            '  ]\n'
            "}"
        )

        user_content = f"Alegação a ser verificada: \"{text}\"\n\n"
        has_debunk = False
        if evidences:
            user_content += "Evidências e matérias encontradas por checadores e veículos confiáveis:\n"
            for ev in evidences[:5]:
                title = ev.get("title", "")
                src = ev.get("source_name", "Fonte")
                rating = ev.get("rating")
                rating_str = f" [Classificação: {rating}]" if rating else ""
                user_content += f"- {src}: \"{title}\"{rating_str}\n"
                if rating in ("Desmentido", "Falso", "Fake", "Mentira") or any(
                    k in title.lower() for k in ("é falso", "é mentira", "desmente", "desmentiu", "boato")
                ):
                    has_debunk = True
            user_content += "\n"
        else:
            user_content += (
                "Atenção: Nenhuma evidência, notícia ou checagem foi encontrada nas buscas externas ('não encontrado'). "
                "Conforme o princípio de ônus da prova, afirmações sem respaldo em fontes confiáveis configuram boato "
                "e devem ser julgadas como FAKE (provavelmente falso).\n\n"
            )

        is_vector_cached = any(sc.get("vector_cache_hit", False) for sc in (sub_claims or []))
        if sub_claims:
            if is_vector_cached:
                user_content += "Vereditos e fundamentações consolidadas recuperadas da Base Vetorial (ChromaDB):\n"
            else:
                user_content += "Vereditos determinados pelo modelo de decisão matemática (Plumb-4B) para cada alegação:\n"
            for i, sc in enumerate(sub_claims, 1):
                s_stmt = sc.get("statement", "")
                s_v = sc.get("verdict", "")
                s_v_str = s_v.value if hasattr(s_v, "value") else str(s_v)
                s_conf = sc.get("confidence", 0.0)
                try:
                    s_conf = float(s_conf)
                    if s_conf > 1.0:
                        s_conf = s_conf / 100.0
                except (ValueError, TypeError):
                    s_conf = 0.80
                user_content += f"  [{i}] \"{s_stmt}\" ➔ Veredito: {s_v_str} (Confiança: {s_conf:.2f})\n"
                just = sc.get("justification", "")
                if is_vector_cached and just:
                    user_content += f"      Fundamentação consolidada: {just}\n"

            if is_vector_cached:
                user_content += (
                    "\n⚡ AVISO DE BASE VETORIAL (MEMÓRIA PERSISTENTE):\n"
                    "- Esta alegação já foi apurada e validada previamente no banco de dados vetorial.\n"
                    "- Redija o resumo ('summary'), as razões fáticas ('reasons') e as justificativas em 'claims_evaluation' "
                    "alinhadas à fundamentação já consolidada na checagem anterior, mantendo com fidelidade absoluta o veredito.\n"
                    "- No campo 'confidence', retorne SEMPRE um número decimal entre 0.0 e 1.0 (ex: 0.85; NUNCA use porcentagem ou valores maiores que 1.0).\n\n"
                )
            else:
                user_content += (
                    "\nSUA TAREFA EXCLUSIVA É JORNALÍSTICA E EDITORIAL:\n"
                    "- Redija o resumo ('summary'), as razões fáticas ('reasons') e as justificativas em 'claims_evaluation' "
                    "fundamentando POR QUE cada alegação recebeu esse veredito específico com base nas evidências.\n"
                    "- Mantenha RIGOROSAMENTE o veredito ('verdict') de cada alegação definido acima.\n"
                    "- No campo 'confidence', retorne SEMPRE um número decimal entre 0.0 e 1.0 (ex: 0.85; NUNCA use porcentagem ou valores maiores que 1.0).\n\n"
                )

        if heuristic_features and heuristic_features.get("composite_sensationalism_score", 0) > 0.50:
            score = heuristic_features["composite_sensationalism_score"]
            user_content += f"Nota de alerta: O texto original possui sinais expressivos de sensacionalismo/apelo (índice: {score:.2f}).\n\n"

        has_negation = bool(re.search(r"\b(?:não|nunca|jamais|tampouco|nenhum|nenhuma)\b", text, re.IGNORECASE))
        has_meta_debunk = bool(
            re.match(
                r"^(?:(?:fato ou fake|uol confere|comprova|estadao verifica|estadão verifica|aos fatos|lupa|afp checamos|afp|boatos\.org)\s*[:\-]\s*)?"
                r"(?:não é verdade que|não procede que|é falso que|é mentira que|é fake que|é boato que|boato de que|falso que|falsa que|desmentido:?|alerta:?)\s*",
                text.strip(),
                re.IGNORECASE,
            )
        )

        if has_meta_debunk:
            user_content += (
                "Atenção à meta-asserção: O texto recebido já é uma declaração de que determinada afirmação é falsa ou boato. "
                "Se as evidências e checagens mostram que se trata de boato/falsidade, a afirmação do usuário está correta e o veredito deve ser VERDADEIRO.\n\n"
            )
        elif has_debunk:
            user_content += (
                "Atenção de verificação: As fontes contêm desmentido explícito. "
                "Se as evidências refutam diretamente a afirmação feita pelo usuário, o veredito deve ser FAKE.\n\n"
            )
        if has_negation and not has_meta_debunk:
            user_content += (
                "Atenção à polaridade: A alegação recebida é uma afirmação negativa. "
                "Se as fontes mostram que a afirmação negativa é inverídica (ou seja, a ação é permitida/ocorre), o veredito deve ser FAKE.\n\n"
            )

        user_content += (
            "Avalie as evidências e emita o veredito final com justificativa fundamentada. "
            "Lembre-se: quando não forem encontrados registros jornalísticos ou oficiais confirmando a alegação "
            "('não encontrado'), o veredito deve ser FAKE (provavelmente falso / boato sem respaldo fático)."
        )

        try:
            timeout = 65.0 if provider.lower() == "ollama" else 15.0
            is_ollama = provider.lower() == "ollama"
            if is_ollama:
                payload = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content},
                    ],
                    "stream": False,
                    "think": False,
                    "format": "json",
                    "options": {
                        "temperature": 0.1,
                        "num_predict": 800,
                    },
                }
            else:
                payload = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content},
                    ],
                    "response_format": {"type": "json_object"},
                    "temperature": 0.1,
                }

            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(
                    endpoint,
                    headers=headers,
                    json=payload,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    if "message" in data:
                        content = data["message"].get("content", "")
                    elif "choices" in data and data["choices"]:
                        content = data["choices"][0].get("message", {}).get("content", "")
                    else:
                        content = ""

                    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
                    if "```" in content:
                        fence_m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", content)
                        if fence_m:
                            content = fence_m.group(1).strip()
                    start = content.find("{")
                    end = content.rfind("}")
                    if start != -1 and end != -1 and end > start:
                        content = content[start : end + 1]

                    parsed = json.loads(content)

                    # Mapeia veredito com validação
                    v_raw = parsed.get("verdict", "INCONCLUSIVO").upper()
                    verdict = Verdict[v_raw] if v_raw in Verdict.__members__ else Verdict.INCONCLUSIVO

                    confidence = float(parsed.get("confidence", 0.60))
                    confidence = max(0.0, min(1.0, confidence))

                    summary = parsed.get("summary", "Avaliação realizada via modelo de linguagem.")
                    reasons = parsed.get("reasons", ["Avaliação contextual por LLM"])

                    # Validação de coerência interna semântica (salvaguarda contra inversão de polaridade em LLMs):
                    explanation_corpus = f"{summary} {' '.join(reasons)}".lower()
                    debunk_cues = (
                        "foi desmentid", "foi refutad", "desmentiu a afirmação",
                        "desmentiu a alegação", "desmentida por", "desmentido por",
                        "afirmação é falsa", "alegação é falsa", "afirmação falsa",
                        "alegação falsa", "declaração é falsa",
                        "desmentido oficial", "trata-se de desinformação",
                        "não procede", "afirmação incorreta"
                    )
                    confirm_cues = (
                        "comprovadamente verdadeiro", "fato comprovado",
                        "afirmação é verdadeira", "alegação é verdadeira",
                        "totalmente verdadeiro", "confirmada categoricamente",
                        "comprova a alegação", "confirma a alegação"
                    )

                    polarity_corrected = False
                    if not has_meta_debunk and verdict == Verdict.VERDADEIRO and any(cue in explanation_corpus for cue in debunk_cues):
                        logger.warning(
                            "Inversão de polaridade detectada no LLM Judge: explicação indica desmentido, "
                            "mas veredito emitido foi VERDADEIRO. Corrigindo veredito para FAKE."
                        )
                        verdict = Verdict.FAKE
                        polarity_corrected = True

                    elif (
                        has_meta_debunk
                        and verdict == Verdict.FAKE
                        and any(cue in explanation_corpus for cue in ("foi desmentid", "desmentiu", "desmentido", "trata-se de boato", "é boato", "é falso que"))
                        and not any(cue in explanation_corpus for cue in ("afirmação do usuário é falsa", "declaração do autor é falsa", "alegação do usuário é falsa"))
                    ):
                        logger.warning(
                            "Inversão de polaridade em meta-asserção detectada no LLM Judge: explicação indica desmentido do boato, "
                            "mas veredito emitido foi FAKE. Corrigindo veredito para VERDADEIRO."
                        )
                        verdict = Verdict.VERDADEIRO
                        polarity_corrected = True

                    elif (
                        verdict == Verdict.FAKE
                        and any(cue in explanation_corpus for cue in confirm_cues)
                        and not any(cue in explanation_corpus for cue in debunk_cues)
                    ):
                        logger.warning(
                            "Inversão de polaridade detectada no LLM Judge: explicação indica confirmação, "
                            "mas veredito emitido foi FAKE. Corrigindo veredito para VERDADEIRO."
                        )
                        verdict = Verdict.VERDADEIRO
                        polarity_corrected = True

                    # Processamento das sub-alegações avaliadas
                    parsed_sub_claims = []
                    raw_eval = parsed.get("claims_evaluation")
                    if isinstance(raw_eval, list) and raw_eval:
                        for item in raw_eval:
                            if isinstance(item, dict):
                                sub_v_raw = str(item.get("verdict", "INCONCLUSIVO")).upper()
                                sub_v = Verdict[sub_v_raw] if sub_v_raw in Verdict.__members__ else Verdict.INCONCLUSIVO
                                sub_stmt = item.get("statement", "")
                                sub_just = item.get("justification", "")

                                # Salvaguarda de polaridade em meta-asserções na avaliação da sub-alegação
                                sub_is_meta = bool(
                                    re.match(
                                        r"^(?:(?:fato ou fake|uol confere|comprova|estadao verifica|estadão verifica|aos fatos|lupa|afp checamos|afp|boatos\.org)\s*[:\-]\s*)?"
                                        r"(?:não é verdade que|não procede que|é falso que|é mentira que|é fake que|é boato que|boato de que|falso que|falsa que|desmentido:?|alerta:?)\s*",
                                        sub_stmt.strip(),
                                        re.IGNORECASE,
                                    )
                                )
                                if sub_is_meta and sub_v == Verdict.FAKE and any(
                                    cue in sub_just.lower()
                                    for cue in ("desmentid", "desmentiu", "classificaram a afirmação como falso", "classificou a afirmação como falso", "é falso", "boato")
                                ):
                                    sub_v = Verdict.VERDADEIRO

                                raw_sub_conf = item.get("confidence", confidence)
                                try:
                                    sub_conf = float(raw_sub_conf)
                                    if sub_conf > 1.0:
                                        sub_conf = sub_conf / 100.0
                                    sub_conf = max(0.0, min(1.0, sub_conf))
                                except (ValueError, TypeError):
                                    sub_conf = confidence

                                parsed_sub_claims.append({
                                    "statement": sub_stmt,
                                    "verdict": sub_v,
                                    "confidence": sub_conf,
                                    "justification": sub_just,
                                })

                        if polarity_corrected and len(parsed_sub_claims) == 1:
                            parsed_sub_claims[0]["verdict"] = verdict
                        elif sub_claims:
                            # Preserva os vereditos objetivos determinados pelo classificador de decisão
                            for idx, sc in enumerate(sub_claims):
                                if idx < len(parsed_sub_claims) and sc.get("verdict"):
                                    parsed_sub_claims[idx]["verdict"] = sc["verdict"]
                                    if sc.get("confidence") is not None:
                                        try:
                                            sc_c = float(sc["confidence"])
                                            if sc_c > 1.0:
                                                sc_c = sc_c / 100.0
                                            parsed_sub_claims[idx]["confidence"] = max(0.0, min(1.0, sc_c))
                                        except (ValueError, TypeError):
                                            pass
                    elif sub_claims:
                        # Fallback se o modelo não gerou o array claims_evaluation
                        for sc in sub_claims:
                            parsed_sub_claims.append({
                                "statement": sc.get("statement", ""),
                                "verdict": sc.get("verdict", verdict),
                                "confidence": sc.get("confidence", confidence),
                                "justification": sc.get("justification", summary),
                            })

                    return AnalyzerResult(
                        analyzer_name="llm_judge",
                        verdict=verdict,
                        confidence=confidence,
                        claim=text.strip(),
                        summary=summary,
                        reasons=reasons,
                        sources=parsed.get("sources", [f"LLM Judge ({model})"]),
                        raw_details={
                            "model": model,
                            "provider": provider,
                            "polarity_corrected": polarity_corrected,
                            "sub_claims": parsed_sub_claims,
                        },
                    )
        except Exception as e:
            logger.debug("LLM Judge (%s: %s) indisponível (%s). Usando retorno padrão.", provider, model, e)

        return self._fallback_result(text, sub_claims=sub_claims)

    def _fallback_result(self, text: str, sub_claims: list[dict[str, Any]] | None = None) -> AnalyzerResult:
        """Resultado padrão caso o provedor LLM esteja indisponível."""
        if sub_claims:
            from app.analyzers.fact_check_api import FactCheckApiAnalyzer
            fc = FactCheckApiAnalyzer()
            verdict, confidence, agg_reasons = fc.aggregate_sub_verdicts(sub_claims)
            summary_parts = []
            for sc in sub_claims:
                s = sc.get("statement", "")
                v = sc.get("verdict", "")
                v_str = v.value if hasattr(v, "value") else str(v)
                j = sc.get("justification", "")
                summary_parts.append(f"• \"{s}\": {v_str} ({j})")
            summary = (
                f"Avaliação fundamentada pelo classificador de decisão Plumb-4B: "
                f"Resultado consolidado '{verdict.value}' com base em {len(sub_claims)} alegação(ões) checada(s)."
            )
            return AnalyzerResult(
                analyzer_name="llm_judge",
                verdict=verdict,
                confidence=confidence,
                claim=text.strip(),
                summary=summary,
                reasons=agg_reasons,
                sources=["Classificador de Decisão Plumb-4B", "Fontes e Checadores Oficiais"],
                raw_details={
                    "model": self.settings.get_llm_model(),
                    "provider": self.settings.LLM_PROVIDER,
                    "engine": "plumb_subclaims",
                    "sub_claims": sub_claims,
                },
            )

        return AnalyzerResult(
            analyzer_name="llm_judge",
            verdict=Verdict.INCONCLUSIVO,
            confidence=0.50,
            claim=text.strip(),
            summary="Avaliador LLM offline ou sem conexão com o servidor local do Ollama.",
            reasons=["Servidor LLM não respondeu à requisição de julgamento."],
            sources=[f"Ollama local ({self.settings.OLLAMA_MODEL})"],
            raw_details={
                "model": self.settings.get_llm_model(),
                "provider": self.settings.LLM_PROVIDER,
                "sub_claims": [],
            },
        )

