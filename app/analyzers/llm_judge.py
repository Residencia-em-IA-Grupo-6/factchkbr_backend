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
            "Você é um perito sênior em verificação de fatos e desinformação no Brasil, seguindo as diretrizes metodológicas do IFCN (International Fact-Checking Network) e das principais agências de checagem brasileiras (Lupa, Aos Fatos, Fato ou Fake).\n"
            "Sua tarefa é avaliar criticamente a alegação confrontando-a com as evidências recuperadas.\n\n"
            "DIRETRIZES EPISTEMOLÓGICAS FUNDAMENTAIS:\n"
            "1. ÔNUS DA PROVA E ALEGAÇÕES NÃO ENCONTRADAS (NÃO ENCONTRADO ➔ PROVAVELMENTE FALSO / FAKE):\n"
            "   - Em checagem de fatos sobre alegações extraordinárias, promessas em saúde, curas ou tratamentos milagrosos, simulações de diálogos/entrevistas com figuras públicas (ex: William Bonner, Drauzio Varella, Vera Fischer) ou anúncios comerciais, a afirmação exige sustentação fática verificável.\n"
            "   - Quando NÃO forem encontradas matérias, comunicados oficiais ou checagens confirmando o fato alegado ('não encontrado' / ausência de respaldo em fontes confiáveis), a alegação DEVE ser julgada como 'FAKE' (classificada como boato sem sustentação factual ou provavelmente falso), com confiança em torno de 0.75 a 0.85.\n"
            "   - Explique claramente no resumo e razões: 'Não foram encontrados registros oficiais ou jornalísticos confirmando a alegação. Afirmações sem respaldo em fontes confiáveis configuram boato / informação provavelmente falsa.'\n"
            "   - Diálogos forjados, simulação de apresentadores ou endossos de celebridades para produtos ou métodos médicos sem registro jornalístico são formatos típicos de golpes comerciais (scams) e DEVEM ser classificados como 'FAKE'.\n\n"
            "2. ATENÇÃO SOBRE O CAMPO 'verdict' (VERACIDADE DA ALEGAÇÃO):\n"
            "   - O campo 'verdict' refere-se ESTRITAMENTE à veracidade da ALEGAÇÃO RECEBIDA (e NÃO à veracidade da notícia de desmentido).\n"
            "   - Se a alegação recebida for desmentida ou refutada pelas fontes (ex: 'vacinas causam autismo'), o veredito DEVE ser 'FAKE' (e NUNCA 'VERDADEIRO') com confiança de 0.90 a 1.0.\n"
            "   - Se a alegação for comprovada como verdadeira pelas fontes (ex: notas oficiais do Ministério da Saúde, confirmação de falecimento, portarias da Anvisa), o veredito é 'VERDADEIRO'.\n"
            "   - Se a alegação contiver promessas, citações, produtos ou fatos NÃO ENCONTRADOS em fontes de referência, o veredito é 'FAKE' (provavelmente falso / boato sem respaldo fático).\n"
            "   - Se a alegação trouxer mistura de fatos reais com afirmações falsas/sem respaldo, o veredito geral é 'SUSPEITO'.\n"
            "   - O veredito 'INCONCLUSIVO' deve ser reservado EXCLUSIVAMENTE para acontecimentos em andamento com cobertura jornalística real de apuração/investigação sem conclusão definitiva.\n\n"
            "3. CUIDADO CRÍTICO COM ALEGAÇÕES NEGATIVAS, DUPLA NEGAÇÃO E META-ASSERÇÕES DE DESMENTIDO:\n"
            "   - Preste atenção extrema quando a alegação contiver negação ou desmentido em si (ex: 'não', 'não pode', 'é falso que...', 'não é verdade que...').\n"
            "   - META-ASSERÇÕES DE DESMENTIDO: Se o texto recebido já afirma que um boato é falso (ex: 'É falso que o voto não vale no INSS', 'É mentira que vacinas causam autismo'), e as checagens/fontes confirmam que o boato é realmente falso, o veredito para a alegação recebida é VERDADEIRO (pois o autor está correto ao afirmar que o boato é falso).\n"
            "   - Se a alegação recebida propaga o boato como se fosse verdade (ex: 'Voto não pode ser usado no INSS', 'Vacinas causam autismo'), e as fontes desmentem o boato, aí sim o veredito da alegação é FAKE.\n"
            "   - Se as fontes confirmam o fato que o autor disse ser falso (ex: autor diz 'É falso que o Brasil ganhou a Copa de 2002', mas o Brasil de fato ganhou), o veredito é FAKE.\n"
            "   - Nunca confunda 'a notícia de checagem é verdadeira' com 'a alegação recebida é verdadeira'. Certifique-se de julgar se o que o autor AFIRMOU no texto corresponde aos fatos.\n"
            "   - COERÊNCIA OBRIGATÓRIA: Se no seu próprio resumo ou razões você afirmar que a tese do autor 'foi desmentida', 'foi refutada', 'é falsa', 'incorreta' ou 'boato sem respaldo', o veredito OBRIGATORIAMENTE deve ser 'FAKE', NUNCA 'VERDADEIRO'.\n\n"
            "4. AVALIAÇÃO DISCRIMINADA DE SUB-ALEGAÇÕES (claims_evaluation):\n"
            "   - Se forem fornecidas múltiplas alegações atômicas, avalie CADA UMA isoladamente no campo 'claims_evaluation'.\n"
            "   - Justifique pontualmente por que cada alegação é verdadeira, falsa ou provavelmente falsa (sem respaldo) com base nas evidências.\n"
            "   - O veredito geral ('verdict') deve refletir a combinação: se contiver alegações falsas e verdadeiras no mesmo texto, o veredito geral DEVE ser 'SUSPEITO'. Se todas forem falsas/sem respaldo, 'FAKE'. Se todas forem verdadeiras, 'VERDADEIRO'.\n\n"
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
            '      "justification": "Explicação pontual do porquê esta alegação é verdadeira, falsa ou provavelmente falsa por ausência de dados"\n'
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

        if sub_claims:
            user_content += "Alegações individuais isoladas para checagem discriminada:\n"
            for i, sc in enumerate(sub_claims, 1):
                s_stmt = sc.get("statement", "")
                s_v = sc.get("verdict", "")
                s_v_str = s_v.value if hasattr(s_v, "value") else str(s_v)
                user_content += f"  [{i}] \"{s_stmt}\" (Varredura preliminar: {s_v_str})\n"
            user_content += "\nPreencha obrigatoriamente o campo 'claims_evaluation' discriminando e justificando cada uma dessas alegações.\n\n"

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

                                parsed_sub_claims.append({
                                    "statement": sub_stmt,
                                    "verdict": sub_v,
                                    "confidence": float(item.get("confidence", confidence)),
                                    "justification": sub_just,
                                })

                        if polarity_corrected and len(parsed_sub_claims) == 1:
                            parsed_sub_claims[0]["verdict"] = verdict
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
                "sub_claims": sub_claims or [],
            },
        )

