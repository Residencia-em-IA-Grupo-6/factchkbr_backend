import asyncio
import logging
import re
import urllib.parse
import xml.etree.ElementTree as ET
from typing import Any
from pydantic import BaseModel, Field

import httpx

from app.config import get_settings
from app.core.base import BaseAnalyzer
from app.core.registry import register_analyzer
from app.schemas.analysis import AnalyzerResult, Verdict

logger = logging.getLogger("factchkbr.analyzers.fact_check_api")


class EvidenceItem(BaseModel):
    """Evidência factual recuperada via Fact Check Tools ou Leitura Horizontal."""
    title: str = Field(..., description="Título da matéria, comunicado ou desmentido")
    source_name: str = Field(..., description="Nome do veículo de imprensa ou autoridade (ex: G1, Lupa, WHO, Anvisa)")
    url: str = Field(..., description="Link da evidência")
    snippet: str = Field(default="", description="Trecho ou resumo descritivo")
    rating: str | None = Field(default=None, description="Classificação do checador (ex: Falso, Enganoso, Verdadeiro)")
    is_fact_check: bool = Field(default=False, description="Indica se é de agência de checagem oficial")
    published_date: str | None = Field(default=None, description="Data de publicação")


# Veículos de imprensa e instituições de autoridade reconhecidos para leitura horizontal
TRUSTED_MEDIA_DOMAINS = {
    # Mídia de Referência Nacional e Internacional
    "g1": "G1 / Fato ou Fake",
    "globo": "O Globo",
    "folha": "Folha de S.Paulo",
    "estadao": "Estadão Verifica",
    "uol": "UOL / Confere",
    "bbc": "BBC News Brasil",
    "cnnbrasil": "CNN Brasil",
    "reuters": "Reuters Brasil",
    "agenciabrasil": "Agência Brasil (EBC)",
    "valor": "Valor Econômico",
    "nexojornal": "Nexo Jornal",
    # Agências Especializadas de Checagem (IFCN)
    "lupa": "Agência Lupa",
    "aosfatos": "Aos Fatos",
    "boatos": "Boatos.org",
    "comprova": "Projeto Comprova",
    "afp": "AFP Checamos",
    # Autoridades Sanitárias, Científicas, Estatísticas e Oficiais
    "who": "Organização Mundial da Saúde (WHO/OMS)",
    "paho": "Organização Pan-Americana da Saúde (OPAS)",
    "webmd": "WebMD Health",
    "fiocruz": "Fiocruz",
    "anvisa": "Anvisa",
    "ibge": "IBGE (Instituto Brasileiro de Geografia e Estatística)",
    "ipea": "Ipea (Instituto de Pesquisa Econômica Aplicada)",
    "saude.gov": "Ministério da Saúde",
    "gov.br": "Portal Gov.br / Órgão Oficial",
}

# Palavras-chave indicativas de desmentido em títulos de leitura horizontal
DEBUNK_TITLE_PATTERNS = re.compile(
    r"\b(?:"
    r"é falso|é mentira|é fake|é boato|não é verdade|desmente|desmentiu|nega|negou|"
    r"boato|fake news|falso|falsa|engana|enganosa|distorce|distorcida|"
    r"não causou|não causa|não causam|não mata|não matam|"
    r"não t[eê]m? relação|não há relação|não provocam?|golpe|falso que"
    r")\b",
    re.IGNORECASE,
)

# Palavras-chave indicativas de confirmação factual em títulos de notícias
CONFIRM_TITLE_PATTERNS = re.compile(
    r"\b(?:confirma|confirmou|aprova|aprovou|autoriza|autorizou|determina|proíbe|proibiu|"
    r"suspende|suspendeu|recolhe|recolhimento|anuncia|anunciou|sanciona|sancionou)\b",
    re.IGNORECASE,
)


@register_analyzer("fact_check_api", weight=1.5)
class FactCheckApiAnalyzer(BaseAnalyzer):
    """
    Analisador 2: Evidências Externas de Fact-Checking & Leitura Horizontal.
    - Fonte Primária: Google Fact Check Tools API (checagens oficiais IFCN).
    - Leitura Horizontal: Varredura em veículos de referência (G1, Folha, Estadão, BBC)
      e autoridades científicas e sanitárias (WHO, WebMD, Fiocruz, Anvisa).
    """

    def __init__(self, http_client: httpx.AsyncClient | None = None) -> None:
        super().__init__()
        self.settings = get_settings()
        self._owned_client = http_client is None
        self.http_client = http_client or httpx.AsyncClient(
            timeout=12.0,
            headers={"User-Agent": "FactChkBR/1.0 (Research & Automated Verification Pipeline)"},
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=40),
        )

    async def aclose(self) -> None:
        """Encerra conexões HTTP caso gerenciadas internamente."""
        if self._owned_client:
            await self.http_client.aclose()

    async def search_google_fact_check(self, query: str) -> list[EvidenceItem]:
        """
        Consulta a Google Fact Check Tools API para localizar checagens prévias
        realizadas por agências certificadas (Lupa, Aos Fatos, Boatos.org, etc.).
        """
        api_key = self.settings.GOOGLE_FACTCHECK_API_KEY
        if not api_key:
            logger.debug("GOOGLE_FACTCHECK_API_KEY não configurada. Prosseguindo para leitura horizontal.")
            return []

        encoded_query = urllib.parse.quote(query[:180])
        url = (
            f"https://factchecktools.googleapis.com/v1alpha1/claims:search"
            f"?query={encoded_query}&languageCode=pt-BR&key={api_key}"
        )

        try:
            resp = await self.http_client.get(url)
            if resp.status_code == 200:
                data = resp.json()
                claims_data = data.get("claims", [])
                evidences: list[EvidenceItem] = []

                for c in claims_data:
                    claim_text = c.get("text", "")
                    reviews = c.get("claimReview", [])
                    for rev in reviews:
                        publisher = rev.get("publisher", {}).get("name", "Checador Independente")
                        title = rev.get("title") or claim_text
                        rating = rev.get("textualRating", "")
                        review_url = rev.get("url", "")
                        review_date = rev.get("reviewDate")

                        evidences.append(
                            EvidenceItem(
                                title=title,
                                source_name=f"{publisher} (Fact-Check)",
                                url=review_url,
                                snippet=f"Alegação revisada: \"{claim_text}\" | Classificação: {rating}",
                                rating=rating,
                                is_fact_check=True,
                                published_date=review_date,
                            )
                        )
                return evidences
            else:
                logger.debug("Google Fact Check API retornou status %s: %s", resp.status_code, resp.text[:200])
        except Exception as e:
            logger.warning("Falha na chamada à Google Fact Check API: %s", e)

        return []

    async def search_lateral_reading(self, query: str) -> list[EvidenceItem]:
        """
        Executa Leitura Horizontal em fontes confiáveis (G1, Folha, Estadão, BBC, WHO, Fiocruz, Anvisa, IBGE, Ipea)
        utilizando busca agregada para obter matérias de apuração e dados oficiais em tempo real.
        """
        clean_q = re.sub(r"[\"']", "", query)
        encoded_query = urllib.parse.quote(clean_q[:160])
        rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=pt-BR&gl=BR&ceid=BR:pt-419"

        tasks = [self._fetch_rss(rss_url, max_items=8)]

        # Se contiver termos estatísticos ou econômicos, realiza busca complementar direcionada ao IBGE e Ipea
        is_stat = bool(
            re.search(
                r"\b(?:pib|inflação|ipca|desemprego|saúde|educação|gastos?|orçamento|taxa|censo|população|salário)\b",
                clean_q,
                re.IGNORECASE,
            )
        )
        if is_stat:
            stat_query = f"(site:ibge.gov.br OR site:ipea.gov.br) {clean_q[:80]}"
            stat_url = f"https://news.google.com/rss/search?q={urllib.parse.quote(stat_query)}&hl=pt-BR&gl=BR&ceid=BR:pt-419"
            tasks.append(self._fetch_rss(stat_url, max_items=4, force_official=True))

        results = await asyncio.gather(*tasks, return_exceptions=True)
        evidences: list[EvidenceItem] = []
        for res in results:
            if isinstance(res, list):
                evidences.extend(res)

        return evidences

    async def _fetch_rss(self, url: str, max_items: int = 8, force_official: bool = False) -> list[EvidenceItem]:
        """Recupera e processa itens de um feed RSS de notícias."""
        evidences: list[EvidenceItem] = []
        try:
            resp = await self.http_client.get(url)
            if resp.status_code == 200:
                root = ET.fromstring(resp.text)
                items = root.findall(".//item")

                for it in items[:max_items]:
                    title_elem = it.find("title")
                    link_elem = it.find("link")
                    source_elem = it.find("source")
                    pub_elem = it.find("pubDate")

                    title = title_elem.text if title_elem is not None else ""
                    link = link_elem.text if link_elem is not None else ""
                    source_name = source_elem.text if source_elem is not None else "Imprensa"
                    pub_date = pub_elem.text if pub_elem is not None else None

                    if force_official:
                        if "ibge.gov.br" in link.lower() or "ibge" in source_name.lower():
                            source_name = "IBGE (Dados Oficiais)"
                        elif "ipea.gov.br" in link.lower() or "ipea" in source_name.lower():
                            source_name = "Ipea (Dados Oficiais)"

                    is_trusted = any(dom in source_name.lower() or dom in link.lower() for dom in TRUSTED_MEDIA_DOMAINS)
                    is_fact_check = (
                        any(fc in title.lower() for fc in ("fato ou fake", "verifica", "confere", "comprova", "checagem"))
                        or any(fc in source_name.lower() for fc in ("lupa", "aos fatos", "boatos"))
                    )

                    evidences.append(
                        EvidenceItem(
                            title=title,
                            source_name=source_name,
                            url=link,
                            snippet=title,
                            rating="Desmentido" if is_fact_check and DEBUNK_TITLE_PATTERNS.search(title) else None,
                            is_fact_check=is_fact_check,
                            published_date=pub_date,
                        )
                    )
        except Exception as e:
            logger.debug("Falha na leitura horizontal de notícias (%s): %s", url, e)

        return evidences

    def evaluate_verdict(
        self,
        evidences: list[EvidenceItem],
        claim: str = "",
    ) -> tuple[Verdict, float, list[str]]:
        """
        Calcula o veredito e nível de confiança a partir das evidências consolidadas.
        Calibrado para evitar falsos positivos de confirmação em alegações numéricas/estatísticas.
        """
        if not evidences:
            return (
                Verdict.INCONCLUSIVO,
                0.50,
                [
                    "Nenhuma checagem prévia ou matéria em veículos de referência foi encontrada para este fato.",
                    "Imprecisão por falta de informações: ausência de registros jornalísticos ou oficiais (pode se tratar de acontecimento muito recente ou rumor sem cobertura comprovada).",
                ],
            )

        # 1. Se houver checagem direta do Google Fact Check Tools (IFCN)
        fact_checks = [e for e in evidences if e.is_fact_check and e.rating]
        if fact_checks:
            ratings_text = " ".join(f.rating.lower() for f in fact_checks if f.rating)
            if any(k in ratings_text for k in ("falso", "fake", "mentira", "adulterado", "falsa", "incorreto", "desmentido")):
                reasons = [
                    f"Desmentido por checador oficial ({f.source_name}): classificação '{f.rating}' para a alegação."
                    for f in fact_checks[:2]
                ]
                return Verdict.FAKE, 0.95, reasons
            elif any(k in ratings_text for k in ("verdadeiro", "fato", "verdade", "correto", "comprovado")):
                reasons = [
                    f"Comprovado por checador oficial ({f.source_name}): classificação '{f.rating}'."
                    for f in fact_checks[:2]
                ]
                return Verdict.VERDADEIRO, 0.92, reasons
            elif any(k in ratings_text for k in ("enganoso", "distorcido", "fora de contexto", "impreciso", "exagerado")):
                reasons = [
                    f"Classificado como enganoso/fora de contexto por {f.source_name}: '{f.rating}'."
                    for f in fact_checks[:2]
                ]
                return Verdict.SUSPEITO, 0.85, reasons

        # 2. Leitura Horizontal em Veículos de Referência e Órgãos Oficiais
        is_quantitative = bool(
            re.search(r"\b(?:\d+[%,\.]?\d*|por\s*cento|pib|taxa|índice|gasto)\b", claim, re.IGNORECASE)
        )

        debunk_matching = [e for e in evidences if DEBUNK_TITLE_PATTERNS.search(e.title)]

        confirm_matching: list[EvidenceItem] = []
        for e in evidences:
            if CONFIRM_TITLE_PATTERNS.search(e.title):
                if is_quantitative:
                    # Em alegações quantitativas/numéricas, evita falsos positivos de verbos genéricos (ex.: 'aprova orçamento')
                    # Exige que seja de checador oficial ou que haja sobreposição temática específica dos termos substantivos
                    claim_words = [
                        w.lower() for w in re.findall(r"\b\w{4,}\b", claim)
                        if w.lower() not in ("aproximadamente", "brasil", "sobre", "entre", "quando", "foram", "disse")
                    ]
                    title_lower = e.title.lower()
                    overlap = sum(1 for w in claim_words if w in title_lower)
                    if e.is_fact_check or overlap >= 2:
                        confirm_matching.append(e)
                else:
                    confirm_matching.append(e)

        if debunk_matching:
            reasons = [
                f"Leitura horizontal ({m.source_name}): aponta desmentido ou contestação na matéria \"{m.title}\"."
                for m in debunk_matching[:2]
            ]
            confidence = 0.90 if len(debunk_matching) >= 2 else 0.80
            return Verdict.FAKE, confidence, reasons

        if confirm_matching:
            reasons = [
                f"Leitura horizontal ({m.source_name}): confirmação de atos ou ocorrência em \"{m.title}\"."
                for m in confirm_matching[:2]
            ]
            confidence = 0.88 if len(confirm_matching) >= 2 else 0.78
            return Verdict.VERDADEIRO, confidence, reasons

        # 3. Caso haja matérias encontradas mas sem sinal claro de confirmação ou desmentido
        top_titles = [f"\"{e.title}\" ({e.source_name})" for e in evidences[:2]]
        return (
            Verdict.INCONCLUSIVO,
            0.55,
            [
                f"Matérias encontradas em fontes de referência, porém sem termo explícito de desmentido ou confirmação direta: {', '.join(top_titles)}.",
                "Imprecisão por dados insuficientes: os registros tratam do assunto de forma genérica, sem comprovar nem desmentir categoricamente os pontos específicos da alegação.",
            ],
        )

    async def check_single_claim(self, claim_text: str) -> dict[str, Any]:
        """
        Executa a checagem isolada para uma única proposição factual:
        1. Consulta Google Fact Check Tools API
        2. Executa Leitura Horizontal
        3. Avalia veredito, confiança e fundamentação específica
        """
        clean_text = claim_text.strip()
        fc_task = self.search_google_fact_check(clean_text)
        lat_task = self.search_lateral_reading(clean_text)

        results = await asyncio.gather(fc_task, lat_task, return_exceptions=True)
        google_evidences = results[0] if isinstance(results[0], list) else []
        lateral_evidences = results[1] if isinstance(results[1], list) else []
        all_evidences = google_evidences + lateral_evidences

        verdict, confidence, reasons = self.evaluate_verdict(all_evidences, claim=clean_text)

        sources = []
        for e in all_evidences[:3]:
            sources.append(f"{e.source_name}: {e.title}")

        if verdict == Verdict.FAKE:
            justification = reasons[0] if reasons else "Desmentida por fontes e agências de checagem."
        elif verdict == Verdict.VERDADEIRO:
            justification = reasons[0] if reasons else "Confirmada por registros jornalísticos e fontes oficiais."
        elif verdict == Verdict.SUSPEITO:
            justification = reasons[0] if reasons else "Alegação distorcida, imprecisa ou fora de contexto."
        else:
            justification = "Ausência de referências comprobatórias ou de desmentido (fato recente ou escassez de dados)."

        return {
            "statement": clean_text,
            "verdict": verdict,
            "confidence": confidence,
            "justification": justification,
            "reasons": reasons,
            "sources": sources,
            "evidences": [e.model_dump() for e in all_evidences[:4]],
        }

    def aggregate_sub_verdicts(self, sub_results: list[dict[str, Any]]) -> tuple[Verdict, float, list[str]]:
        """
        Calcula o impacto composto das sub-alegações no score e no veredito geral:
        - Misto (FAKE + VERDADEIRO): SUSPEITO (desinformação mista/engano)
        - Todas FAKE: FAKE
        - Todas VERDADEIRO: VERDADEIRO
        - FAKE + INCONCLUSIVO: FAKE
        - VERDADEIRO + INCONCLUSIVO: VERDADEIRO (ou SUSPEITO se houver distorção)
        - Todas INCONCLUSIVO: INCONCLUSIVO
        """
        if not sub_results:
            return Verdict.INCONCLUSIVO, 0.50, ["Nenhuma alegação checável foi fornecida."]

        if len(sub_results) == 1:
            r = sub_results[0]
            return r["verdict"], r["confidence"], r["reasons"]

        fake_claims = [r for r in sub_results if r["verdict"] == Verdict.FAKE]
        true_claims = [r for r in sub_results if r["verdict"] == Verdict.VERDADEIRO]
        suspect_claims = [r for r in sub_results if r["verdict"] == Verdict.SUSPEITO]
        inconclusive_claims = [r for r in sub_results if r["verdict"] == Verdict.INCONCLUSIVO]

        itemized_reasons: list[str] = []
        for idx, r in enumerate(sub_results, 1):
            stmt = r["statement"]
            v = r["verdict"]
            v_str = v.value if hasattr(v, "value") else str(v)
            just = r.get("justification", "")
            itemized_reasons.append(f"[Alegação {idx} - {v_str}]: \"{stmt}\" ➔ {just}")

        # 1. Se contiver alegações falsas e verdadeiras no mesmo conteúdo -> SUSPEITO
        if fake_claims and true_claims:
            verdict = Verdict.SUSPEITO
            avg_conf = sum(r["confidence"] for r in fake_claims + true_claims) / len(fake_claims + true_claims)
            confidence = round(avg_conf, 2)
            summary_reason = (
                f"Conteúdo misto detectado: {len(fake_claims)} alegação(ões) falsa(s) e "
                f"{len(true_claims)} verdadeira(s) identificadas no mesmo texto."
            )
            return verdict, confidence, [summary_reason] + itemized_reasons

        # 2. Se contiver apenas alegações falsas (ou falsas + inconclusivas) -> FAKE
        if fake_claims:
            verdict = Verdict.FAKE
            confidence = round(max(r["confidence"] for r in fake_claims), 2)
            summary_reason = f"Falsidade factual: {len(fake_claims)} alegação(ões) desmentida(s) pelas fontes oficiais/checadores."
            return verdict, confidence, [summary_reason] + itemized_reasons

        # 3. Se contiver alegações suspeitas -> SUSPEITO
        if suspect_claims:
            verdict = Verdict.SUSPEITO
            confidence = round(sum(r["confidence"] for r in suspect_claims) / len(suspect_claims), 2)
            summary_reason = "Alegações classificadas como distorcidas ou fora de contexto pelas fontes."
            return verdict, confidence, [summary_reason] + itemized_reasons

        # 4. Se contiver apenas verdadeiras -> VERDADEIRO
        if true_claims:
            if inconclusive_claims:
                verdict = Verdict.VERDADEIRO
                confidence = round(sum(r["confidence"] for r in true_claims) / len(true_claims) * 0.9, 2)
                summary_reason = f"{len(true_claims)} alegação(ões) confirmada(s) por fontes oficiais, com partes sem cobertura conclusiva."
            else:
                verdict = Verdict.VERDADEIRO
                confidence = round(sum(r["confidence"] for r in true_claims) / len(true_claims), 2)
                summary_reason = f"Todas as alegações ({len(true_claims)}) foram confirmadas por fontes oficiais e órgãos de referência."
            return verdict, confidence, [summary_reason] + itemized_reasons

        # 5. Todas inconclusivas -> INCONCLUSIVO
        verdict = Verdict.INCONCLUSIVO
        confidence = 0.55
        summary_reason = "Nenhuma das alegações possui referências conclusivas suficientes para confirmação ou desmentido."
        return verdict, confidence, [summary_reason] + itemized_reasons

    async def analyze(self, text: str, urls: list[str], assertions: list[str] | None = None) -> AnalyzerResult:
        """
        Executa a checagem das alegações isoladamente.
        Se 'assertions' for fornecido com múltiplas alegações, checa cada uma em paralelo.
        Caso contrário, checa a alegação contida em 'text'.
        """
        targets = [a.strip() for a in assertions if a and a.strip()] if assertions else []
        if not targets:
            targets = [text.strip()] if text.strip() else []

        # Limita a no máximo 4 alegações para evitar sobrecarga de rede
        targets = targets[:4]

        # Executa a checagem de cada alegação em paralelo
        sub_results = await asyncio.gather(*[self.check_single_claim(t) for t in targets])

        all_evidences = []
        all_sources = []
        for sr in sub_results:
            all_evidences.extend(sr.get("evidences", []))
            all_sources.extend(sr.get("sources", []))

        verdict, confidence, reasons = self.aggregate_sub_verdicts(sub_results)

        if len(sub_results) > 1:
            summary = (
                f"Varredura em fontes externas avaliou {len(sub_results)} alegações isoladamente. "
                f"Resultado composto: {verdict.value} com {len(all_evidences)} registro(s) localizado(s)."
            )
        else:
            summary = (
                f"Varredura em fontes externas encontrou {len(all_evidences)} registro(s) relevante(s)."
                if all_evidences
                else "Nenhuma evidência externa direta localizada nas fontes consultadas."
            )

        return AnalyzerResult(
            analyzer_name="fact_check_api",
            verdict=verdict,
            confidence=confidence,
            claim=text,
            summary=summary,
            reasons=reasons,
            sources=list(dict.fromkeys(all_sources))[:6] or ["Mídia de Referência e Órgãos Oficiais"],
            raw_details={
                "total_evidences": len(all_evidences),
                "claims_checked": len(sub_results),
                "sub_claims": sub_results,
                "evidences": all_evidences,
            },
        )
