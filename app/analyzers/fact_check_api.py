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
    # Autoridades Sanitárias, Científicas e Oficiais
    "who": "Organização Mundial da Saúde (WHO/OMS)",
    "paho": "Organização Pan-Americana da Saúde (OPAS)",
    "webmd": "WebMD Health",
    "fiocruz": "Fiocruz",
    "anvisa": "Anvisa",
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
        Executa Leitura Horizontal em fontes confiáveis (G1, Folha, Estadão, BBC, WHO, WebMD, Fiocruz, Anvisa)
        utilizando busca jornalística agregada para obter matérias de apuração em tempo real.
        """
        # Limpa termos redundantes de busca para maximizar recall
        clean_q = re.sub(r"[\"']", "", query)
        encoded_query = urllib.parse.quote(clean_q[:160])
        rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=pt-BR&gl=BR&ceid=BR:pt-419"

        evidences: list[EvidenceItem] = []
        try:
            resp = await self.http_client.get(rss_url)
            if resp.status_code == 200:
                root = ET.fromstring(resp.text)
                items = root.findall(".//item")

                for it in items[:8]:
                    title_elem = it.find("title")
                    link_elem = it.find("link")
                    source_elem = it.find("source")
                    pub_elem = it.find("pubDate")

                    title = title_elem.text if title_elem is not None else ""
                    link = link_elem.text if link_elem is not None else ""
                    source_name = source_elem.text if source_elem is not None else "Imprensa"
                    pub_date = pub_elem.text if pub_elem is not None else None

                    # Identifica se o veículo pertence à lista de fontes confiáveis
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
            logger.debug("Falha na leitura horizontal de notícias: %s", e)

        return evidences

    def evaluate_verdict(self, evidences: list[EvidenceItem]) -> tuple[Verdict, float, list[str]]:
        """
        Calcula o veredito e nível de confiança a partir das evidências consolidadas.
        """
        if not evidences:
            return (
                Verdict.INCONCLUSIVO,
                0.50,
                ["Nenhuma checagem prévia ou matéria em veículos de referência foi encontrada para este fato."],
            )

        # 1. Se houver checagem direta do Google Fact Check Tools (IFCN)
        fact_checks = [e for e in evidences if e.is_fact_check and e.rating]
        if fact_checks:
            ratings_text = " ".join(f.rating.lower() for f in fact_checks if f.rating)
            if any(k in ratings_text for k in ("falso", "fake", "mentira", "adulterado", "falsa", "incorreto")):
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
        debunk_count = sum(1 for e in evidences if DEBUNK_TITLE_PATTERNS.search(e.title))
        confirm_count = sum(1 for e in evidences if CONFIRM_TITLE_PATTERNS.search(e.title))

        if debunk_count >= 1:
            matching = [e for e in evidences if DEBUNK_TITLE_PATTERNS.search(e.title)]
            reasons = [
                f"Leitura horizontal ({m.source_name}): aponta desmentido ou contestação na matéria \"{m.title}\"."
                for m in matching[:2]
            ]
            confidence = 0.90 if debunk_count >= 2 else 0.80
            return Verdict.FAKE, confidence, reasons

        if confirm_count >= 1:
            matching = [e for e in evidences if CONFIRM_TITLE_PATTERNS.search(e.title)]
            reasons = [
                f"Leitura horizontal ({m.source_name}): confirmação de atos ou ocorrência em \"{m.title}\"."
                for m in matching[:2]
            ]
            confidence = 0.88 if confirm_count >= 2 else 0.78
            return Verdict.VERDADEIRO, confidence, reasons

        # 3. Caso haja matérias encontradas mas sem sinal claro de confirmação ou desmentido
        top_titles = [f"\"{e.title}\" ({e.source_name})" for e in evidences[:2]]
        return (
            Verdict.INCONCLUSIVO,
            0.55,
            [
                f"Matérias encontradas em fontes de referência, porém sem termo explícito de desmentido ou confirmação direta: {', '.join(top_titles)}."
            ],
        )

    async def analyze(self, text: str, urls: list[str]) -> AnalyzerResult:
        """
        Executa a checagem em 2 camadas: Google Fact Check Tools API e Leitura Horizontal.
        """
        # 1. Consulta em paralelo: Google Fact Check API + Leitura Horizontal
        fc_task = self.search_google_fact_check(text)
        lateral_task = self.search_lateral_reading(text)

        results = await asyncio.gather(fc_task, lateral_task, return_exceptions=True)
        google_evidences = results[0] if isinstance(results[0], list) else []
        lateral_evidences = results[1] if isinstance(results[1], list) else []

        all_evidences = google_evidences + lateral_evidences

        # 2. Avalia veredito e confiança baseado no conjunto de evidências
        verdict, confidence, reasons = self.evaluate_verdict(all_evidences)

        # 3. Consolida fontes para exibição
        sources_list: list[str] = []
        for e in all_evidences:
            src = f"{e.source_name}: {e.title}"
            if src not in sources_list:
                sources_list.append(src)

        if not sources_list:
            sources_list = ["Google Fact Check Tools API / Leitura Horizontal em Mídia de Referência"]

        return AnalyzerResult(
            analyzer_name="fact_check_api",
            verdict=verdict,
            confidence=confidence,
            claim=text,
            summary=(
                f"Varredura em fontes externas encontrou {len(all_evidences)} registro(s) relevante(s)."
                if all_evidences
                else "Nenhuma evidência externa direta localizada nas fontes consultadas."
            ),
            reasons=reasons,
            sources=sources_list[:5],
            raw_details={
                "total_evidences": len(all_evidences),
                "google_fact_checks_count": len(google_evidences),
                "lateral_reading_count": len(lateral_evidences),
                "evidences": [e.model_dump() for e in all_evidences[:6]],
            },
        )
