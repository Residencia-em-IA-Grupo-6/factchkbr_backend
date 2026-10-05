import asyncio
import logging
import re
import urllib.parse
import xml.etree.ElementTree as ET
from enum import Enum
from typing import Any
from pydantic import BaseModel, Field

import httpx

from app.config import get_settings
from app.core.base import BaseAnalyzer
from app.core.registry import register_analyzer
from app.schemas.analysis import AnalyzerResult, Verdict

logger = logging.getLogger("factchkbr.analyzers.fact_check_api")


class SourceTier(str, Enum):
    """Níveis de credibilidade e autoridade das fontes de evidência."""
    TIER1_OFFICIAL_OR_IFCN = "tier1_official_or_ifcn"
    TIER2_MAINSTREAM_MEDIA = "tier2_mainstream_media"
    TIER3_GENERAL_MEDIA = "tier3_general_media"
    UNKNOWN = "unknown"


TIER_WEIGHTS = {
    SourceTier.TIER1_OFFICIAL_OR_IFCN.value: 3.0,
    SourceTier.TIER2_MAINSTREAM_MEDIA.value: 1.5,
    SourceTier.TIER3_GENERAL_MEDIA.value: 0.8,
    SourceTier.UNKNOWN.value: 0.0,
}


class EvidenceItem(BaseModel):
    """Evidência factual recuperada via Fact Check Tools ou Leitura Horizontal."""
    title: str = Field(..., description="Título da matéria, comunicado ou desmentido")
    source_name: str = Field(..., description="Nome do veículo de imprensa ou autoridade (ex: G1, Lupa, WHO, Anvisa)")
    url: str = Field(..., description="Link da evidência")
    snippet: str = Field(default="", description="Trecho ou resumo descritivo")
    rating: str | None = Field(default=None, description="Classificação do checador (ex: Falso, Enganoso, Verdadeiro)")
    is_fact_check: bool = Field(default=False, description="Indica se é de agência de checagem oficial")
    published_date: str | None = Field(default=None, description="Data de publicação")
    claim_reviewed: str | None = Field(default=None, description="Alegação original revisada pelo checador")
    source_tier: str = Field(default=SourceTier.TIER2_MAINSTREAM_MEDIA.value, description="Nível de credibilidade da fonte")
    is_relevant: bool = Field(default=True, description="Indica se o conteúdo tem relação temática direta com a alegação")
    stance: str = Field(default="NEUTRAL", description="Posicionamento da evidência: SUPPORTS | REFUTES | NEUTRAL")


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

TIER1_HOST_SUFFIXES = (
    ".gov.br", ".leg.br", ".jus.br", ".mil.br",
    "gov.br", "senado.leg.br", "camara.leg.br", "tse.jus.br", "stf.jus.br", "planalto.gov.br",
    "anvisa.gov.br", "fiocruz.br", "ibge.gov.br", "ipea.gov.br", "saude.gov.br",
    "aosfatos.org", "lupa.news", "boatos.org", "projetocomprova.com.br",
    "who.int", "paho.org", "cdc.gov",
)

TIER2_HOST_SUFFIXES = (
    "g1.globo.com", "globo.com", "folha.uol.com.br", "folha.com.br",
    "estadao.com.br", "uol.com.br", "bbc.com", "bbc.co.uk",
    "cnnbrasil.com.br", "cnn.com", "reuters.com", "agenciabrasil.ebc.com.br",
    "ebc.com.br", "valor.globo.com", "valorinveste.globo.com",
    "nexojornal.com.br", "metropoles.com", "terra.com.br",
    "dw.com", "elpais.com",
    # Grandes veículos editoriais e de imprensa nacional
    "abril.com.br", "veja.abril.com.br", "vejasp.abril.com.br", "exame.com",
    "r7.com", "correiobraziliense.com.br", "cartacapital.com.br",
    "jovempan.com.br", "poder360.com.br", "congressoemfoco.com.br",
    "sbtnews.com.br", "band.uol.com.br", "band.com.br",
)

TIER1_SOURCE_NAMES = {
    "agência lupa", "lupa", "aos fatos", "boatos.org", "boatos",
    "projeto comprova", "comprova", "afp checamos", "checamos",
    "fato ou fake", "estadao verifica", "estadão verifica", "uol confere", "confere",
    "ministério da saúde", "anvisa", "fiocruz", "ibge", "ipea",
    "tribunal superior eleitoral", "tse", "supremo tribunal federal", "stf",
    "organização mundial da saúde", "oms", "who", "opas",
}

TIER2_SOURCE_NAMES = {
    "g1", "o globo", "globo", "folha de s.paulo", "folha", "estadão", "estadao",
    "uol", "bbc news brasil", "bbc", "cnn brasil", "cnn", "reuters",
    "agência brasil", "ebc", "valor econômico", "nexo jornal", "metrópoles", "metropoles", "terra",
    "veja", "veja são paulo", "veja sp", "veja rio", "abril", "exame", "r7",
    "correio braziliense", "correio brasiliense", "carta capital", "jovem pan",
    "poder360", "congresso em foco", "sbt news", "sbt", "band",
}

STOP_WORDS_PT = {
    "de", "a", "o", "que", "e", "do", "da", "em", "um", "para", "é", "com", "não",
    "uma", "os", "no", "se", "na", "por", "mais", "as", "dos", "como", "mas", "foi",
    "ao", "ele", "das", "tem", "à", "seu", "sua", "ou", "ser", "quando", "muito",
    "há", "nos", "já", "está", "eu", "também", "só", "pelo", "pela", "até", "isso",
    "ela", "entre", "era", "depois", "sem", "mesmo", "aos", "ter", "seus", "quem",
    "nas", "me", "esse", "eles", "estão", "você", "tinha", "foram", "essa", "num",
    "nem", "suas", "meu", "às", "minha", "têm", "numa", "pelos", "elas", "havia",
    "este", "esta", "estes", "estas", "ontem", "hoje", "amanhã", "disse", "diz",
    "sobre", "após", "segundo", "onde", "qual", "pode", "podem", "vai", "vão",
    "foram", "tudo", "todo", "toda", "todos", "todas", "outro", "outra", "outros",
}

NEGATION_PATTERNS = re.compile(
    r"\b(?:não|nunca|jamais|tampouco|nenhum|nenhuma|sem|falso que|inverídico|impossível)\b",
    re.IGNORECASE,
)

DEBUNK_TITLE_PATTERNS = re.compile(
    r"\b(?:"
    r"é falso|é mentira|é fake|é boato|não é verdade|desmente|desmentiu|nega|negou|"
    r"boato|fake news|falso|falsa|engana|enganosa|distorce|distorcida|"
    r"não causou|não causa|não causam|não mata|não matam|"
    r"não t[eê]m? relação|não há relação|não provocam?|"
    r"golpe do|golpe da|é golpe|trata-se de golpe|caiu no golpe|alerta de golpe|falso que"
    r")\b",
    re.IGNORECASE,
)

CONFIRM_TITLE_PATTERNS = re.compile(
    r"\b(?:"
    r"confirma|confirmou|aprova|aprovou|autoriza|autorizou|determina|determinou|"
    r"proíbe|proibiu|suspende|suspendeu|recolhe|recolhimento|anuncia|anunciou|"
    r"sanciona|sancionou|publica|publicou|pode ser utilizad[oa]|passa a valer|"
    r"passam a valer|servirá como|valerá como|é verdade|é fato|comprova|comprovou|"
    r"lamenta|lamentou|lamento|pesar|nota de pesar|pesar pelo falecimento|"
    r"morre|morreu|morte|falecimento|falece|faleceu|"
    r"emite nota|emitiu nota|manifesta|manifestou|divulga|divulgou|informa|informou|"
    r"alerta|alertou|comunica|comunicou|declara|declarou|registra|registrou|"
    r"lança|lançou|inicia|iniciou|atinge|atingiu|assina|assinou|decreta|decretou|"
    r"entrega|entregou|recomenda|recomendou|presta homenagem|homenageia|homenageou|"
    r"interdita|interditou|autua|autuou|apreende|apreendeu|reconhece|reconheceu|"
    r"concede|concedeu|destaca|destacou|reforça|reforçou"
    r")\b",
    re.IGNORECASE,
)


META_DEBUNK_PREFIX = re.compile(
    r"^(?:(?:fato ou fake|uol confere|comprova|estadao verifica|estadão verifica|aos fatos|lupa|afp checamos|afp|boatos\.org)\s*[:\-]\s*)?"
    r"(?:não é verdade que|não procede que|é falso que|é mentira que|é fake que|é boato que|boato de que|falso que|falsa que|desmentido:?|alerta:?)\s*",
    re.IGNORECASE,
)


def is_meta_debunk_claim(text: str) -> bool:
    """Verifica se a alegação é uma meta-asserção de desmentido (ex: 'É falso que...', 'Não é verdade que...')."""
    return bool(META_DEBUNK_PREFIX.match(text.strip()))


def clean_reviewed_claim(text: str) -> str:
    """Remove prefixos jornalísticos comuns de checagem para isolar o núcleo da alegação."""
    cleaned = re.sub(
        r"^(?:(?:fato ou fake|uol confere|comprova|estadao verifica|estadão verifica|aos fatos|lupa|afp checamos|afp|boatos\.org)\s*[:\-]\s*)?",
        "",
        text.strip(),
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"^(?:não é verdade que|não procede que|é falso que|é mentira que|é fake que|é boato que|boato de que|falso que|falsa que|desmentido:?|alerta:?)\s*",
        "",
        cleaned.strip(),
        flags=re.IGNORECASE,
    )
    return cleaned.strip()



TIER1_FACT_CHECK_NAMES = {
    "agência lupa", "lupa", "aos fatos", "boatos.org", "boatos",
    "projeto comprova", "comprova", "afp checamos", "checamos",
    "fato ou fake", "estadao verifica", "estadão verifica", "uol confere", "confere",
}


def get_source_tier(source_name: str, url: str = "", is_fact_check: bool = False) -> SourceTier:
    """Classifica a credibilidade da fonte em tiers baseados na tipologia da fonte."""
    if is_fact_check:
        return SourceTier.TIER1_OFFICIAL_OR_IFCN

    host = ""
    if url:
        try:
            parsed = urllib.parse.urlparse(url)
            host = (parsed.netloc or "").lower().split(":")[0]
        except Exception:
            host = ""

    # Rejeita plataformas de blogs gratuitos ou domínios não confiáveis
    if host and any(host.endswith(b) for b in ("blogspot.com", "wordpress.com", "wixsite.com")):
        return SourceTier.UNKNOWN

    # 1. Agências de fact-checking IFCN reconhecidas (mesmo hospedadas em portais parceiros como Folha, UOL, etc.)
    s_clean = re.sub(r"[^\w\s]", "", source_name.lower()).strip()
    if any(s_clean == fc or re.search(rf"\b{re.escape(fc)}\b", s_clean) for fc in TIER1_FACT_CHECK_NAMES):
        return SourceTier.TIER1_OFFICIAL_OR_IFCN

    # 2. Validação para agregadores (ex: Google News RSS com links em news.google.com)
    is_aggregator = host in ("news.google.com", "google.com", "news.google.com.br")
    if not host or is_aggregator:
        for t1 in TIER1_SOURCE_NAMES:
            if s_clean == t1 or re.search(rf"\b{re.escape(t1)}\b", s_clean):
                return SourceTier.TIER1_OFFICIAL_OR_IFCN
        for t2 in TIER2_SOURCE_NAMES:
            if s_clean == t2 or re.search(rf"\b{re.escape(t2)}\b", s_clean):
                return SourceTier.TIER2_MAINSTREAM_MEDIA
        if s_clean:
            return SourceTier.TIER3_GENERAL_MEDIA
        return SourceTier.UNKNOWN

    # 3. Validação por domínio qualificado (evita spoofing por substring no path ou query)
    if any(host == d or host.endswith("." + d) for d in TIER1_HOST_SUFFIXES):
        return SourceTier.TIER1_OFFICIAL_OR_IFCN
    if any(host == d or host.endswith("." + d) for d in TIER2_HOST_SUFFIXES):
        return SourceTier.TIER2_MAINSTREAM_MEDIA

    # Se a URL existe mas não pertence aos domínios confiáveis oficiais/IFCN, classifica como Tier 3 se TLD jornalístico
    if any(host.endswith(tld) for tld in (".com.br", ".org.br", ".edu.br", ".net.br")):
        return SourceTier.TIER3_GENERAL_MEDIA
    return SourceTier.UNKNOWN


def extract_substantive_tokens(text: str) -> list[str]:
    """Extrai palavras substantivas (sem stop words) para verificação temática."""
    raw_tokens = re.findall(r"\b[a-zA-ZáéíóúâêîôûãõçÁÉÍÓÚÂÊÎÔÛÃÕÇ0-9]{3,}\b", text.lower())
    return [t for t in raw_tokens if t not in STOP_WORDS_PT]


def check_evidence_relevance(claim: str, evidence_text: str) -> bool:
    """
    Verifica se a evidência trata especificamente do mesmo assunto da alegação,
    evitando falsos positivos gerados por notícias tangenciais ou coincidentes.
    """
    clean_c = clean_reviewed_claim(claim)
    claim_tokens = extract_substantive_tokens(clean_c if clean_c else claim)
    if not claim_tokens:
        return True
    ev_tokens = set(extract_substantive_tokens(evidence_text))

    def _token_match(ct: str, et: str) -> bool:
        if ct == et:
            return True
        # Variações flexionais comuns (plural / singular / gênero) com raiz idêntica
        if len(ct) >= 4 and len(et) >= 4:
            if ct.startswith(et) or et.startswith(ct):
                return abs(len(ct) - len(et)) <= 2
            if len(ct) >= 5 and len(et) >= 5 and ct[:5] == et[:5]:
                return abs(len(ct) - len(et)) <= 2
        return False

    matches = 0
    for ct in claim_tokens:
        if any(_token_match(ct, et) for et in ev_tokens):
            matches += 1

    n_tokens = len(claim_tokens)
    if n_tokens <= 2:
        return matches >= n_tokens
    return (matches / n_tokens) >= 0.50


@register_analyzer("fact_check_api", weight=1.5)
class FactCheckApiAnalyzer(BaseAnalyzer):
    """
    Analisador 2: Evidências Externas de Fact-Checking & Leitura Horizontal.
    - Fonte Primária: Google Fact Check Tools API (checagens oficiais IFCN).
    - Leitura Horizontal: Varredura em veículos de referência (G1, Folha, Estadão, BBC)
      e autoridades científicas e sanitárias (WHO, Fiocruz, Anvisa, IBGE).
    - Stance Detection com sensibilidade à polaridade e credibilidade de fontes.
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

    async def fetch_user_urls(self, urls: list[str]) -> list[EvidenceItem]:
        """Extrai conteúdo e metadados de URLs fornecidas diretamente pelo usuário."""
        if not urls:
            return []
        evidences: list[EvidenceItem] = []
        for url in urls[:3]:
            try:
                resp = await self.http_client.get(url, timeout=4.0, follow_redirects=True)
                if resp.status_code == 200:
                    text_html = resp.text
                    title_m = re.search(r"<title[^>]*>(.*?)</title>", text_html, re.IGNORECASE | re.DOTALL)
                    title = title_m.group(1).strip() if title_m else url
                    title = re.sub(r"\s+", " ", title)

                    desc_m = re.search(r'<meta[^>]+name=["\']description["\'][^>]+content=["\']([^"\']+)["\']', text_html, re.IGNORECASE)
                    desc = desc_m.group(1).strip() if desc_m else ""

                    parsed_url = urllib.parse.urlparse(url)
                    source_name = parsed_url.netloc or "Link Fornecido"
                    tier = get_source_tier(source_name, url)

                    evidences.append(
                        EvidenceItem(
                            title=title,
                            source_name=f"{source_name} (URL informada)",
                            url=url,
                            snippet=desc or title,
                            source_tier=tier.value,
                            is_fact_check=tier == SourceTier.TIER1_OFFICIAL_OR_IFCN,
                        )
                    )
            except Exception as e:
                logger.debug("Não foi possível acessar a URL informada pelo usuário (%s): %s", url, e)
        return evidences

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
                                claim_reviewed=claim_text,
                                source_tier=SourceTier.TIER1_OFFICIAL_OR_IFCN.value,
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
        """Recupera e processa itens de um feed RSS de notícias com extração de lead/snippet."""
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
                    desc_elem = it.find("description")

                    title = title_elem.text if title_elem is not None else ""
                    link = link_elem.text if link_elem is not None else ""
                    source_name = source_elem.text if source_elem is not None else "Imprensa"
                    pub_date = pub_elem.text if pub_elem is not None else None

                    desc_raw = desc_elem.text if desc_elem is not None and desc_elem.text else ""
                    desc_clean = re.sub(r"<[^>]+>", " ", desc_raw)
                    desc_clean = re.sub(r"\s+", " ", desc_clean).strip()
                    snippet = desc_clean if desc_clean else title

                    if force_official:
                        if "ibge.gov.br" in link.lower() or "ibge" in source_name.lower():
                            source_name = "IBGE (Dados Oficiais)"
                        elif "ipea.gov.br" in link.lower() or "ipea" in source_name.lower():
                            source_name = "Ipea (Dados Oficiais)"

                    is_fact_check = (
                        any(fc in title.lower() for fc in ("fato ou fake", "verifica", "confere", "comprova", "checagem"))
                        or any(fc in source_name.lower() for fc in ("lupa", "aos fatos", "boatos"))
                    )
                    tier = get_source_tier(source_name, link, is_fact_check)

                    rating = None
                    if is_fact_check and DEBUNK_TITLE_PATTERNS.search(title):
                        rating = "Desmentido"

                    evidences.append(
                        EvidenceItem(
                            title=title,
                            source_name=source_name,
                            url=link,
                            snippet=snippet,
                            rating=rating,
                            is_fact_check=is_fact_check,
                            published_date=pub_date,
                            source_tier=tier.value,
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
        Calcula o veredito e nível de confiança a partir das evidências consolidadas,
        utilizando filtro de relevância temática, alinhamento de polaridade (stance)
        e ponderação por autoridade da fonte.
        """
        if not evidences:
            return (
                Verdict.FAKE,
                0.75,
                [
                    "Nenhuma checagem prévia ou matéria em veículos de referência foi encontrada para este fato.",
                    "Alegação não encontrada em fontes oficiais ou veículos confiáveis: sob o princípio de ônus da prova, afirmações públicas, promessas de tratamento ou declarações atribuídas sem respaldo factual são tratadas como provavelmente falsas (boato sem sustentação).",
                ],
            )

        user_has_meta = is_meta_debunk_claim(claim)
        core_claim = clean_reviewed_claim(claim) if user_has_meta else claim.strip()
        core_claim_has_neg = bool(NEGATION_PATTERNS.search(core_claim))
        supporting: list[tuple[EvidenceItem, float, str]] = []
        refuting: list[tuple[EvidenceItem, float, str]] = []
        suspect_evidences: list[tuple[EvidenceItem, float, str]] = []

        for ev in evidences:
            ev_text = f"{ev.title} {ev.snippet} {ev.claim_reviewed or ''}"
            is_relevant = check_evidence_relevance(claim, ev_text)
            ev.is_relevant = is_relevant
            if not is_relevant:
                ev.stance = "NEUTRAL"
                continue

            tier = SourceTier(ev.source_tier) if ev.source_tier in SourceTier.__members__.values() else get_source_tier(ev.source_name, ev.url, ev.is_fact_check)
            weight = TIER_WEIGHTS.get(tier.value, 0.0)
            if weight <= 0.0:
                ev.stance = "NEUTRAL"
                continue

            # 1. Se for checagem formal do Google Fact Check Tools / IFCN
            if ev.is_fact_check and ev.rating:
                r_lower = ev.rating.lower()
                is_debunk = any(k in r_lower for k in ("falso", "fake", "mentira", "adulterado", "falsa", "incorreto", "desmentido"))
                is_confirm = any(k in r_lower for k in ("verdadeiro", "fato", "verdade", "correto", "comprovado"))
                is_misleading = any(k in r_lower for k in ("enganoso", "distorcido", "fora de contexto", "impreciso", "exagerado"))

                # Remove prefixos jornalísticos de desmentido para avaliar a polaridade real da tese apurada
                clean_rev = clean_reviewed_claim(ev.claim_reviewed or ev.title or "")
                rev_has_neg = bool(NEGATION_PATTERNS.search(clean_rev))
                same_polarity = (core_claim_has_neg == rev_has_neg)

                # Se a alegação de teste não foi passada (chamada direta), assume alinhamento direto
                if not claim:
                    same_polarity = True

                if is_misleading:
                    ev.stance = "SUSPECT"
                    suspect_evidences.append((ev, weight, f"Classificado como enganoso/fora de contexto por {ev.source_name}: '{ev.rating}'."))
                elif is_debunk:
                    if user_has_meta:
                        if same_polarity:
                            ev.stance = "SUPPORTS"
                            supporting.append((ev, weight, f"Checador oficial ({ev.source_name}) confirma desmentido do boato apontado: '{ev.rating}'."))
                        else:
                            ev.stance = "REFUTES"
                            refuting.append((ev, weight, f"Checador oficial ({ev.source_name}) desmentiu a tese contrária: '{ev.rating}'."))
                    else:
                        if same_polarity:
                            ev.stance = "REFUTES"
                            refuting.append((ev, weight, f"Desmentido por checador oficial ({ev.source_name}): classificação '{ev.rating}' para a alegação."))
                        else:
                            ev.stance = "SUPPORTS"
                            supporting.append((ev, weight, f"Desmentido da tese oposta por checador oficial ({ev.source_name}): classificação '{ev.rating}'."))
                elif is_confirm:
                    if user_has_meta:
                        if same_polarity:
                            ev.stance = "REFUTES"
                            refuting.append((ev, weight, f"Checador oficial ({ev.source_name}) confirmou a tese, contrapondo a alegação de boato: '{ev.rating}'."))
                        else:
                            ev.stance = "SUPPORTS"
                            supporting.append((ev, weight, f"Checador oficial ({ev.source_name}) comprovou tese oposta: '{ev.rating}'."))
                    else:
                        if same_polarity:
                            ev.stance = "SUPPORTS"
                            supporting.append((ev, weight, f"Comprovado por checador oficial ({ev.source_name}): classificação '{ev.rating}'."))
                        else:
                            ev.stance = "REFUTES"
                            refuting.append((ev, weight, f"Checador oficial ({ev.source_name}) confirmou a tese oposta: '{ev.rating}'."))
                continue

            # 2. Leitura Horizontal em Mídia de Referência / Órgãos Oficiais
            has_debunk = bool(DEBUNK_TITLE_PATTERNS.search(ev.title))
            has_confirm = bool(CONFIRM_TITLE_PATTERNS.search(ev.title))

            clean_title = clean_reviewed_claim(ev.title)
            title_has_neg = bool(NEGATION_PATTERNS.search(clean_title))
            same_polarity = (core_claim_has_neg == title_has_neg)
            if not claim:
                same_polarity = True

            if has_debunk:
                if user_has_meta:
                    if same_polarity:
                        ev.stance = "SUPPORTS"
                        supporting.append((ev, weight, f"Leitura horizontal ({ev.source_name}): confirmação de desmentido do boato em \"{ev.title}\"."))
                    else:
                        ev.stance = "REFUTES"
                        refuting.append((ev, weight, f"Leitura horizontal ({ev.source_name}): desmente tese contrária em \"{ev.title}\"."))
                else:
                    if same_polarity:
                        ev.stance = "REFUTES"
                        refuting.append((ev, weight, f"Leitura horizontal ({ev.source_name}): aponta desmentido ou contestação na matéria \"{ev.title}\"."))
                    else:
                        ev.stance = "SUPPORTS"
                        supporting.append((ev, weight, f"Leitura horizontal ({ev.source_name}): desmente a tese contrária em \"{ev.title}\"."))
            elif has_confirm:
                if user_has_meta:
                    if same_polarity:
                        ev.stance = "REFUTES"
                        refuting.append((ev, weight, f"Leitura horizontal ({ev.source_name}): comprovação do fato contrapõe a alegação de boato em \"{ev.title}\"."))
                    else:
                        ev.stance = "SUPPORTS"
                        supporting.append((ev, weight, f"Leitura horizontal ({ev.source_name}): confirmação em \"{ev.title}\"."))
                else:
                    if same_polarity:
                        ev.stance = "SUPPORTS"
                        supporting.append((ev, weight, f"Leitura horizontal ({ev.source_name}): confirmação de atos ou ocorrência em \"{ev.title}\"."))
                    else:
                        ev.stance = "REFUTES"
                        refuting.append((ev, weight, f"Leitura horizontal ({ev.source_name}): confirmação da tese oposta em \"{ev.title}\"."))
            else:
                ev.stance = "NEUTRAL"

        # Se houver checagem expressa classificando como enganoso/distorcido
        if suspect_evidences and not (refuting and not supporting):
            reasons = [r for _, _, r in suspect_evidences[:2]]
            return Verdict.SUSPEITO, 0.85, reasons

        sup_w = sum(w for _, w, _ in supporting)
        ref_w = sum(w for _, w, _ in refuting)
        has_ifcn = any(ev.is_fact_check for ev, _, _ in (refuting + supporting + suspect_evidences))

        # Decisão calibrada baseada no peso das evidências
        if ref_w >= 1.5 and ref_w > sup_w:
            base_conf = 0.92 if has_ifcn else 0.78
            conf = min(0.98, max(base_conf, round(0.70 + (ref_w / (ref_w + sup_w + 1.0)) * 0.28, 2)))
            reasons = [r for _, _, r in refuting[:2]]
            return Verdict.FAKE, conf, reasons
        elif sup_w >= 1.5 and sup_w > ref_w:
            base_conf = 0.90 if has_ifcn else 0.78
            conf = min(0.95, max(base_conf, round(0.70 + (sup_w / (sup_w + ref_w + 1.0)) * 0.25, 2)))
            reasons = [r for _, _, r in supporting[:2]]
            return Verdict.VERDADEIRO, conf, reasons
        elif (ref_w >= 1.2 and sup_w >= 1.2) or suspect_evidences:
            reasons = [r for _, _, r in suspect_evidences[:2]] if suspect_evidences else ["Fontes confiáveis apresentam dados ou posições divergentes sobre o tema."]
            return Verdict.SUSPEITO, 0.85, reasons
        elif ref_w >= 1.2 and sup_w >= 1.2:
            return Verdict.SUSPEITO, 0.85, ["Fontes confiáveis apresentam dados ou posições divergentes sobre o tema."]

        # 3. Caso não haja evidências com peso suficiente para confirmação ou refutação
        relevant_evidences = [e for e in evidences if e.is_relevant]
        if relevant_evidences:
            top_titles = [f"\"{e.title}\" ({e.source_name})" for e in relevant_evidences[:2]]
            return (
                Verdict.INCONCLUSIVO,
                0.55,
                [
                    f"Matérias encontradas em fontes de referência, porém sem termo explícito de desmentido ou confirmação direta: {', '.join(top_titles)}.",
                    "Imprecisão por dados insuficientes: os registros tratam do assunto de forma genérica, sem comprovar nem desmentir categoricamente os pontos específicos da alegação.",
                ],
            )

        return (
            Verdict.FAKE,
            0.75,
            [
                "Nenhuma checagem prévia ou matéria em veículos de referência foi encontrada para este fato.",
                "Alegação não encontrada em fontes oficiais ou veículos confiáveis: sob o princípio de ônus da prova, afirmações públicas, promessas de tratamento ou declarações atribuídas sem respaldo factual são tratadas como provavelmente falsas (boato sem sustentação).",
            ],
        )

    async def check_single_claim(self, claim_text: str, extra_evidences: list[EvidenceItem] | None = None) -> dict[str, Any]:
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
        all_evidences = (extra_evidences or []) + google_evidences + lateral_evidences

        verdict, confidence, reasons = self.evaluate_verdict(all_evidences, claim=clean_text)

        sources = []
        for e in all_evidences[:3]:
            sources.append(f"{e.source_name}: {e.title}")

        if verdict == Verdict.FAKE:
            if reasons and any("não encontrada" in r.lower() for r in reasons):
                justification = "Alegação sem respaldo em fontes oficiais ou veículos confiáveis (não encontrada / provavelmente falso)."
            else:
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
            "google_fact_check_count": len(google_evidences),
            "lateral_reading_count": len(lateral_evidences),
        }

    def aggregate_sub_verdicts(self, sub_results: list[dict[str, Any]]) -> tuple[Verdict, float, list[str]]:
        """
        Calcula o impacto composto das sub-alegações no score e no veredito geral:
        - Misto (FAKE + VERDADEIRO): SUSPEITO (desinformação mista/engano)
        - Todas FAKE: FAKE
        - Todas VERDADEIRO: VERDADEIRO
        - FAKE + INCONCLUSIVO: FAKE
        - VERDADEIRO + INCONCLUSIVO: VERDADEIRO
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
                f"Conteúdo misto detectado: {len(fake_claims)} alegação(ões) falsa(s)/sem respaldo e "
                f"{len(true_claims)} verdadeira(s) identificadas no mesmo texto."
            )
            return verdict, confidence, [summary_reason] + itemized_reasons

        # 2. Se contiver apenas alegações falsas (ou falsas + inconclusivas) -> FAKE
        if fake_claims:
            verdict = Verdict.FAKE
            confidence = round(max(r["confidence"] for r in fake_claims), 2)
            summary_reason = f"Falsidade factual / boato: {len(fake_claims)} alegação(ões) sem respaldo em fontes confiáveis ou desmentidas."
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
        Lê e incorpora metadados de 'urls' enviadas pelo usuário.
        """
        targets = [a.strip() for a in assertions if a and a.strip()] if assertions else []
        if not targets:
            targets = [text.strip()] if text.strip() else []

        targets = targets[:4]

        # Extrai conteúdo de URLs informadas pelo usuário
        user_evidences = await self.fetch_user_urls(urls) if urls else []

        # Executa a checagem de cada alegação em paralelo
        sub_results = await asyncio.gather(*[self.check_single_claim(t, extra_evidences=user_evidences) for t in targets])

        all_evidences = []
        all_sources = []
        total_fc_count = 0
        total_lat_count = 0
        for sr in sub_results:
            all_evidences.extend(sr.get("evidences", []))
            all_sources.extend(sr.get("sources", []))
            total_fc_count += sr.get("google_fact_check_count", 0)
            total_lat_count += sr.get("lateral_reading_count", 0)

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
                "google_fact_check_count": total_fc_count,
                "lateral_reading_count": total_lat_count,
                "claims_checked": len(sub_results),
                "sub_claims": sub_results,
                "evidences": all_evidences,
            },
        )
