import re

from app.analyzers.lexicon_repository import BaseLexiconRepository, get_lexicon_repository
from app.core.base import BaseAnalyzer
from app.core.registry import register_analyzer
from app.schemas.analysis import AnalyzerResult


@register_analyzer("heuristic", weight=0.6)
class HeuristicAnalyzer(BaseAnalyzer):
    """
    Analisador 1: Regras e Heurísticas de texto (clickbait, sensacionalismo e estilometria).
    
    IMPORTANTE: Este analisador NÃO emite veredito de veracidade (retorna verdict=None).
    Ele opera exclusivamente como módulo de apoio e extração de features normalizadas
    para alimentar modelos de decisão mais complexos (LLMs, classificadores e ensemble).
    """

    def __init__(self, repository: BaseLexiconRepository | None = None) -> None:
        super().__init__()
        self.repository = repository or get_lexicon_repository()

    def extract_features(self, text: str) -> dict[str, float]:
        """
        Extrai as 7 métricas padronizadas de estilometria e sensacionalismo:
        - uppercase_ratio
        - allcaps_words_ratio
        - excessive_punctuation_count
        - exclamation_density
        - question_density
        - urgency_lexicon_density
        - composite_sensationalism_score
        """
        # 1. Proporção de Uppercase
        letters = [c for c in text if c.isalpha()]
        n_letters = len(letters)
        n_upper = sum(1 for c in letters if c.isupper())
        uppercase_ratio = (n_upper / n_letters) if n_letters > 0 else 0.0

        # 2. Tokens em ALL CAPS (comprimento >= 3 ou artigos/palavras curtas fora de início de frase/pós-ponto)
        common_acronyms = self.repository.get_acronyms()
        word_matches = list(re.finditer(r"\b[a-zA-ZáéíóúâêîôûãõçÁÉÍÓÚÂÊÎÔÛÃÕÇ]+\b", text))
        words = [m.group(0) for m in word_matches]
        n_words = len(words)

        allcaps_tokens: list[str] = []
        for idx, match in enumerate(word_matches):
            w = match.group(0)
            if not w.isupper() or w in common_acronyms:
                continue

            if len(w) >= 3:
                allcaps_tokens.append(w)
            else:
                # Para palavras curtas (artigos/conjunções como A, E, O, DE, NO, etc.):
                # Considera ALL CAPS apenas quando NÃO estiver no início de frase ou após pontuação terminal (. ! ? \n …)
                if idx == 0:
                    is_sentence_start = True
                else:
                    prev_end = word_matches[idx - 1].end()
                    separator = text[prev_end : match.start()]
                    is_sentence_start = bool(re.search(r"[.!?…\n]", separator))

                if not is_sentence_start:
                    allcaps_tokens.append(w)

        allcaps_words_ratio = (len(allcaps_tokens) / n_words) if n_words > 0 else 0.0

        # 3. Sentenças para normalização de pontuação
        sentences = [s.strip() for s in re.split(r"(?:[.!?]+\s+|\n+)", text) if s.strip()]
        n_sentences = max(len(sentences), 1)

        # 4. Pontuação Excessiva (repetições de !!, ???, !?!, .....)
        excessive_punc_matches = re.findall(
            r"(!{2,}|\?{2,}|[!]{1,}[\?]{1,}|[\?]{1,}[!]{1,}|\.{3,})", text
        )
        excessive_punc = float(len(excessive_punc_matches))

        # 5. Densidade de Exclamações e Interrogações
        exclamation_density = text.count("!") / n_sentences
        question_density = text.count("?") / n_sentences

        # 6. Densidade de Léxico de Urgência, Clickbait e Golpes Comerciais (do repositório desacoplado)
        urgency_patterns = self.repository.get_urgency_patterns()
        urgency_matches_count = 0

        for _, pattern, _ in urgency_patterns:
            matches = pattern.findall(text)
            if matches:
                urgency_matches_count += len(matches)

        urgency_density = (urgency_matches_count / n_words) if n_words > 0 else 0.0

        # 7. Detecção de Formato de Diálogo Simulado / Atribuição de Entrevista Forjada
        dialogue_pattern = re.compile(
            r"(?:^[A-Z][a-záéíóúâêîôûãõç]+(?:\s+[A-Z][a-záéíóúâêîôûãõç]+)+\s*:\s*|\b[A-Z][a-záéíóúâêîôûãõç]+(?:\s+[A-Z][a-záéíóúâêîôûãõç]+)+\s*:\s*(?:boa noite|olá|em meus|confira|veja))",
            re.IGNORECASE | re.MULTILINE,
        )
        dialogue_matches = len(dialogue_pattern.findall(text))

        # 8. Escore Composto de Sensacionalismo e Risco Estilístico (Ponderação normalizada)
        raw_score = (
            0.15 * min(uppercase_ratio * 3.0, 1.0)
            + 0.20 * min(allcaps_words_ratio * 4.0, 1.0)
            + 0.15 * min(excessive_punc * 0.25, 1.0)
            + 0.10 * min(exclamation_density * 0.5, 1.0)
            + 0.25 * min(urgency_density * 5.0, 1.0)
            + 0.15 * min(dialogue_matches * 0.5, 1.0)
        )

        return {
            "uppercase_ratio": round(uppercase_ratio, 4),
            "allcaps_words_ratio": round(allcaps_words_ratio, 4),
            "excessive_punctuation_count": float(excessive_punc),
            "exclamation_density": round(exclamation_density, 4),
            "question_density": round(question_density, 4),
            "urgency_lexicon_density": round(urgency_density, 4),
            "composite_sensationalism_score": round(min(raw_score, 1.0), 4),
        }

    async def analyze(self, text: str, urls: list[str]) -> AnalyzerResult:
        """
        Executa a extração das features estilométricas e de sensacionalismo.
        NÃO emite veredito (verdict=None) nem confiança de decisão (confidence=0.0).
        Serve como base de apoio para análises mais completas e outros modelos.
        """
        metrics = self.extract_features(text)

        # Detecta diálogo simulado para enriquecer as razões e detalhes
        dialogue_pattern = re.compile(
            r"(?:^[A-Z][a-záéíóúâêîôûãõç]+(?:\s+[A-Z][a-záéíóúâêîôûãõç]+)+\s*:\s*|\b[A-Z][a-záéíóúâêîôûãõç]+(?:\s+[A-Z][a-záéíóúâêîôûãõç]+)+\s*:\s*(?:boa noite|olá|em meus|confira|veja))",
            re.IGNORECASE | re.MULTILINE,
        )
        dialogue_matches = len(dialogue_pattern.findall(text))

        reasons: list[str] = []
        if dialogue_matches > 0:
            reasons.append(
                "Presença de estrutura de diálogo/entrevista simulada (atribuição direta de falas a figuras públicas, formato frequente em golpes e publicidade fraudulenta)."
            )
        if metrics["allcaps_words_ratio"] > 0.15:
            reasons.append(
                f"Uso acentuado de palavras em caixa alta ({int(metrics['allcaps_words_ratio'] * 100)}%), indicando ênfase visual atípica."
            )
        if metrics["excessive_punctuation_count"] > 0:
            reasons.append(
                f"Presença de pontuação repetida/enfática ({int(metrics['excessive_punctuation_count'])} ocorrências)."
            )
        if metrics["urgency_lexicon_density"] > 0.0:
            reasons.append(
                "Detecção de termos apelativos, gatilhos de urgência ou fórmulas de apelo comercial/cura milagrosa."
            )
        if metrics["exclamation_density"] > 1.0:
            reasons.append(
                f"Densidade elevada de pontos de exclamação ({metrics['exclamation_density']} por sentença)."
            )

        if not reasons:
            reasons.append("Extração concluída: texto com padrões estilísticos neutros.")

        return AnalyzerResult(
            analyzer_name="heuristic",
            verdict=None,  # Nenhum veredito emitido (módulo puramente descritivo/feature extractor)
            confidence=0.0,
            reasons=reasons,
            sources=["Heurística Textual FactChkBR (Base: FACTCKBR)"],
            summary="Extração de métricas de sensacionalismo e estilometria para suporte a análises completas.",
            raw_details=metrics,
        )

