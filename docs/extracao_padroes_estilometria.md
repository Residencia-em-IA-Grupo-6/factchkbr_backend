# Extração de Padrões de Estilometria e Sensacionalismo

Este documento descreve a metodologia, o dataset e as formulações matemáticas utilizadas para extrair features estilométricas e léxicas de textos no analisador heurístico do **FactChkBR** (`app/analyzers/heuristic.py`).

---

## 1. Dataset Utilizado

A extração dos padrões foi calibrada utilizando dados empíricos de checagens reais de desinformação em língua portuguesa.

- **Repositório de Origem:** `FACTCK.BR` / `factckbr-extended`
- **Arquivo:** `FACTCKBR_updated_saude.tsv`
- **Tamanho da Amostra:** 1.364 alegações rotuladas como falsas ou enganosas (`falso`, `fake`, `errado`, `enganoso`, `distorcido`).
- **Campo Analisado:** `claimReviewed` (texto direto do boato, afirmação ou notícia falsa compartilhada em redes sociais e aplicativos de mensagens).

---


> **Nota de Arquitetura:** O `HeuristicAnalyzer` foi projetado para atuar **exclusivamente como extrator de features estilométricas e de sensacionalismo**, retornando `verdict = None` e `confidence = 0.0`. Ele não toma decisões categóricas isoladas; suas métricas e observações servem como base de apoio explicativo para modelos mais robustos (LLMs, classificadores BERTimbau e o ensemble).

## 2. Métricas Extraídas

O método `HeuristicAnalyzer.extract_features(text: str)` decompõe o texto nas 7 métricas padronizadas:

```python
{
    "uppercase_ratio": round(uppercase_ratio, 4),
    "allcaps_words_ratio": round(allcaps_words_ratio, 4),
    "excessive_punctuation_count": float(excessive_punc),
    "exclamation_density": round(exclamation_density, 4),
    "question_density": round(question_density, 4),
    "urgency_lexicon_density": round(urgency_density, 4),
    "composite_sensationalism_score": round(min(raw_score, 1.0), 4)
}
```

---

## 3. Metodologia e Formulação Matemática

### 3.1. Densidade de Uppercase (`uppercase_ratio`)
Mede a proporção de letras maiúsculas em relação ao total de caracteres alfabéticos no texto:
$$\text{uppercase\_ratio} = \frac{N_{\text{upper}}}{N_{\text{letters}}}$$
- Neutraliza o comprimento do texto.
- Se $N_{\text{letters}} = 0$, retorna `0.0`.

### 3.2. Proporção de Palavras em ALL CAPS (`allcaps_words_ratio`)
Mede o uso de palavras inteiras em caixa alta em relação ao total de palavras:
$$\text{allcaps\_words\_ratio} = \frac{N_{\text{allcaps}}}{N_{\text{words}}}$$

- **Critérios de Elegibilidade de Tokens:**
  1. **Palavras com comprimento $\ge 3$ caracteres:** São computadas como ALL CAPS se todas as letras forem maiúsculas e o termo não for uma sigla catalogada no repositório.
  2. **Artigos, conjunções e palavras curtas ($< 3$ caracteres, ex.: `A`, `E`, `O`, `DE`, `NO`):** 
     - **Início de frase ou pós-pontuação terminal (`.`, `!`, `?`, `\n`, `…`):** **Ignoradas** na contagem de ALL CAPS. Isso preserva a conformidade gramatical padrão da língua portuguesa (capitalização compulsória na abertura de períodos) e anula falsos positivos em notícias neutras (ex.: *"A Anvisa aprovou a vacina. O médico confirmou."*).
     - **Meio de frase:** **Computadas** como ALL CAPS caso estejam inteiramente em maiúsculas (ex.: em *"VACINA MATA E ALTERA O DNA"*, `E` e `O` são contabilizados como grito visual / ênfase sensacionalista).
- **Lista de Exclusão de Siglas Legítimas:** `STF`, `STJ`, `TSE`, `SUS`, `PIB`, `DF`, `EUA`, `OMS`, `ONU`, `ANVISA`, `FIOCRUZ`, `USP`, `CPI`, `UTI`, `DNA`, etc. Evita que menções a órgãos públicos, estados ou termos científicos inflem a métrica.

### 3.3. Contagem de Pontuação Excessiva (`excessive_punctuation_count`)
Mede a quantidade de sequências atípicas de pontuação repetida, expressas pela expressão regular:
$$\verb|(!{2,}|\?{2,}|[!]{1,}[\?]{1,}|[\?]{1,}[!]{1,}|\.{3,})|$$
- Captura ocorrências como `!!`, `???`, `!?!`, `.....`.
- Retornado como valor numérico (`float`).

### 3.4. Densidade de Exclamações (`exclamation_density`)
Frequência média de pontos de exclamação por sentença:
$$\text{exclamation\_density} = \frac{N_{!}}{N_{\text{sentences}}}$$
- A contagem de sentenças é obtida segmentando o texto por quebras de linha e delimitadores pontuais (`.`, `!`, `?`).

### 3.5. Densidade de Interrogações (`question_density`)
Frequência média de pontos de interrogação por sentença:
$$\text{question\_density} = \frac{N_{?}}{N_{\text{sentences}}}$$
- Utilizada para capturar títulos conspiratórios ou indutivos (ex: *"Você sabia que...?"*, *"Estariam escondendo...?"*).

### 3.6. Densidade de Léxico de Urgência e Clickbait (`urgency_lexicon_density`)
Proporção de termos apelativos e gatilhos de compartilhamento em relação ao total de palavras:
$$\text{urgency\_lexicon\_density} = \frac{N_{\text{urgency\_matches}}}{N_{\text{words}}}$$

O vocabulário foi expandido com base nas expressões mais frequentes no `FACTCKBR_updated_saude.tsv`:
- **Gatilhos de urgência e contágio:** `"urgente"`, `"compartilhe"`, `"repasse para todos"`, `"a mídia esconde"`, `"bomba"`, `"veja antes que apaguem"`, `"não vai passar na tv"`.
- **Sensacionalismo em saúde:** `"cura secreta"`, `"cura do câncer"`, `"altera dna"`, `"vacina mata"`, `"cloroquina cura"`, `"anvisa proibiu ou esconde"`, `"chip/grafeno"`.

### 3.7. Escore Composto de Sensacionalismo (`composite_sensationalism_score`)
Combinação linear normalizada ponderando as cinco dimensões de anomalia estilística:

$$\text{raw\_score} = 0.20 \cdot \min(3 \cdot \text{upper}, 1.0) + 0.25 \cdot \min(4 \cdot \text{allcaps}, 1.0) + 0.20 \cdot \min(0.25 \cdot \text{punc}, 1.0) + 0.15 \cdot \min(0.5 \cdot \text{excl}, 1.0) + 0.20 \cdot \min(5 \cdot \text{urgency}, 1.0)$$

$$\text{composite\_sensationalism\_score} = \min(\text{raw\_score}, 1.0)$$

---

## 4. Distribuição Empírica Observada no Dataset

Abaixo, a síntese das estatísticas calculadas sobre as **1.364 notícias falsas** de saúde do dataset `FACTCKBR_updated_saude.tsv`:

| Métrica | Média | Mediana | Percentil 90 ($p_{90}$) | Máximo |
| :--- | :---: | :---: | :---: | :---: |
| **Densidade de Uppercase** | 0.0501 | 0.0361 | 0.0849 | 1.0000 |
| **Densidade de Exclamações** | 0.0376 | 0.0000 | 0.0000 | 5.0000 |
| **Pontuação Repetida (`...`, `!!`)** | Presente em 6.3% das alegações | 0 | 0 | 4 ocorrências |
| **Tokens em ALL CAPS** | Predomínio de siglas médicas e alarmes (`DNA`, `RNA`, `UTI`, `VIRUS`, `CORONAVIRUS`) | 0 | 1 | 8 tokens |

---

## 5. Implementação no Código

O extrator e o classificador residem diretamente em:
- **Arquivo:** [`app/analyzers/heuristic.py`](file:///Users/aluno1/repos/factchkbr_backend/app/analyzers/heuristic.py)
- **Classe:** `HeuristicAnalyzer(BaseAnalyzer)`
- **Decorator de Registro:** `@register_analyzer("heuristic", weight=0.6)`

O resultado da extração alimenta a decisão do veredito (`SUSPEITO`, `FAKE`, `INCONCLUSIVO`) e é retornado no campo `raw_details` do objeto `AnalyzerResult` para auditoria do Bot do Telegram.
