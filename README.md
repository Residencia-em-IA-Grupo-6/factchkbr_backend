# FactChkBR - Back-end de Detecção de Fake News e Desinformação

Back-end assíncrono em **FastAPI (Python 3.11+)** estruturado sob o padrão **Strategy / Pipeline Plug-and-Play (Registry Pattern)**, projetado para integração direta com Bot do Telegram para verificação de fatos e desinformação.

---

## 📂 Estrutura do Repositório

```
factchkbr_backend/
├── app/
│   ├── __init__.py
│   ├── main.py                  # Inicialização do FastAPI, CORS e inclusão de rotas
│   ├── config.py                # Configurações de ambiente (Pydantic Settings / .env)
│   ├── schemas/
│   │   ├── __init__.py
│   │   └── analysis.py          # Schemas Pydantic de Request e Response (Contrato Telegram)
│   ├── api/
│   │   ├── __init__.py
│   │   └── v1/
│   │       ├── __init__.py
│   │       ├── router.py        # Agrupador de rotas
│   │       └── endpoints.py     # Rota POST /api/analyze e GET /health
│   ├── core/
│   │   ├── __init__.py
│   │   ├── base.py              # Interface abstrata BaseAnalyzer (ABC)
│   │   ├── registry.py          # Registro dinâmico de analisadores disponíveis (@register_analyzer)
│   │   └── orchestrator.py      # Orquestrador/Ensemble que executa os modelos e consolida o veredito
│   └── analyzers/               # Pasta onde algoritmos e léxicos residem
│       ├── __init__.py
│       ├── heuristic.py         # Analisador 1: Regras e Heurísticas de texto (clickbait, sensacionalismo)
│       ├── fact_check_api.py    # Analisador 2: Consulta a bases externas / APIs de checagem
│       ├── llm_judge.py         # Analisador 3: Avaliador via LLM / Modelo de NLP
│       ├── lexicon_repository.py # Repositório desacoplado de léxicos (preparado para Banco de Dados)
│       └── data/                # Dados e dicionários desacoplados
│           ├── acronyms.json    # Acrônimos legítimos categorizados
│           └── urgency_patterns.json # Padrões regex de urgência com severidade
├── docs/
│   ├── extracao_padroes_estilometria.md # Documentação da extração empírica de padrões
│   └── arquitetura_banco_padroes.md     # Modelagem e arquitetura de Banco de Dados escalável
├── tests/
│   ├── __init__.py
│   ├── test_api.py              # Testes dos endpoints e conformidade de contrato
│   └── test_analyzers.py        # Testes de herança, registro e orquestração
├── .env.example
├── .gitignore
├── .dockerignore
├── Dockerfile
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

---

## 🔌 Como Adicionar um Novo Modelo / Analisador (Plug-and-Play)

Graças ao padrão **Registry Pattern**, adicionar um novo algoritmo (BERTimbau, modelo fine-tuned, busca vetorial/RAG, nova API) **não exige alterar o código central ou as rotas da API**.

### Passo a Passo:

1. Crie um novo arquivo dentro de `app/analyzers/` (ex: `app/analyzers/bert_classifier.py`).
2. Herde de `BaseAnalyzer` e decore com `@register_analyzer("nome_do_modelo", weight=1.0)`:

```python
from app.core.base import BaseAnalyzer
from app.core.registry import register_analyzer
from app.schemas.analysis import AnalyzerResult, Verdict

@register_analyzer("bert_classifier", weight=1.2)
class BertClassifierAnalyzer(BaseAnalyzer):
    async def analyze(self, text: str, urls: list[str]) -> AnalyzerResult:
        # Sua inferência aqui (ex: Hugging Face transformers, ONNX, etc.)
        return AnalyzerResult(
            analyzer_name="bert_classifier",
            verdict=Verdict.FAKE,
            confidence=0.94,
            reasons=["Padrão de desinformação detectado pelo classificador BERT."],
            sources=["BERTimbau Fine-Tuning v1"]
        )
```

3. Exporte o novo analisador em `app/analyzers/__init__.py`.
4. Ative o novo analisador no `.env`:
```env
ACTIVE_ANALYZERS="heuristic,fact_check_api,llm_judge,bert_classifier"
```

O `FactCheckOrchestrator` carregará e executará automaticamente o novo modelo em paralelo via `asyncio.gather`.

---

## 🎯 Contrato da API (Compatível com Bot do Telegram)

### `POST /api/analyze`

**Payload de Entrada (JSON):**
```json
{
  "text": "Texto completo da mensagem ou fato a ser verificado",
  "urls": ["https://exemplo.com/noticia-opcional"],
  "user_id": 123456789,
  "chat_id": 987654321
}
```

**Payload de Saída (JSON):**
```json
{
  "claim": "Resumo objetivo da afirmação/fato verificado",
  "verdict": "FAKE",
  "confidence": 0.92,
  "summary": "Resumo contextualizado da análise realizada pelos modelos",
  "reasons": [
    "Motivo 1 apontado pelos modelos",
    "Motivo 2 apontado pelos modelos"
  ],
  "sources": [
    "Agência Lupa",
    "Aos Fatos"
  ]
}
```

> **Vereditos permitidos:** `VERDADEIRO`, `FAKE`, `SUSPEITO`, `INCONCLUSIVO`.

### `GET /health` e `GET /api/health`

Retorna o status da aplicação e a lista de analisadores ativos e disponíveis:
```json
{
  "status": "ok",
  "version": "1.0.0",
  "active_analyzers": ["heuristic", "fact_check_api", "llm_judge"],
  "available_analyzers": ["fact_check_api", "heuristic", "llm_judge"]
}
```

---

## 🚀 Como Executar

### 1. Ambiente Local com Python

```bash
# Criar e ativar ambiente virtual
python3 -m venv .venv
source .venv/bin/activate

# Instalar dependências
pip install -r requirements.txt

# Configurar variáveis de ambiente
cp .env.example .env

# Iniciar servidor FastAPI
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Acesse a documentação interativa em: [http://localhost:8000/docs](http://localhost:8000/docs).

### 2. Executar Testes

```bash
pip install -r requirements-dev.txt
pytest -v
```

### 3. Execução via Docker

```bash
# Construir imagem
docker build -t factchkbr-backend .

# Executar container
docker run -p 8000:8000 --env-file .env factchkbr-backend
```
