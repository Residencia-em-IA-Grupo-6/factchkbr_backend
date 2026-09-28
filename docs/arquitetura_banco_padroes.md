# Arquitetura e Modelagem de Banco de Dados Escalável para Padrões e Acrônimos

Este documento detalha o desacoplamento dos padrões léxicos e acrônimos do analisador heurístico (`HeuristicAnalyzer`), bem como a proposta de arquitetura de banco de dados escalável para suportar atualizações dinâmicas, controle de versões e curadoria colaborativa contínua.

---

## 1. Justificativa do Desacoplamento

Manter termos de urgência e siglas *hardcoded* no código-fonte introduz gargalos operacionais:
1. **Necessidade de Deploy para Atualizações:** A emergência de novas narrativas de desinformação (ex.: novas falsas curas ou termos virais) exigiria novos deploys de código.
2. **Incapacidade de Personalização por Domínio:** Diferentes domínios temáticos (saúde, política, eleições, finanças) possuem vocabulários específicos de urgência e conjuntos distintos de siglas institucionais.
3. **Falta de Rastreabilidade e Auditoria:** Não é possível rastrear quem cadastrou um novo padrão, quando e com qual severidade.

Com o desacoplamento implementado via [`BaseLexiconRepository`](file:///Users/aluno1/repos/factchkbr_backend/app/analyzers/lexicon_repository.py), o analisador heurístico opera sobre uma interface abstrata que hoje consome arquivos JSON estruturados e está preparada para conexão direta a um banco de dados relacional (PostgreSQL) ou distribuído.

---

## 2. Estrutura Atual Desacoplada

Os dados foram migrados para a pasta de dados do módulo de analisadores:
- [`app/analyzers/data/acronyms.json`](file:///Users/aluno1/repos/factchkbr_backend/app/analyzers/data/acronyms.json): Siglas organizadas por categoria (judiciário, saúde, economia, segurança, política, internacional).
- [`app/analyzers/data/urgency_patterns.json`](file:///Users/aluno1/repos/factchkbr_backend/app/analyzers/data/urgency_patterns.json): Padrões de expressões regulares com metadados de severidade, categoria e origem no dataset `FACTCKBR`.

---

## 3. Esquema Recomendado de Banco de Dados Relacional (PostgreSQL)

Para migração para um banco escalável (ex: PostgreSQL / Supabase / AWS RDS), propõe-se o seguinte esquema relacional normalizado:

```sql
-- Tabela de Categorias Temáticas
CREATE TABLE pattern_categories (
    id VARCHAR(50) PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    description TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Tabela de Acrônimos e Siglas Legítimas
CREATE TABLE acronyms (
    id SERIAL PRIMARY KEY,
    acronym VARCHAR(20) NOT NULL,
    category_id VARCHAR(50) REFERENCES pattern_categories(id),
    description VARCHAR(255),
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE UNIQUE INDEX idx_acronyms_unique_upper ON acronyms (UPPER(acronym));
CREATE INDEX idx_acronyms_category ON acronyms (category_id);

-- Tabela de Padrões Léxicos de Urgência e Clickbait
CREATE TABLE urgency_patterns (
    id VARCHAR(50) PRIMARY KEY,
    label VARCHAR(100) NOT NULL,
    regex_pattern TEXT NOT NULL,
    category_id VARCHAR(50) REFERENCES pattern_categories(id),
    severity NUMERIC(3,2) NOT NULL DEFAULT 1.00 CHECK (severity >= 0.0 AND severity <= 1.0),
    source VARCHAR(100),
    description TEXT,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_patterns_active ON urgency_patterns (is_active);
CREATE INDEX idx_patterns_category ON urgency_patterns (category_id);
```

---

## 4. Estratégia de Cache e Baixa Latência (Redis / In-Memory)

Para atender a alta demanda do Bot do Telegram sem sobrecarregar o banco de dados relacional:

```
[ Telegram Bot ]
       │
       ▼
[ FastAPI / HeuristicAnalyzer ]
       │
       ▼
[ LexiconRepository ]
  ├── 1. Cache Local em Memória (LRU / Compilação re.Pattern O(1))
  ├── 2. Camada Distribuída (Redis - Cache Invalidation via Pub/Sub)
  └── 3. Banco Persistente (PostgreSQL - Fonte da Verdade)
```

1. **Compilação em Inicialização:** Expressões regulares são compiladas em memória (`re.compile`) na inicialização do serviço.
2. **Invalidação de Cache via Webhook / Pub-Sub:** Quando um checador adiciona um novo padrão via painel administrativo, uma mensagem Redis ou sinal HTTP aciona o método `repository.reload()`, atualizando os workers sem necessidade de *restart* ou *downtime*.

---

## 5. Pipeline de Aprendizado Contínuo

Com o banco de dados ativo:
1. **Feedback Loop:** Alegações classificadas como `FAKE` pelo ensemble podem passar por minerador de n-grams para sugerir novos termos apelativos aos curadores.
2. **Ajuste Fino de Severidade:** O peso (`severity`) de cada padrão pode ser calibrado com base no ganho de informação (Information Gain) medido no corpus factual do FactChkBR.
