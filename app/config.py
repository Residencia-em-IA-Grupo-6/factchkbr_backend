from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Configurações centrais da aplicação carregadas de variáveis de ambiente ou .env.
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    APP_NAME: str = "FactChkBR API"
    APP_VERSION: str = "1.0.0"
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    PORT: int = 8000
    HOST: str = "0.0.0.0"

    # Lista de analisadores ativos separados por vírgula
    # Ex: "claim_extractor,heuristic,fact_check_api,llm_judge"
    ACTIVE_ANALYZERS: str = "claim_extractor,heuristic,fact_check_api,llm_judge"


    # Configuração de Provedores de LLM (Ollama Local ou OpenAI)
    # Padrão: "ollama" com "qwen3.5:9b" para execução local gratuita
    LLM_PROVIDER: str = "ollama"  # "ollama" | "openai"
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "qwen3.5:9b"

    # Chaves e integrações com APIs externas (caso LLM_PROVIDER="openai")
    GOOGLE_FACTCHECK_API_KEY: str | None = None
    OPENAI_API_KEY: str | None = None
    OPENAI_MODEL: str = "gpt-4o-mini"
    OPENAI_BASE_URL: str = "https://api.openai.com/v1"

    def get_llm_endpoint(self) -> str:
        """Retorna a URL de chat completions compatível com OpenAI."""
        if self.LLM_PROVIDER.lower() == "ollama":
            base = self.OLLAMA_BASE_URL.rstrip("/")
            return f"{base}/v1/chat/completions"
        return f"{self.OPENAI_BASE_URL.rstrip('/')}/chat/completions"

    def get_llm_model(self) -> str:
        """Retorna o identificador do modelo para o provedor ativo."""
        if self.LLM_PROVIDER.lower() == "ollama":
            return self.OLLAMA_MODEL
        return self.OPENAI_MODEL

    def get_llm_headers(self) -> dict[str, str]:
        """Retorna os headers HTTP para autenticação e formato."""
        headers = {"Content-Type": "application/json"}
        if self.LLM_PROVIDER.lower() == "openai" and self.OPENAI_API_KEY:
            headers["Authorization"] = f"Bearer {self.OPENAI_API_KEY}"
        elif self.LLM_PROVIDER.lower() == "ollama":
            headers["Authorization"] = "Bearer ollama"
        return headers

    def get_active_analyzers_list(self) -> list[str]:
        """Retorna os nomes dos analisadores ativos como lista."""
        return [item.strip() for item in self.ACTIVE_ANALYZERS.split(",") if item.strip()]



@lru_cache()
def get_settings() -> Settings:
    """Retorna instância singleton das configurações."""
    return Settings()
