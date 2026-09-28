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


    # Chaves e integrações com APIs externas
    GOOGLE_FACTCHECK_API_KEY: str | None = None
    OPENAI_API_KEY: str | None = None
    OPENAI_MODEL: str = "gpt-4o-mini"
    OPENAI_BASE_URL: str = "https://api.openai.com/v1"

    def get_active_analyzers_list(self) -> list[str]:
        """Retorna os nomes dos analisadores ativos como lista."""
        return [item.strip() for item in self.ACTIVE_ANALYZERS.split(",") if item.strip()]


@lru_cache()
def get_settings() -> Settings:
    """Retorna instância singleton das configurações."""
    return Settings()
