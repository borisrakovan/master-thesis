from pydantic_settings import BaseSettings, SettingsConfigDict

from src.llm.enums import ChatModel


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file="./.env", env_file_encoding="utf-8")

    # OpenAI
    openai_api_base: str = "https://api.openai.com/v1"
    openai_api_key: str

    anyscale_api_base: str = "https://api.endpoints.anyscale.com/v1"
    anyscale_api_key: str

    llm_temperature: float = 0.0
    llm_max_tokens: int = 1000
    llm_model: ChatModel = ChatModel.GPT_35
    llm_max_retries: int = 3
    llm_timeout: int = 10

    accuracy_confidence_interval_level: float = 0.95


settings = Settings()  # type: ignore
