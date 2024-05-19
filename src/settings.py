from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.llm.enums import ChatModel

# Load google application credentials
load_dotenv()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file="./.env", env_file_encoding="utf-8")

    # OpenAI
    openai_api_base: str = "https://api.openai.com/v1"
    openai_api_key: str

    anyscale_api_base: str = "https://api.endpoints.anyscale.com/v1"
    anyscale_api_key: str

    google_application_credentials: str
    vertexai_project_id: str = "famous-rhythm-417205"
    vertexai_location: str = "us-central1"
    # vertexai_project_id: str = "valid-volt-411420"
    # vertexai_location: str = "us-central1"
    # vertexai_location: str = "europe-cntral2"
    # vertexai_location: str = "europe-west2"
    # vertexai_locations = [
    #     "us-central1", "europe-west2", "europe-central2", "europe-west8", "us-west4", "europe-north1", "europe-west3",
    #     "us-west1", "us-west2"
    # ]

    llm_temperature: float = 0.0
    llm_max_tokens: int = 1000
    llm_model: ChatModel = ChatModel.GPT_35
    llm_max_retries: int = 16
    llm_timeout: int = 15

    accuracy_confidence_interval_level: float = 0.95


settings = Settings()  # type: ignore
