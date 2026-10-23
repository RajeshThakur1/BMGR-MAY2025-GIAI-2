from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    liquid_api_key: str = ""
    decision_provider: str = "openai"  # mock | openai | liquid

    openai_api_key: str = ""
    decision_model: str = "gpt-6-luna"  # OpenAI Decisions API
    generation_model: str = "gpt-4o-mini"

    model_server_url: str = "http://127.0.0.1:8001"
    model_server_api_key: str = "dev-local-key"

    backend_host: str = "127.0.0.1"
    backend_port: int = 8000
    data_dir: Path = Path("./data")

    text_relevance_threshold: float = 0.55
    text_answerable_threshold: float = 0.40
    image_relevance_threshold: float = 0.50
    image_answerable_threshold: float = 0.50

    embedding_dim: int = 384  # all-MiniLM-L6-v2 (local default; PDF uses 768 with EmbeddingGemma 2)

    company_id: str = "demo-company"


settings = Settings()
