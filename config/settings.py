from typing import List

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Environment
    KANG_ENV: str = "local"

    # PostgreSQL Individual Settings
    postgres_db: str
    postgres_user: str
    postgres_password: str
    postgres_host: str = "localhost"  # Default for local development
    postgres_port: int = 5432

    # Test database settings
    test_postgres_db: str
    test_postgres_user: str
    test_postgres_password: str
    test_postgres_host: str = "localhost"  # Usually localhost for tests
    test_postgres_port: int = 5432

    # Redis
    redis_url: str
    redis_auth_db: int = 1
    redis_auth_minsize: int = 1
    redis_auth_maxsize: int = 10

    # JWT
    jwt_secret_key: str
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = 30

    # Google OAuth
    google_client_id: str
    google_client_secret: str
    google_redirect_uri: str

    # CORS
    allowed_origins: str = "http://localhost:3000,http://localhost:8080"

    # LangSmith Configuration
    langchain_tracing_v2: bool = True
    langchain_endpoint: str = "https://api.smith.langchain.com"
    langchain_api_key: str = ""
    langchain_project: str = "kang-rag-system"

    # AI Model Configuration
    # GPT-OSS Configuration (Primary)
    gpt_oss_base_url: str = "http://localhost:11434"
    gpt_oss_model: str = "llama3.2:3b"
    gpt_oss_max_tokens: int = 4096
    gpt_oss_temperature: float = 0.1
    gpt_oss_reasoning_level: str = "medium"
    gpt_oss_timeout: int = 120

    # Embedding settings
    embedding_model: str = "jhgan/ko-sroberta-multitask"
    embedding_dimension: int = 768
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # Vector search settings
    similarity_threshold: float = 0.7
    max_retrieved_docs: int = 5

    # File upload settings
    max_file_size_mb: int = 10
    allowed_file_types: str = "pdf,docx,txt"
    upload_dir: str = "uploads/documents/"

    def is_prod(self) -> bool:
        return self.KANG_ENV.lower() == "prod"

    @property
    def allowed_origins_list(self) -> List[str]:
        return [origin.strip() for origin in self.allowed_origins.split(",")]

    # PostgreSQL URL properties - dynamically constructed from individual settings
    @property
    def postgres_url(self) -> str:
        return f"postgresql://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"

    @property
    def postgres_read_url(self) -> str:
        return self.postgres_url  # Can be customized later for read replicas

    @property
    def postgres_write_url(self) -> str:
        return self.postgres_url  # Can be customized later for write masters

    @property
    def test_postgres_url(self) -> str:
        return f"postgresql://{self.test_postgres_user}:{self.test_postgres_password}@{self.test_postgres_host}:{self.test_postgres_port}/{self.test_postgres_db}"

    # Domain-specific PostgreSQL URLs (using same base configuration)
    @property
    def user_postgres_url(self) -> str:
        return self.postgres_url

    @property
    def user_postgres_read_url(self) -> str:
        return self.postgres_read_url

    @property
    def user_postgres_write_url(self) -> str:
        return self.postgres_write_url

    # File upload helpers
    @property
    def allowed_file_types_list(self) -> List[str]:
        return [ext.strip() for ext in self.allowed_file_types.split(",")]

    @property
    def max_file_size_bytes(self) -> int:
        return self.max_file_size_mb * 1024 * 1024

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()
