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

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()
