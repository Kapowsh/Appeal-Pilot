from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", protected_namespaces=())

    nebius_api_key: str = ""
    nebius_base_url: str = "https://api.tokenfactory.nebius.com/v1/"
    model_fast: str = ""
    model_reason: str = ""
    model_embed: str = ""
    tavily_api_key: str = ""
    enable_tavily: bool = False
    max_runs_per_ip_per_hour: int = 10
    trust_proxy: bool = False  # honor X-Forwarded-For only behind a known reverse proxy
    data_dir: Path = Path("data")
    cache_dir: Path = Path("cache")


@lru_cache
def get_settings() -> Settings:
    return Settings()
