import os
from functools import lru_cache

from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class GoogleApiConfig(BaseModel):
    gemini_api_key: str


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_nested_delimiter="__")

    gemini_api_key: str
    fal_api_key: str


def get_runtime_environment() -> str:
    return os.environ.get("RUNTIME_ENV") or "dev"


@lru_cache
def get_settings() -> Settings:
    config_path = os.path.abspath(__file__)
    app_root = os.path.dirname(config_path)
    proj_root = os.path.abspath(os.path.join(app_root, os.pardir))

    runtime_env = get_runtime_environment()

    if runtime_env == "prod":
        env_file = (".env", ".env.prod")
    elif runtime_env == "stage":
        env_file = (".env", ".env.stage")
    else:
        env_file = (".env", ".env.dev")

    env_file = [os.path.join(proj_root, e) for e in env_file]
    return Settings(_env_file=env_file, _env_file_encoding="utf-8")


def set_google_api_key():
    # for google gemini use
    cfg = get_settings()
    os.environ["GOOGLE_API_KEY"] = cfg.gemini_api_key


def set_fal_api_key():
    cfg = get_settings()
    os.environ["FAL_KEY"] = cfg.fal_api_key
