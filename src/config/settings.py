from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    OPENAI_API_KEY: str = Field(..., description="OpenAI API Key")

    # Class Config to load environment variables from .env file
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "allow"


# Instantiate the settings object
settings = Settings()
