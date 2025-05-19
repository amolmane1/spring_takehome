from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    reference_data_path: str = "data/reference_data.json"
    image_size: int = 160
    margin: int = 0

    class Config:
        env_file = ".env"

settings = Settings()