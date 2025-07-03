from pathlib import Path
from pydantic_settings import BaseSettings

ROOT_DIR = Path(__file__).parent.parent.parent.parent

class Settings(BaseSettings):
    # Server configuration
    APP_NAME: str = 'NH Model Server'
    APP_VERSION: str = '0.1.0'
    APP_DESCRIPTION: str = 'Model Server for NH'
    DEBUG: bool = True
    SERVER_PORT: int = 6000
    SERVER_HOST: str = '0.0.0.0'
        
    # Model configuration
    RESOURCE_PATH: str = str(ROOT_DIR / 'resource')

    # Resource server proxy address
    RESOURCE_SERVER_PROXY_ADDRESS: str = 'http://localhost:9000/api/proxy/discover'

    # CORS
    CORS_ORIGINS: list[str] = ['*']
    CORS_HEADERS: list[str] = ['*']
    CORS_METHODS: list[str] = ['*']
    CORS_CREDENTIALS: bool = True

settings = Settings()
