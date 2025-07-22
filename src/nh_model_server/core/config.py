from pathlib import Path
from pydantic_settings import BaseSettings

ROOT_DIR = Path(__file__).parent.parent.parent.parent

class Settings(BaseSettings):
    # Server configuration
    APP_NAME: str = 'NH Model Server'
    APP_VERSION: str = '0.1.0'
    APP_DESCRIPTION: str = 'Model Server for NH'
    DEBUG: bool = True
    SERVER_PORT: int = 6543
    SERVER_HOST: str = '0.0.0.0'
        
    # Memory temp directory
    MEMORY_TEMP_DIR: str | None = None
    PRE_REMOVE_MEMORY_TEMP_DIR: bool = False

    # Proxy configuration
    HTTP_PROXY: str
    HTTPS_PROXY: str
    
    # Treeger meta configuration
    TREEGER_SERVER_ADDRESS: str = 'thread://gridman_bstreeger'
    SCENARIO_META_PATH: str = str(ROOT_DIR / 'scenario.meta.yaml')

    # Model configuration
    RESOURCE_PATH: str = str(ROOT_DIR / 'resource')
    MODEL_PATH: str = 'model'

    # Resource server proxy address
    RESOURCE_SERVER_PROXY_ADDRESS: str = 'http://localhost:9000/api/proxy/discover'

    # DB configuration
    PERSISTENCE_PATH: str = str(ROOT_DIR / 'persistence')

    # Solution related constants
    SOLUTION_DIR: str = 'resource/solutions/'

    # Simulation related constants
    SIMULATION_DIR: str = 'resource/simulations/'

    # AI MCP configuration
    DEEPSEEK_API_KEY: str
    ANTHROPIC_API_KEY: str

    # CORS
    CORS_ORIGINS: list[str] = ['*']
    CORS_HEADERS: list[str] = ['*']
    CORS_METHODS: list[str] = ['*']
    CORS_CREDENTIALS: bool = True

    class Config:
        env_file = '.env'

settings = Settings()
