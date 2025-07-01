import uvicorn
from src.nh_model_server.core.config import settings

if __name__ == "__main__":
    uvicorn.run("src.nh_model_server.main:app", host=settings.SERVER_HOST, port=settings.SERVER_PORT, reload=True)
    
