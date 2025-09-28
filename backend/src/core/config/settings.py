from pydantic import BaseModel
from pydantic_settings import BaseSettings

class RunConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000

class ApiPrefix(BaseModel):
    tools: str = "/tools"

class Settings(BaseSettings):
    run: RunConfig = RunConfig()
    api: ApiPrefix = ApiPrefix()

settings = Settings()