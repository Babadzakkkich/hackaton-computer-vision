from pydantic_settings import BaseSettings

class MLConfig(BaseSettings):
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    
    class Config:
        env_prefix = "ML_"

ml_config = MLConfig()