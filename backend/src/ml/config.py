from pydantic_settings import BaseSettings

class MLConfig(BaseSettings):
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.5
    
    class Config:
        env_prefix = "ML_"

ml_config = MLConfig()