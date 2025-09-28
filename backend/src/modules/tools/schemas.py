from pydantic import BaseModel
from typing import List, Optional

class DetectionItem(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    bbox: List[float]  # [x1, y1, x2, y2]

class AnalysisConfig(BaseModel):
    confidence_threshold: float
    iou_threshold: float
    annotated_image_path: Optional[str] = None
    output_directory: Optional[str] = None
    total_annotated_images: Optional[int] = None

class SingleAnalysisResponse(BaseModel):
    status: str
    total_detections: int
    detections: List[DetectionItem]
    message: Optional[str] = None
    config: AnalysisConfig

class ImageAnalysisResult(BaseModel):
    filename: str
    status: str
    total_detections: int
    detections: List[DetectionItem]
    message: Optional[str] = None
    annotated_image_path: Optional[str] = None

class BatchAnalysisResponse(BaseModel):
    status: str
    total_images: int
    processed_images: int
    total_detections: int
    results: List[ImageAnalysisResult]
    processing_time: float
    message: Optional[str] = None
    config: Optional[AnalysisConfig] = None