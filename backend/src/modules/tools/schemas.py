from pydantic import BaseModel
from typing import List, Optional, Dict

class DetectionItem(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    bbox: List[float]

class AnalysisResult(BaseModel):
    status: str
    total_detections: int
    expected_count: int = 11
    missing_tools: List[str] = []
    extra_tools: List[str] = []
    detected_tools: List[str] = []
    detections: List[DetectionItem]
    message: str

class AnalysisConfig(BaseModel):
    confidence_threshold: float
    iou_threshold: float
    annotated_image_path: Optional[str] = None
    output_directory: Optional[str] = None
    total_annotated_images: Optional[int] = None

class SingleAnalysisResponse(BaseModel):
    status: str
    analysis_result: AnalysisResult
    config: AnalysisConfig

class ImageAnalysisResult(BaseModel):
    filename: str
    analysis_result: AnalysisResult
    annotated_image_path: Optional[str] = None

class BatchAnalysisResponse(BaseModel):
    status: str
    total_images: int
    processed_images: int
    results: List[ImageAnalysisResult]
    processing_time: float
    summary: Dict[str, int]
    config: Optional[AnalysisConfig] = None