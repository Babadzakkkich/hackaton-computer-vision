from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from .service import tool_service
from .schemas import SingleAnalysisResponse, BatchAnalysisResponse, AnalysisConfig

router = APIRouter()

@router.post("/analyze", response_model=SingleAnalysisResponse)
async def analyze_single_image(
    file: UploadFile = File(..., description="Изображение с инструментами"),
    confidence: float = Query(0.25, ge=0.0, le=1.0, description="Порог уверенности (0.0-1.0)"),
    iou: float = Query(0.45, ge=0.0, le=1.0, description="Порог пересечения (0.0-1.0)"),
    save_annotated: bool = Query(True, description="Сохранять аннотированное изображение")
):
    """
    Анализирует одно изображение с инструментами с помощью YOLO11 модели.
    Сохраняет изображение с bounding boxes в папку results/.
    """
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Файл должен быть изображением")
    
    try:
        image_data = await file.read()
        
        if len(image_data) == 0:
            raise HTTPException(status_code=400, detail="Файл пустой")
        
        result = tool_service.analyze_single_image(
            image_data, 
            file.filename, 
            confidence_threshold=confidence,
            iou_threshold=iou,
            save_annotated=save_annotated
        )
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при обработке изображения: {str(e)}")

@router.post("/analyze-batch", response_model=BatchAnalysisResponse)
async def analyze_batch_images(
    file: UploadFile = File(..., description="ZIP-архив с изображениями"),
    max_images: int = Query(100, ge=1, le=1000),
    confidence: float = Query(0.25, ge=0.0, le=1.0, description="Порог уверенности (0.0-1.0)"),
    iou: float = Query(0.45, ge=0.0, le=1.0, description="Порог пересечения (0.0-1.0)"),
    save_annotated: bool = Query(True, description="Сохранять аннотированные изображения")
):
    """
    Анализирует batch изображений из ZIP-архива с помощью YOLO11 модели.
    Сохраняет все аннотированные изображения в папку results/batch_session_YYYYMMDD_HHMMSS/.
    """
    if not file.filename or not file.filename.lower().endswith('.zip'):
        raise HTTPException(status_code=400, detail="Файл должен быть ZIP-архивом")
    
    try:
        zip_data = await file.read()
        
        if len(zip_data) == 0:
            raise HTTPException(status_code=400, detail="Файл пустой")
        
        if len(zip_data) > 100 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="Размер архива не должен превышать 100MB")
        
        result = tool_service.analyze_batch_images(
            zip_data, 
            max_images,
            confidence_threshold=confidence,
            iou_threshold=iou,
            save_annotated=save_annotated
        )
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при обработке архива: {str(e)}")