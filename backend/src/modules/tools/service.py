import zipfile
import io
import time
import os
from datetime import datetime
from typing import Dict, List
from collections import Counter

from ml.model import YOLOModel
from .schemas import SingleAnalysisResponse, BatchAnalysisResponse, ImageAnalysisResult, DetectionItem, AnalysisConfig, AnalysisResult

class ToolService:
    def __init__(self):
        self.model = YOLOModel()
        self.base_output_dir = "results"
        os.makedirs(self.base_output_dir, exist_ok=True)
        self.expected_tools_count = 11
    
    def convert_to_detection_items(self, detections_dict: List[Dict]) -> List[DetectionItem]:
        """Конвертирует словари детекций в объекты DetectionItem"""
        detection_items = []
        for detection in detections_dict:
            detection_item = DetectionItem(
                class_id=detection['class_id'],
                class_name=detection['class_name'],
                confidence=detection['confidence'],
                bbox=detection['bbox']
            )
            detection_items.append(detection_item)
        return detection_items
    
    def analyze_tool_completeness(self, detections: List[DetectionItem]) -> AnalysisResult:
        """Анализирует полноту набора инструментов"""
        detected_classes = set()
        class_counts = Counter()
        detected_tools_names = []
        
        for detection in detections:
            detected_classes.add(detection.class_id)
            class_counts[detection.class_id] += 1
            detected_tools_names.append(detection.class_name)
        
        # Все ожидаемые классы (0-10)
        expected_classes = set(range(self.expected_tools_count))
        
        missing_classes = expected_classes - detected_classes
        extra_classes = detected_classes - expected_classes
        
        # Определяем дубликаты (классы, которые встречаются больше 1 раза)
        duplicate_classes = {class_id for class_id, count in class_counts.items() if count > 1}
        
        # Преобразуем ID классов в названия инструментов
        missing_tools = [self.model.class_names[class_id] for class_id in missing_classes]
        extra_tools = [self.model.class_names[class_id] for class_id in extra_classes]
        duplicate_tools = [self.model.class_names[class_id] for class_id in duplicate_classes]
        
        # Определяем статус анализа с учетом дубликатов
        total_unique_classes = len(detected_classes)
        
        if total_unique_classes == self.expected_tools_count and not extra_classes and not duplicate_classes:
            status = "complete"
            message = "✅ Полный набор: все 11 инструментов обнаружены (без дубликатов)"
        elif total_unique_classes == self.expected_tools_count and not extra_classes and duplicate_classes:
            status = "duplicates"
            message = f"🔄 Полный набор с дубликатами: все 11 инструментов, но {len(duplicate_tools)} дублируются"
        elif missing_classes and not extra_classes and not duplicate_classes:
            status = "missing"
            message = f"⚠️ Неполный набор: отсутствует {len(missing_classes)} инструмент(ов)"
        elif missing_classes and duplicate_classes and not extra_classes:
            status = "missing_duplicates"
            message = f"🔄 Неполный набор с дубликатами: отсутствует {len(missing_tools)}, дублируется {len(duplicate_tools)}"
        elif extra_classes and not missing_classes:
            status = "extra"
            message = f"❌ Лишние инструменты: обнаружено {len(extra_classes)} лишних инструмент(ов)"
        elif extra_classes and missing_classes:
            status = "mixed"
            message = f"🔀 Смешанный результат: отсутствует {len(missing_tools)}, лишних {len(extra_tools)}"
        elif duplicate_classes and not missing_classes and not extra_classes:
            status = "duplicates_only"
            message = f"🔄 Дубликаты: все инструменты присутствуют, но {len(duplicate_tools)} дублируются"
        else:
            status = "unknown"
            message = f"❓ Неопределенный результат: {total_unique_classes} уникальных инструментов"
        
        # Для обратной совместимости объединяем дубликаты с extra_tools в некоторых случаях
        if status in ["duplicates", "duplicates_only", "missing_duplicates"]:
            # Добавляем информацию о дубликатах в extra_tools для отображения на фронтенде
            extra_tools.extend([f"{tool} (дубликат)" for tool in duplicate_tools])
        
        return AnalysisResult(
            status=status,
            total_detections=len(detections),
            expected_count=self.expected_tools_count,
            missing_tools=missing_tools,
            extra_tools=extra_tools,
            detected_tools=detected_tools_names,
            detections=detections,
            message=message
        )
    
    def analyze_single_image(self, image_data: bytes, filename: str = "image", 
                           confidence_threshold: float = 0.25, iou_threshold: float = 0.45) -> SingleAnalysisResponse:
        """Анализирует одно изображение на полноту набора инструментов"""
        try:
            # Получаем предсказания от модели и исходное изображение
            detections_dict, original_image = self.model.predict(image_data, confidence_threshold, iou_threshold)
            
            # Конвертируем словари в объекты DetectionItem
            detection_items = self.convert_to_detection_items(detections_dict)
            
            # Анализируем полноту набора
            analysis_result = self.analyze_tool_completeness(detection_items)
            
            # Сохраняем аннотированное изображение
            annotated_path = self.model.save_annotated_image(original_image, detections_dict, filename)
            
            # Добавляем информацию о использованных настройках
            config = AnalysisConfig(
                confidence_threshold=confidence_threshold,
                iou_threshold=iou_threshold,
                annotated_image_path=annotated_path
            )
            
            return SingleAnalysisResponse(
                status="success",
                analysis_result=analysis_result,
                config=config
            )
            
        except Exception as e:
            error_result = AnalysisResult(
                status="error",
                total_detections=0,
                message=f"Ошибка обработки: {str(e)}",
                detections=[]
            )
            
            return SingleAnalysisResponse(
                status="error",
                analysis_result=error_result,
                config=AnalysisConfig(
                    confidence_threshold=confidence_threshold,
                    iou_threshold=iou_threshold
                )
            )
    
    def analyze_batch_images(self, zip_data: bytes,
                           confidence_threshold: float = 0.25, iou_threshold: float = 0.45) -> BatchAnalysisResponse:
        """Анализирует группу изображений на полноту набора инструментов"""
        start_time = time.time()
        
        try:
            zip_file = zipfile.ZipFile(io.BytesIO(zip_data))
            file_list = zip_file.namelist()
            
            image_extensions = {'.jpg', '.jpeg', '.png'}
            image_files = [
                f for f in file_list 
                if any(f.lower().endswith(ext) for ext in image_extensions)
            ]
            
            if not image_files:
                return BatchAnalysisResponse(
                    status="error",
                    total_images=0,
                    processed_images=0,
                    results=[],
                    processing_time=0,
                    summary={}
                )
            
            # Создаем директорию для этой сессии
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            session_dir = os.path.join(self.base_output_dir, f"batch_session_{timestamp}")
            os.makedirs(session_dir, exist_ok=True)
            
            results = []
            processed_count = 0
            status_summary = {
                "complete": 0,
                "missing": 0,
                "extra": 0,
                "mixed": 0,
                "duplicates": 0,
                "duplicates_only": 0,
                "missing_duplicates": 0,
                "error": 0
            }
            
            for filename in image_files:
                try:
                    with zip_file.open(filename) as image_file:
                        image_data = image_file.read()
                    
                    detections_dict, original_image = self.model.predict(image_data, confidence_threshold, iou_threshold)
                    
                    # Конвертируем словари в объекты DetectionItem
                    detection_items = self.convert_to_detection_items(detections_dict)
                    
                    # Анализируем полноту набора
                    analysis_result = self.analyze_tool_completeness(detection_items)
                    
                    # Сохраняем аннотированное изображение
                    annotated_path = self.model.save_annotated_image(original_image, detections_dict, filename, session_dir)
                    
                    image_result = ImageAnalysisResult(
                        filename=filename,
                        analysis_result=analysis_result,
                        annotated_image_path=annotated_path
                    )
                    
                    results.append(image_result)
                    status_summary[analysis_result.status] += 1
                    processed_count += 1
                    
                    print(f"Обработано: {filename} - {analysis_result.status}")
                    
                except Exception as e:
                    error_result = AnalysisResult(
                        status="error",
                        total_detections=0,
                        message=f"Ошибка обработки: {str(e)}",
                        detections=[]
                    )
                    
                    error_image_result = ImageAnalysisResult(
                        filename=filename,
                        analysis_result=error_result
                    )
                    
                    results.append(error_image_result)
                    status_summary["error"] += 1
                    processed_count += 1
            
            processing_time = time.time() - start_time
            
            config = AnalysisConfig(
                confidence_threshold=confidence_threshold,
                iou_threshold=iou_threshold,
                output_directory=session_dir,
                total_annotated_images=len([r for r in results if r.annotated_image_path])
            )
            
            return BatchAnalysisResponse(
                status="completed",
                total_images=len(image_files),
                processed_images=processed_count,
                results=results,
                processing_time=round(processing_time, 2),
                summary=status_summary,
                config=config
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            return BatchAnalysisResponse(
                status="error",
                total_images=0,
                processed_images=0,
                results=[],
                processing_time=round(processing_time, 2),
                summary={}
            )

tool_service = ToolService()